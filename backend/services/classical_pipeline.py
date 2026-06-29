"""
Classical CV pipeline — NO AI, NO LLM, NO EXTERNAL API CALLS.
Pure OpenCV + Tesseract only.

================================================================================
STRATEGY: TEXT-FIRST (not shape-first).
================================================================================
Architecture diagrams are *designed to be read*: every meaningful component
carries a text label. Box borders are decorative and unreliable (AWS icons have
the label BELOW the icon, C4/UML put it INSIDE a box, hand-drawn boxes are
broken). So instead of "find a box then OCR inside it", we:

1. Pre-process: grayscale, auto-invert dark-mode diagrams, upscale to Tesseract's
   sweet spot (~2000px long side) so small labels become legible.
2. OCR the WHOLE image at word granularity (image_to_data, sparse-text PSM).
   Map every word's box back to original-image coordinates.
3. Cluster spatially-close words with union-find -> each cluster is one component
   label (handles multi-word AND multi-line labels). Gaps are adaptive to text
   height, so two separate components don't merge.
4. Each surviving cluster = one component. Name = words in reading order,
   centroid = position, type = keyword classifier. confidence ALWAYS None.
5. Connections: build a text mask, erase text strokes from the Canny edge map so
   only connector lines survive, run HoughLinesP, snap each endpoint to the
   nearest component centroid (adaptive radius). Dedup undirected pairs.

Returns the SAME ArchitectureSchema as the Gemini pipeline.
Never raises — returns whatever was found, even if partial.
Always logs response_time_ms.
"""

from __future__ import annotations

import io
import os
import re
import time
import shutil
import numpy as np
import cv2
import pytesseract
from PIL import Image

from models.schemas import (
    ArchitectureSchema,
    ComponentSchema,
    ConnectionSchema,
    ComponentPosition,
)


# ── Tesseract binary auto-detection ───────────────────────────────────────────
def _locate_tesseract() -> str | None:
    """Find tesseract.exe so pytesseract works even when it is not on PATH."""
    env = os.environ.get("TESSERACT_CMD")
    if env and os.path.exists(env):
        return env
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


_TESS = _locate_tesseract()
if _TESS:
    pytesseract.pytesseract.tesseract_cmd = _TESS


# ── Tunable parameters ────────────────────────────────────────────────────────
OCR_TARGET_LONG_SIDE = 2000   # upscale (or downscale) so long side ~ this many px
OCR_MAX_LONG_SIDE    = 3200   # never blow OCR cost up beyond this
MIN_WORD_CONF        = 35     # tesseract per-word confidence floor
HORIZ_GAP_FACTOR     = 1.4    # same-line merge if x-gap < factor * text_height
VERT_GAP_FACTOR      = 0.7    # stacked merge if y-gap < factor * text_height
SNAP_RADIUS_FACTOR   = 0.6    # connection endpoint snaps within factor * comp_diag


# ── Type classification keywords ──────────────────────────────────────────────
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "database":      ["database", "db", "postgres", "postgresql", "mysql", "mongo", "mongodb",
                      "dynamo", "dynamodb", "sql", "sqlite", "oracle", "rds", "cassandra",
                      "aurora", "redshift", "datastore", "bigtable", "spanner"],
    "cache":         ["cache", "redis", "memcache", "memcached", "elasticache"],
    "queue":         ["queue", "kafka", "rabbitmq", "rabbit", "sqs", "sns", "pubsub", "pub/sub",
                      "eventbus", "event bus", "message bus", "broker", "kinesis", "celery",
                      "eventbridge", "topic"],
    "gateway":       ["gateway", "api gw", "api gateway", "apigw", "kong", "envoy", "ingress",
                      "proxy", "reverse proxy", "nginx"],
    "load_balancer": ["load balancer", "load-balancer", "loadbalancer", "alb", "elb", "nlb",
                      "balancer", "haproxy", "traffic manager"],
    "cdn":           ["cdn", "cloudfront", "akamai", "fastly", "edge", "content delivery"],
    "storage":       ["s3", "blob", "storage", "gcs", "file store", "filestore", "object store",
                      "bucket", "efs", "ebs", "minio"],
    "client":        ["client", "browser", "mobile", "user", "frontend", "front-end", "web app",
                      "webapp", "ios", "android", "spa", "ui"],
    "monitoring":    ["monitoring", "monitor", "prometheus", "grafana", "datadog", "cloudwatch",
                      "logging", "tracing", "jaeger", "kibana", "elk", "sentry"],
    "notification":  ["notification", "notify", "email", "smtp", "ses", "twilio", "push",
                      "firebase messaging", "fcm", "webhook"],
}

_AWS_KEYWORDS = {"aws", "ec2", "s3", "lambda", "cloudfront", "rds", "sqs", "sns", "eks", "ecs",
                 "fargate", "dynamodb", "elasticache", "alb", "elb", "cloudwatch", "route53",
                 "kinesis", "aurora", "redshift", "eventbridge", "cognito", "amazon"}
_C4_KEYWORDS  = {"person", "system", "container", "component", "bounded", "context", "c4",
                 "boundary"}
_UML_KEYWORDS = {"actor", "interface", "class", "abstract", "package", "stereotype", "extends",
                 "implements", "usecase", "use case"}

_STOPWORDS = {"the", "and", "for", "with", "via", "to", "of", "a", "an", "or", "in", "on",
              "by", "is", "are"}

# Short tokens (<=3 alpha chars) that are still meaningful components/acronyms.
# Anything <=3 chars NOT in here is treated as OCR garbage from icon glyphs.
_KNOWN_SHORT = {
    "elb", "alb", "nlb", "rds", "cdn", "ec2", "s3", "api", "vpc", "iam", "sns",
    "sqs", "ecs", "eks", "emr", "ses", "efs", "ebs", "az", "lb", "db", "ui", "ux",
    "app", "web", "dns", "ssl", "tls", "vm", "vms", "waf", "kms", "ec", "elk",
    "kafka", "id", "cli", "sdk", "gpu", "cpu", "iot", "mq", "fcm", "spa",
}


def _classify_type(text: str) -> str:
    lower = f" {text.lower()} "
    for component_type, keywords in _TYPE_KEYWORDS.items():
        for kw in keywords:
            if f" {kw} " in lower or lower.strip().endswith(kw) or lower.strip().startswith(kw):
                return component_type.strip()
    return "service"


def _infer_diagram_standard(names: list[str]) -> str:
    words = set(re.split(r"[\s,./_-]+", " ".join(names).lower()))
    if words & _AWS_KEYWORDS:
        return "aws"
    if words & _C4_KEYWORDS:
        return "c4"
    if words & _UML_KEYWORDS:
        return "uml"
    return "informal"


def _infer_arch_type(component_types: list[str]) -> str:
    types = component_types
    if "queue" in types:
        return "event_driven"
    if types.count("service") >= 3:
        return "microservices"
    return "other"


def _complexity(n: int) -> str:
    if n < 8:
        return "low"
    if n <= 14:
        return "medium"
    return "high"


# ── Pre-processing ────────────────────────────────────────────────────────────
def _preprocess(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (gray_original, gray_upscaled_for_ocr, scale).
    Auto-inverts dark-mode diagrams so OCR sees dark text on light background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect polarity: if the image is mostly dark, invert it.
    if float(gray.mean()) < 110:
        gray = cv2.bitwise_not(gray)

    h, w = gray.shape[:2]
    long_side = max(h, w)
    scale = OCR_TARGET_LONG_SIDE / long_side
    if long_side * scale > OCR_MAX_LONG_SIDE:
        scale = OCR_MAX_LONG_SIDE / long_side
    if abs(scale - 1.0) < 0.05:
        scale = 1.0

    if scale != 1.0:
        upscaled = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    else:
        upscaled = gray
    return gray, upscaled, scale


# ── OCR -> words ──────────────────────────────────────────────────────────────
class _Word:
    __slots__ = ("text", "conf", "l", "t", "w", "h")

    def __init__(self, text: str, conf: float, l: int, t: int, w: int, h: int):
        self.text, self.conf = text, conf
        self.l, self.t, self.w, self.h = l, t, w, h

    @property
    def cx(self) -> float: return self.l + self.w / 2
    @property
    def cy(self) -> float: return self.t + self.h / 2
    @property
    def right(self) -> int: return self.l + self.w
    @property
    def bottom(self) -> int: return self.t + self.h


def _ocr_words(ocr_img: np.ndarray, scale: float) -> list[_Word]:
    """Run sparse-text OCR and return word boxes in ORIGINAL image coordinates."""
    configs = ["--oem 3 --psm 11", "--oem 3 --psm 3"]
    best: list[_Word] = []
    for cfg in configs:
        try:
            data = pytesseract.image_to_data(
                ocr_img, output_type=pytesseract.Output.DICT, config=cfg
            )
        except Exception as exc:  # tesseract missing / failed — bail gracefully
            print(f"[classical_pipeline] OCR failed ({cfg}): {exc}")
            continue

        words: list[_Word] = []
        for i, raw_conf in enumerate(data["conf"]):
            try:
                conf = float(raw_conf)
            except (ValueError, TypeError):
                continue
            if conf < MIN_WORD_CONF:
                continue
            text = data["text"][i].strip()
            if not text or not re.search(r"[A-Za-z0-9]", text):
                continue
            # strip stray punctuation-only tokens
            if len(re.sub(r"[^A-Za-z0-9]", "", text)) < 2:
                continue
            l = int(data["left"][i] / scale)
            t = int(data["top"][i] / scale)
            w = int(data["width"][i] / scale)
            h = int(data["height"][i] / scale)
            words.append(_Word(text, conf, l, t, w, h))

        if len(words) > len(best):
            best = words
    return best


# ── Union-find clustering of words -> component labels ────────────────────────
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _same_component(a: _Word, b: _Word) -> bool:
    """True if two words belong to the same label (adaptive to text height)."""
    th = max(a.h, b.h)
    # vertical overlap fraction
    v_overlap = min(a.bottom, b.bottom) - max(a.t, b.t)
    h_overlap = min(a.right, b.right) - max(a.l, b.l)

    # Same line: vertically overlapping, small horizontal gap
    if v_overlap > 0.3 * min(a.h, b.h):
        gap = max(a.l, b.l) - min(a.right, b.right)
        if gap < HORIZ_GAP_FACTOR * th:
            return True

    # Stacked (multi-line label): horizontally overlapping, small vertical gap
    if h_overlap > 0.3 * min(a.w, b.w):
        gap = max(a.t, b.t) - min(a.bottom, b.bottom)
        if gap < VERT_GAP_FACTOR * th:
            return True
    return False


def _cluster_words(words: list[_Word]) -> list[list[_Word]]:
    n = len(words)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _same_component(words[i], words[j]):
                uf.union(i, j)
    groups: dict[int, list[_Word]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(words[i])
    return list(groups.values())


def _cluster_label(cluster: list[_Word]) -> str:
    """Join words in reading order (top rows first, left-to-right within a row)."""
    th = np.median([w.h for w in cluster]) or 1
    rows: list[list[_Word]] = []
    for w in sorted(cluster, key=lambda x: x.cy):
        placed = False
        for row in rows:
            if abs(row[0].cy - w.cy) < 0.6 * th:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])
    parts: list[str] = []
    for row in rows:
        row.sort(key=lambda x: x.l)
        parts.append(" ".join(w.text for w in row))
    name = " ".join(parts)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:80]


def _is_noise(name: str) -> bool:
    """Reject OCR garbage: icon glyphs ('ee','oO'), symbol soup ('�Kd=p)'), stopwords."""
    letters = re.sub(r"[^A-Za-z]", "", name)
    if len(letters) < 2:                       # pure numbers / symbols
        return True
    if name.lower() in _STOPWORDS:
        return True

    # Low alpha ratio = symbol soup ("�Kd=p)", "ic)")
    non_space = re.sub(r"\s", "", name)
    if non_space and len(letters) / len(non_space) < 0.5:
        return True

    # A real label needs at least one "substantial" token:
    # >=4 alpha chars, OR a known short acronym/word. Kills "ee", "ol Lae", "Ey", "fal".
    for tok in name.split():
        tok_alpha = re.sub(r"[^A-Za-z]", "", tok)
        if len(tok_alpha) >= 4 or tok_alpha.lower() in _KNOWN_SHORT:
            return False
    return True


# ── Connection detection ──────────────────────────────────────────────────────
def _nearest_component(point: tuple[float, float],
                       centroids: list[tuple[float, float]],
                       radius: float) -> int | None:
    px, py = point
    best_idx, best_dist = None, radius
    for i, (cx, cy) in enumerate(centroids):
        d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        if d <= best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def _detect_connections(
    gray: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    centroids: list[tuple[float, float]],
    component_ids: list[str],
) -> list[ConnectionSchema]:
    if len(centroids) < 2:
        return []

    h, w = gray.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)

    # Erase text/label regions so connector lines survive but glyph strokes don't.
    mask = np.zeros((h, w), np.uint8)
    for (x, y, bw, bh) in boxes:
        pad = int(0.15 * bh) + 2
        cv2.rectangle(mask, (max(0, x - pad), max(0, y - pad)),
                      (min(w, x + bw + pad), min(h, y + bh + pad)), 255, -1)
    edges[mask > 0] = 0

    diag = (h ** 2 + w ** 2) ** 0.5
    min_len = max(25, int(0.025 * diag))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45,
                            minLineLength=min_len, maxLineGap=12)
    if lines is None:
        return []

    # snap radius adaptive to median component size
    med_diag = float(np.median([(bw ** 2 + bh ** 2) ** 0.5 for (_, _, bw, bh) in boxes])) or 60
    radius = max(40.0, SNAP_RADIUS_FACTOR * med_diag + 0.04 * diag)

    seen: set[tuple[int, int]] = set()
    conns: list[ConnectionSchema] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        s = _nearest_component((x1, y1), centroids, radius)
        t = _nearest_component((x2, y2), centroids, radius)
        if s is None or t is None or s == t:
            continue
        pair = (min(s, t), max(s, t))
        if pair in seen:
            continue
        seen.add(pair)
        conns.append(ConnectionSchema(
            id=f"e{len(conns) + 1}",
            source=component_ids[s],
            target=component_ids[t],
            label="",
            directed=True,
        ))
    return conns


# ── Entry point ───────────────────────────────────────────────────────────────
async def run_classical_pipeline(image_bytes: bytes, session_id: str) -> ArchitectureSchema:
    """Full classical CV extraction. Never raises — returns partial results on error."""
    start = time.time()

    def _elapsed() -> int:
        return int((time.time() - start) * 1000)

    def _fallback(reason: str) -> ArchitectureSchema:
        print(f"[classical_pipeline] fallback: {reason}")
        return ArchitectureSchema(
            session_id=session_id, pipeline="classical",
            diagram_standard="informal", complexity="low", arch_type="other",
            components=[ComponentSchema(id="c1", name="Unknown", type="other", confidence=None)],
            connections=[], response_time_ms=_elapsed(), confidence_score=None,
        )

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return _fallback("cv2 could not decode image")

        gray, ocr_img, scale = _preprocess(img)

        words = _ocr_words(ocr_img, scale)
        if not words:
            return _fallback("OCR returned no usable words")

        # Cluster words -> component labels
        clusters = _cluster_words(words)

        components: list[ComponentSchema] = []
        boxes: list[tuple[int, int, int, int]] = []
        centroids: list[tuple[float, float]] = []

        for cluster in clusters:
            name = _cluster_label(cluster)
            if _is_noise(name):
                continue
            x0 = min(w.l for w in cluster)
            y0 = min(w.t for w in cluster)
            x1 = max(w.right for w in cluster)
            y1 = max(w.bottom for w in cluster)
            cid = f"c{len(components) + 1}"
            components.append(ComponentSchema(
                id=cid,
                name=name,
                type=_classify_type(name),
                confidence=None,
                position=ComponentPosition(x=float((x0 + x1) / 2), y=float((y0 + y1) / 2)),
            ))
            boxes.append((x0, y0, x1 - x0, y1 - y0))
            centroids.append(((x0 + x1) / 2, (y0 + y1) / 2))

        if not components:
            return _fallback("all clusters filtered as noise")

        component_ids = [c.id for c in components]
        connections = _detect_connections(gray, boxes, centroids, component_ids)

        names = [c.name for c in components]
        comp_types = [c.type for c in components]

        return ArchitectureSchema(
            session_id=session_id,
            pipeline="classical",
            diagram_standard=_infer_diagram_standard(names),
            complexity=_complexity(len(components)),
            arch_type=_infer_arch_type(comp_types),
            components=components,
            connections=connections,
            response_time_ms=_elapsed(),
            confidence_score=None,
        )

    except Exception as exc:
        return _fallback(f"unexpected error: {exc!r}")
