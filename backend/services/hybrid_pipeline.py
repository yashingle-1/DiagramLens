"""
Hybrid arm v2 — notation-adaptive proposal fusion.

NO generative model, NO LLM, NO API calls. Specialised discriminative models
(PaddleOCR PP-OCRv5 detection+recognition, CLIP image embeddings) plus
deterministic geometry.

Three proposers run over the same image and are merged into one component list:

    P1  icon_bank      CLIP image->image retrieval against official vendor icon
                       packs. Carries AWS / Azure / GCP, where identity lives in
                       the glyph and the label sits outside it.
    P2  shape_detector Contour geometry. Carries C4 / UML / informal, where
                       identity lives in a drawn box with the label inside.
    P3  text clusters  PaddleOCR word boxes clustered spatially. THE UNIVERSAL
                       FLOOR — every component in every notation carries a
                       label, so this always fires. P1 and P2 only upgrade
                       precision and typing.

Why this replaced SAM + CLIP-prompting:
  SAM's automatic mask generator is class-agnostic. On synthetic diagrams it
  over-segmented decorative gradients and icon sub-parts while under-
  distinguishing semantic units, so the old code needed a wall of filters
  (MAX_REGIONS / DUPLICATE_IOU / CONTAINMENT_RATIO) and still produced ~6
  components in ~137s. CLIP was scored against text prompts, which is out of
  distribution for abstract vector glyphs — the old code already overrode its
  verdict whenever a label keyword matched. Both findings are reportable.
  The previous implementation is preserved in hybrid_pipeline_v1.py and is
  still reachable via HYBRID_VERSION=v1 for the ablation table.

Never raises — returns whatever was found, even if partial.
"""

from __future__ import annotations

import asyncio
import io
import os
import time

import cv2
import numpy as np
from PIL import Image

from models.schemas import ArchitectureSchema, ComponentSchema, ConnectionSchema, ComponentPosition
from services import icon_bank, ocr_engine
from services.common import (
    classify_type,
    complexity,
    infer_arch_type,
    is_noise,
    looks_like_container,
)
from services.metrics import fuzzy_match
from services.connection_detector import (
    detect_connections, detect_connector_segments, segment_midspan_to_box_distance,
)
from services.notation_classifier import classify_notation, is_icon_centric
from services.notation_profiles import (
    clean_name, is_compartment_header, is_watermark, profile_for, split_stereotype,
)
from services.shape_detector import (
    Shape, are_compartments, children_of, compartment_ids, contains,
    detect_dashed_boundaries, detect_shapes, merge_compartments, split_containers,
)

# ── Tuning ────────────────────────────────────────────────────────────────────
TEXT_GAP_HORIZ = 1.4     # same-line merge if x-gap < factor × text height
TEXT_GAP_VERT  = 0.7     # stacked merge if y-gap < factor × text height
MERGE_IOU      = 0.60    # proposals overlapping this much are the same component
MAX_NAME_CHARS = 60      # longer than this is a description, not a label
MAX_NAME_WORDS = 8
MAX_COMPONENTS = 60      # guard against a pathological image; logged if hit
EDGE_LABEL_MIN_PX        = 10    # text this close to a connector is its label
EDGE_LABEL_HEIGHT_FACTOR = 1.0   # ...scaled by the label's own text height
# Compartment geometry lives in shape_detector — split_containers needs it too.

# Evidence ranking — richer evidence wins when two proposals overlap.
_EVIDENCE_RANK = {"icon+text": 4, "shape+text": 3, "icon": 2, "text": 1}


class _Proposal:
    __slots__ = ("box", "name", "type", "proposer", "confidence", "icon_match",
                 "icon_score", "shape_kind", "stereotype", "description")

    def __init__(self, box, name, type_, proposer, confidence=None,
                 icon_match=None, icon_score=None, shape_kind=None,
                 stereotype=None, description=None):
        self.box, self.name, self.type = box, name, type_
        self.proposer, self.confidence = proposer, confidence
        self.icon_match, self.icon_score = icon_match, icon_score
        self.shape_kind = shape_kind
        self.stereotype, self.description = stereotype, description


# ── Shape kind -> component type ──────────────────────────────────────────────
_SHAPE_TYPE = {
    "cylinder":     "database",
    "diamond":      "gateway",
    "hexagon":      "load_balancer",
    "ellipse":      "other",
    "rect":         "service",
    "rounded_rect": "service",
}


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ── P3: text clustering ───────────────────────────────────────────────────────
def _same_label(a: ocr_engine.OcrWord, b: ocr_engine.OcrWord) -> bool:
    """Two word boxes belong to one label. Gaps scale with text height so a
    heading and a nearby body label are not merged."""
    th = max(a.h, b.h)
    v_overlap = min(a.bottom, b.bottom) - max(a.y, b.y)
    h_overlap = min(a.right, b.right) - max(a.x, b.x)

    if v_overlap > 0.3 * min(a.h, b.h):                       # same line
        if max(a.x, b.x) - min(a.right, b.right) < TEXT_GAP_HORIZ * th:
            return True
    # Stacked. Centre alignment is required, not mere overlap: a wrapped label
    # ("Elastic Load" / "Balancing") is centred on itself, whereas a component
    # label and the group caption beneath it ("Web Server" / "Auto Scaling
    # Group") are offset. Overlap alone merges the two into one component.
    if h_overlap > 0.3 * min(a.w, b.w) and abs(a.cx - b.cx) < 0.35 * min(a.w, b.w):
        if max(a.y, b.y) - min(a.bottom, b.bottom) < TEXT_GAP_VERT * th:
            return True
    return False


def _cluster_words(words: list[ocr_engine.OcrWord]) -> list[list[ocr_engine.OcrWord]]:
    n = len(words)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if _same_label(words[i], words[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(words[i])
    return list(groups.values())


def _cluster_box(cluster: list[ocr_engine.OcrWord]) -> tuple[int, int, int, int]:
    x0 = min(w.x for w in cluster)
    y0 = min(w.y for w in cluster)
    x1 = max(w.right for w in cluster)
    y1 = max(w.bottom for w in cluster)
    return (x0, y0, x1 - x0, y1 - y0)


# ── Fusion ────────────────────────────────────────────────────────────────────
def _build_proposals(
    img_rgb: np.ndarray,
    words: list[ocr_engine.OcrWord],
    shapes: list[Shape],
    icon_matches: dict[int, icon_bank.IconMatch],
    notation: str = "informal",
    notation_confidence: float = 0.0,
    segments: list[tuple[float, float, float, float]] | None = None,
) -> list[_Proposal]:
    """Merge the three proposers. Text is the floor; shape and icon upgrade it."""
    profile = profile_for(notation, notation_confidence)

    clusters = _cluster_words(words)
    cluster_boxes = [_cluster_box(c) for c in clusters]
    cluster_names = [ocr_engine.reading_order(c) for c in clusters]
    claimed: set[int] = set()

    # Text lying on a connector is that link's label, not a component. Marked
    # up front so no proposer can claim it.
    edge_labels = _edge_label_indices(
        cluster_boxes, cluster_names, shapes, segments or [], profile)
    claimed.update(edge_labels)

    proposals: list[_Proposal] = []

    # UML/C4 compartments are themselves rectangles, so each would otherwise be
    # proposed as its own component ("artifacts", "license_service.dll", ...).
    compartments = compartment_ids(shapes)

    # Shape + text: label sits INSIDE the box (C4, UML, informal)
    for shape in shapes:
        if id(shape) in compartments:
            continue
        inside = [i for i, box in enumerate(cluster_boxes)
                  if i not in claimed and contains(shape.box, box, 0.7)]
        if not inside:
            continue

        # Does this shape enclose other shapes that are NOT its compartments?
        #
        # This is the discriminator, and it replaces counting text clusters.
        # A UML/C4 box holds several text clusters because it has compartments
        # and is ONE component. An Auto Scaling Group box also holds several,
        # but it is a group boundary. Counting clusters cannot tell them apart:
        # it either fuses the ASG's children into one node or explodes every
        # UML box into four.
        if _is_group_boundary(shape, shapes):
            continue

        # Notations WITHOUT compartments: several distinct labels inside one
        # detected rectangle means the rectangle is a group the detector failed
        # to recognise, not a component with sections. Reading it as one
        # component collapses several real components into one — measured as a
        # recall drop from 0.73 to 0.59. Leave them to the text proposer.
        if (not profile.merge_compartments and len(inside) > 1
                and not _is_one_label([cluster_boxes[i] for i in inside])):
            continue

        claimed.update(inside)
        ordered = sorted(inside, key=lambda i: cluster_boxes[i][1])
        name, marker, description = _read_compartments(
            [cluster_names[i] for i in ordered], profile
        )
        if not name or is_noise(name):
            continue
        proposals.append(_Proposal(
            box=shape.box, name=name[:MAX_NAME_CHARS],
            type_=_SHAPE_TYPE.get(shape.kind, "service"),
            proposer="shape+text", shape_kind=shape.kind,
            stereotype=marker, description=description,
        ))

    # Icon + text: label sits OUTSIDE, below or beside the glyph (AWS/Azure/GCP)
    for shape_idx, hit in icon_matches.items():
        box = shapes[shape_idx].box
        near = [i for i, cbox in enumerate(cluster_boxes)
                if i not in claimed and _near_icon(box, cbox)]
        marker = None
        if near:
            claimed.update(near)
            name, marker = clean_name(
                " ".join(cluster_names[i] for i in near).strip(), profile)
        else:
            name = hit.name          # glyph with no readable label
        proposals.append(_Proposal(
            box=box, name=name[:MAX_NAME_CHARS] or hit.name, type_=hit.type,
            proposer="icon+text" if near else "icon",
            confidence=round(hit.score, 3), icon_match=hit.name,
            icon_score=round(hit.score, 3), stereotype=marker,
        ))

    # Shape + label-below: the icon-centric pattern without an icon bank. The
    # glyph is a detected shape and its label sits underneath. Without this the
    # component's box is only its text, so connectors which attach to the
    # glyph never come within snapping distance and the link is lost.
    used_shapes = {id(s) for s in shapes if any(
        contains(s.box, cluster_boxes[i], 0.7) for i in claimed)}
    for shape in shapes:
        if id(shape) in used_shapes:
            continue
        near = [i for i, cbox in enumerate(cluster_boxes)
                if i not in claimed and _near_icon(shape.box, cbox)]
        if not near:
            continue
        claimed.update(near)
        # Same profile rules as the compartment path without this, bracket
        # tags survived here and produced names like "Amazon RDS
        # [Deployment Node]".
        name, marker = clean_name(
            " ".join(cluster_names[i] for i in near).strip(), profile)
        if not name or is_noise(name):
            continue
        proposals.append(_Proposal(
            box=_union(shape.box, *[cluster_boxes[i] for i in near]),
            name=name[:MAX_NAME_CHARS],
            type_=_SHAPE_TYPE.get(shape.kind, "service"),
            proposer="shape+text", shape_kind=shape.kind, stereotype=marker,
        ))

    # Text alone the floor. Any label not claimed above is still a component.
    for i, cluster in enumerate(clusters):
        if i in claimed:
            continue
        name, marker = clean_name(cluster_names[i], profile)
        if not name or is_noise(name) or _is_description(name):
            continue
        if is_watermark(name) or is_compartment_header(name, profile):
            continue
        proposals.append(_Proposal(
            box=cluster_boxes[i], name=name, type_=classify_type(name),
            proposer="text", stereotype=marker,
        ))

    return _nms(proposals)


def _is_group_boundary(shape: Shape, shapes: list[Shape]) -> bool:
    """Encloses other shapes that are NOT its own compartments."""
    children = children_of(shape, shapes)
    return bool(children) and not are_compartments(shape, children)


def _read_compartments(lines: list[str], profile) -> tuple[str, str | None, str | None]:
    """Turn a box's stacked text into (name, stereotype/tag, description).

    UML and C4 boxes are divided into compartments. The first non-header
    compartment is the component's name; the rest describe it. Joining them all
    produced names like "«application» License Status artifacts
    license_status.exe", which matches no ground-truth entry.
    """
    kept = [ln for ln in lines
            if ln and not is_watermark(ln) and not is_compartment_header(ln, profile)]
    if not kept:
        return "", None, None

    name, marker = clean_name(kept[0], profile)
    rest = " ".join(kept[1:]).strip() or None

    # A first compartment holding only a stereotype means the name wrapped to
    # the following line.
    if not name and len(kept) > 1:
        name, _ = clean_name(kept[1], profile)
        rest = " ".join(kept[2:]).strip() or None
    return name, marker, (rest[:200] if rest else None)


def _edge_label_indices(cluster_boxes, cluster_names, shapes, segments, profile) -> set[int]:
    """Text clusters sitting on a connector — link labels, not components.

    Covers UML interface names ("«API» HASP Java"), C4 relationship sentences
    and cloud protocol tags in one rule, because the geometry is the same in
    every notation. Only text OUTSIDE every shape is eligible, so a label
    inside a box adjacent to its own border is never mistaken for one.
    """
    found: set[int] = set()
    for i, box in enumerate(cluster_boxes):
        if any(contains(shape.box, box, 0.6) for shape in shapes):
            continue

        # UML puts an interface name on the connector, marked with a
        # stereotype. A stereotyped label OUTSIDE every box is therefore a
        # link label, whatever its distance from the line the lollipop
        # glyph breaks the segment and pushes the label clear of it.
        if profile.stereotyped_text_outside_is_edge_label and split_stereotype(
                cluster_names[i])[1]:
            found.add(i)
            continue

        if not profile.geometric_edge_labels:
            continue
        tol = max(EDGE_LABEL_MIN_PX, EDGE_LABEL_HEIGHT_FACTOR * box[3])
        if any(segment_midspan_to_box_distance(seg, box) <= tol for seg in segments):
            found.add(i)
    return found


def _union(*boxes: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


# Prose that describes a component or a relationship is not itself a component.
# C4 puts a description line inside every box and a sentence on every arrow
# ("Sends e-mail using", "Reads from and writes to"), which is the single
# largest source of false positives in the box-centric notations.
_DESCRIPTION_VERBS = (
    "sends", "send", "reads", "read", "writes", "write", "provides", "provide",
    "makes", "make", "uses", "use", "views", "view", "delivers", "deliver",
    "gets", "get", "calls", "call", "allows", "allow", "stores", "store",
    "returns", "return", "manages", "manage", "handles", "handle",
)


def _is_description(name: str) -> bool:
    stripped = name.strip()
    if len(stripped) > MAX_NAME_CHARS:
        return True
    words = stripped.split()
    if len(words) > MAX_NAME_WORDS:
        return True
    if words and words[0].lower() in _DESCRIPTION_VERBS:
        return True
    # A trailing full stop marks a sentence; component labels do not carry one.
    return stripped.endswith(".") and len(words) > 3


def _is_one_label(boxes: list[tuple[int, int, int, int]]) -> bool:
    """True if these text boxes are the wrapped lines of a SINGLE label:
    centred on each other and stacked without a gap."""
    ordered = sorted(boxes, key=lambda b: b[1])
    for (ax, ay, aw, ah), (bx, by, bw, bh) in zip(ordered, ordered[1:]):
        if abs((ax + aw / 2) - (bx + bw / 2)) > 0.35 * min(aw, bw):
            return False
        if by - (ay + ah) > TEXT_GAP_VERT * max(ah, bh):
            return False
    return True


def _near_icon(icon_box: tuple[int, int, int, int],
               text_box: tuple[int, int, int, int],
               gap_factor: float = 1.2) -> bool:
    """Icon-centric notations print the label under the glyph."""
    ix, iy, iw, ih = icon_box
    tx, ty, tw, th = text_box
    if tx + tw < ix - iw * 0.5 or tx > ix + iw * 1.5:
        return False
    return 0 <= ty - (iy + ih) <= gap_factor * ih


def _nms(proposals: list[_Proposal]) -> list[_Proposal]:
    """Keep the richest-evidence proposal where two overlap."""
    ordered = sorted(proposals, key=lambda p: -_EVIDENCE_RANK.get(p.proposer, 0))
    kept: list[_Proposal] = []
    for p in ordered:
        if any(_iou(p.box, k.box) >= MERGE_IOU for k in kept):
            continue
        kept.append(p)
    return kept


# ── Sync core ─────────────────────────────────────────────────────────────────
def _extract(image_bytes: bytes, session_id: str, start: float) -> ArchitectureSchema:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.array(pil)

    # Stage 1 one full-page OCR pass. Text is needed by every later stage.
    words = ocr_engine.read_page(img_rgb)

    # Stage 2 notation FIRST, so the rule profile is available to the
    # geometry stage. Stereotypes and bracket tags come from the OCR text, so
    # this needs no shapes; the icon-bank signal is folded in afterwards.
    ocr_text = " ".join(w.text for w in words)
    notation, notation_conf = classify_notation(img_rgb, ocr_text)
    profile = profile_for(notation, notation_conf)

    # Stage 3 geometry, split into components and group boundaries.
    # Dashed boundaries are found separately: contour analysis on a dashed
    # outline returns one contour per dash, never a rectangle.
    shapes = detect_shapes(img_rgb)
    if profile.merge_compartments:
        # Only for compartmented notations see Profile.merge_compartments.
        shapes = merge_compartments(shapes)
    shape_components, containers = split_containers(shapes)
    containers += [b for b in detect_dashed_boundaries(img_rgb)
                   if not any(_iou(b.box, c.box) >= 0.55 for c in containers)]

    # Stage 4 icon retrieval over the non-container shapes. A confident
    # vendor-icon consensus overrides the text-only notation guess.
    icon_matches: dict[int, icon_bank.IconMatch] = {}
    if icon_bank.available() and shape_components:
        crops = [img_rgb[y:y + h, x:x + w] for (x, y, w, h) in
                 (s.box for s in shape_components)]
        results = icon_bank.match_batch(crops)
        icon_matches = {i: m for i, m in enumerate(results) if m}
        if icon_matches:
            providers = [m.provider for m in icon_matches.values()]
            provider = max(set(providers), key=providers.count)
            notation, notation_conf = classify_notation(
                img_rgb, ocr_text, icon_bank.hit_rate(results), provider
            )

    # Stage 5 connector segments from shapes alone. Needed BEFORE components
    # exist so that text lying on a link can be recognised as that link's
    # label rather than proposed as a component.
    container_boxes = [c.box for c in containers]
    segments = detect_connector_segments(
        img_rgb, [s.box for s in shape_components], container_boxes,
        [w.box for w in words],
    )

    # Stage 6 fuse proposers into one component list, under the notation's
    # rule profile (see notation_profiles.py).
    proposals = _build_proposals(
        img_rgb, words, shape_components, icon_matches,
        notation, notation_conf, segments,
    )
    if len(proposals) > MAX_COMPONENTS:
        print(f"[hybrid_pipeline] capping {len(proposals)} proposals to {MAX_COMPONENTS}")
        proposals = proposals[:MAX_COMPONENTS]

    components: list[ComponentSchema] = []
    boxes: list[tuple[int, int, int, int]] = []
    for i, p in enumerate(proposals):
        # An explicit label beats a shape or icon guess ("Redis Cache" is a
        # cache regardless of what box it was drawn in).
        keyword_type = classify_type(p.name) if p.name else "other"
        final_type = keyword_type if keyword_type not in ("other", "service") else p.type
        x, y, w, h = p.box
        components.append(ComponentSchema(
            id=f"h{i + 1}", name=p.name or f"Component {i + 1}", type=final_type,
            confidence=p.confidence,
            position=ComponentPosition(x=float(x + w / 2), y=float(y + h / 2)),
            icon_match=p.icon_match, icon_score=p.icon_score, proposer=p.proposer,
            stereotype=p.stereotype, description=p.description,
        ))
        boxes.append(p.box)

    # Stage 6 containers kept as components with children linked via parent_id.
    # The old pipeline discarded these; VPC / subnet / AZ / C4 boundaries are
    # architectural knowledge and are what a knowledge graph needs.
    _attach_containers(components, boxes, containers, words, profile)

    # Stage 7 connections, snapped to each component's VISUAL EXTENT.
    #
    # A proposal's box is usually its label: in vendor diagrams the name sits
    # under the icon, so the box is the caption, not the thing. Connectors are
    # drawn to the ICON. Snapping to caption boxes left endpoints stranded 10-50px
    # from any component (measured: 0 cross-component links on aws_cicd_pipeline)
    # while the icon glyphs, never erased, were traced as if they were connectors.
    # Unioning each component with the shape it labels fixes both at once.
    text_boxes = [w.box for w in words]
    extents = _visual_extents(boxes, [s.box for s in shape_components])
    detected = detect_connections(
        img_rgb, extents, [c.id for c in components[:len(boxes)]], text_boxes, notation,
        container_boxes=container_boxes,
    )
    connections = _to_connection_schemas(detected, components)

    names = [c.name for c in components]
    return ArchitectureSchema(
        session_id=session_id,
        pipeline="hybrid",
        diagram_standard=notation,
        complexity=complexity(len(components)),
        arch_type=infer_arch_type([c.type for c in components]),
        components=components,
        connections=connections,
        response_time_ms=int((time.time() - start) * 1000),
        notation_confidence=notation_conf,
    )


EXTENT_GAP_RATIO   = 1.20   # caption within this × its own height of a shape
EXTENT_MAX_GROWTH  = 8.0    # never grow a component past this × its own area


def _visual_extents(boxes: list[tuple[int, int, int, int]],
                    shape_boxes: list[tuple[int, int, int, int]],
                    ) -> list[tuple[int, int, int, int]]:
    """Each component's drawn extent: its own box unioned with the shape it labels.

    A caption belongs to a shape when it sits directly under, over or inside it —
    horizontally overlapping and vertically close. The growth cap stops a stray
    caption from swallowing a whole panel it happens to sit near.
    """
    out: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x, y, w, h = box
        area = max(1, w * h)
        ext = box
        for (sx, sy, sw, sh) in shape_boxes:
            overlap = min(x + w, sx + sw) - max(x, sx)
            if overlap < 0.5 * min(w, sw):
                continue
            gap = max(sy - (y + h), y - (sy + sh))       # negative when overlapping
            if gap > EXTENT_GAP_RATIO * h:
                continue
            nx, ny = min(ext[0], sx), min(ext[1], sy)
            nw = max(ext[0] + ext[2], sx + sw) - nx
            nh = max(ext[1] + ext[3], sy + sh) - ny
            if nw * nh <= EXTENT_MAX_GROWTH * area:
                ext = (nx, ny, nw, nh)
        out.append(ext)
    return out


def _attach_containers(components: list[ComponentSchema],
                       boxes: list[tuple[int, int, int, int]],
                       containers: list[Shape],
                       words: list[ocr_engine.OcrWord],
                       profile) -> None:
    """Append group boundaries as components and set parent_id on their children.
    Smallest container wins, so a subnet inside a VPC is the direct parent."""
    if not containers:
        return
    start_idx = len(components)
    for n, container in enumerate(sorted(containers, key=lambda s: s.area)):
        # Boundary labels sit at the top-left of the region
        x, y, w, h = container.box
        # Boundary labels sit in a corner strip, but which corner depends on
        # the notation: AWS puts VPC/subnet names top-left, C4 deployment
        # diagrams put the node name bottom-left. Looking only at the top
        # left every C4 deployment node unnamed.
        strip = max(24, int(h * 0.18))
        header = (ocr_engine.text_for_box(words, (x, y, w, strip))
                  or ocr_engine.text_for_box(words, (x, y + h - strip, w, strip)))
        # An unnamed region is not evidence of a boundary. The old fallback
        # name ("Boundary 1") itself contained the word "boundary", so the
        # looks_like_container guard passed and every unlabelled rectangle was
        # emitted as a container.
        if not header or is_noise(header):
            continue
        name, _ = clean_name(header, profile)
        if not name or is_noise(name):
            continue
        name = name[:MAX_NAME_CHARS]

        # If a component already carries this name, the region is that
        # component's own boundary, not a separate node. A C4 deployment node
        # or an AWS VPC IS a component that contains others; emitting a second
        # entry for it double-counts and costs precision.
        existing = next(
            (c for c in components[:start_idx] if fuzzy_match(c.name, name)), None)
        if existing is not None:
            existing.is_container = True
            cid = existing.id
        else:
            cid = f"h{start_idx + n + 1}"
            components.append(ComponentSchema(
                id=cid, name=name, type="other", is_container=True,
                position=ComponentPosition(x=float(x + w / 2), y=float(y + h / 2)),
                proposer="shape",
            ))
        for child, box in zip(components[:start_idx], boxes):
            if child.id != cid and child.parent_id is None and contains(container.box, box):
                child.parent_id = cid


def _to_connection_schemas(detected, components: list[ComponentSchema]) -> list[ConnectionSchema]:
    out: list[ConnectionSchema] = []
    for n, d in enumerate(detected):
        if d.source_idx >= len(components) or d.target_idx >= len(components):
            continue
        src, tgt = components[d.source_idx], components[d.target_idx]
        out.append(ConnectionSchema(
            id=f"e{n + 1}", source=src.id, target=tgt.id,
            source_name=src.name, target_name=tgt.name,
            label="", directed=d.directed,
            line_style=d.line_style,
            arrowhead_source=d.arrowhead_source,
            arrowhead_target=d.arrowhead_target,
            relationship=d.relationship,
        ))
    return out


# ── Entry point ───────────────────────────────────────────────────────────────
async def run_hybrid_pipeline(image_bytes: bytes, session_id: str) -> ArchitectureSchema:
    """Full hybrid extraction. Never raises returns partial results on error."""
    start = time.time()

    if os.environ.get("HYBRID_VERSION", "v2").lower() == "v1":
        from services.hybrid_pipeline_v1 import run_hybrid_pipeline_v1
        return await run_hybrid_pipeline_v1(image_bytes, session_id)

    try:
        return await asyncio.to_thread(_extract, image_bytes, session_id, start)
    except Exception as e:
        print(f"[hybrid_pipeline] fallback: {e}")
        return ArchitectureSchema(
            session_id=session_id,
            pipeline="hybrid",
            diagram_standard="informal",
            complexity="low",
            arch_type="other",
            components=[],      # empty = honest zero; a fake "Unknown" would
            connections=[],     # count as a false positive in the benchmark
            response_time_ms=int((time.time() - start) * 1000),
            extraction_error=str(e)[:500],
        )
