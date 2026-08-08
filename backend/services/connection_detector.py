"""
Connection detection for the HYBRID arm — LSD line segments, arrowhead
decoration, and line style.

Deliberately NOT shared with classical_pipeline._detect_connections. The two
arms previously called the same function, which meant their connection scores
were identical by construction and the three-way comparison was really a
two-way one on connections.

Why LSD instead of the classical arm's HoughLinesP:
  - Hough votes for infinitely long lines and then reconstructs segments, so it
    needs threshold/minLineLength/maxLineGap tuning per image and floods
    edge-dense regions with false positives. Published comparisons put it near
    30% recall against LSD's ~66%.
  - LSD is parameter-free, returns sub-pixel endpoints and per-segment width.
  - Missing connections, not inventing them, is this project's binding
    constraint, so recall is what matters.

Direction is MEASURED, never assumed. classical_pipeline deduplicated to
undirected pairs and then stamped directed=True; here an endpoint only becomes
a target if an arrowhead was actually found there.

Scope note: arrowhead PRESENCE (which drives `directed`) is the Tier A signal.
Fine-grained UML head classification (triangle vs diamond, filled vs hollow)
is best-effort and is reported qualitatively — see relationship_table.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from services.relationship_table import relationship_for

# ── Tuning ────────────────────────────────────────────────────────────────────
MIN_SEG_FRACTION   = 0.012  # segment shorter than this × image diagonal = noise.
                            # Measured: dropping from 0.020 raised unique linked
                            # component pairs 9->12 on a 21-component diagram,
                            # and recall is the binding constraint here.
MERGE_ANGLE_DEG    = 6.0    # collinear if headings within this
MERGE_OFFSET_PX    = 6.0    # ...and perpendicular offset within this
MERGE_GAP_PX       = 24.0   # ...and axial gap below this (labels break connectors)
SNAP_BASE_PX       = 18.0   # endpoint-to-box slack, before the adaptive term
SNAP_DIAG_FRACTION = 0.035  # plus this × image diagonal
TEXT_PAD           = 0.08   # grow text boxes by this before erasing them
TEXT_PAD_MAX_PX    = 4      # ...but never by more than this, or short
                            # connectors between dense labels are erased too
HEAD_WIDEN_RATIO   = 1.80   # stroke this many × its own mid-line width = head
HEAD_SAMPLE_PX     = 6.0    # measure widening within this × stroke of the tip
ARROW_MIN_STROKE   = 1.5
DASH_DUTY_SOLID    = 0.88   # ink duty cycle at/above this = solid
DASH_DUTY_MIN      = 0.25   # below this the "line" is probably not a line


@dataclass
class DetectedConnection:
    source_idx: int
    target_idx: int
    directed: bool
    line_style: str            # solid | dashed | unknown
    arrowhead_source: str      # none | open_arrow | filled_arrow | hollow_triangle | ...
    arrowhead_target: str
    relationship: str | None
    length: float


# ── Masks ─────────────────────────────────────────────────────────────────────
def _ink_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Binary mask of drawn strokes (255 = ink). Adaptive threshold handles the
    tinted panel backgrounds common in cloud vendor diagrams, where a global
    threshold erases whole regions."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if float(gray.mean()) < 110:
        gray = cv2.bitwise_not(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )


def _erase_filled(mask: np.ndarray, boxes, pad: float = 0.0, cap: int = 10**9) -> None:
    h, w = mask.shape[:2]
    for (x, y, bw, bh) in boxes:
        px = min(cap, int(bw * pad)) + (1 if pad else 0)
        py = min(cap, int(bh * pad)) + (1 if pad else 0)
        cv2.rectangle(mask, (max(0, x - px), max(0, y - py)),
                      (min(w, x + bw + px), min(h, y + bh + py)), 0, -1)


def _erase_border(mask: np.ndarray, boxes, thickness: int) -> None:
    """Erase only the outline of a box, leaving its interior untouched."""
    h, w = mask.shape[:2]
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(mask, (max(0, x), max(0, y)),
                      (min(w - 1, x + bw), min(h - 1, y + bh)), 0, thickness)


def _connector_mask(ink: np.ndarray,
                    component_boxes: list[tuple[int, int, int, int]],
                    container_boxes: list[tuple[int, int, int, int]],
                    text_boxes: list[tuple[int, int, int, int]],
                    stroke: float = 2.0) -> np.ndarray:
    """Ink reduced to connectors only.

    Components are erased WHOLE, border included — internal iconography and box
    outlines are both long straight strokes that otherwise dominate the segment
    list (on a test AWS diagram they produced most of 79 segments for ~13 real
    connectors). Connectors approach from outside, so nothing is lost:
    endpoint snapping re-associates them by proximity.

    Containers get only their BORDER erased. Filling them would blank
    everything nested inside — an Auto Scaling Group boundary would delete the
    very components and connectors it contains.

    Text padding is small and absolutely capped: a generous margin around a
    dense label cluster swallows the short connectors running between them.
    """
    out = ink.copy()
    _erase_filled(out, text_boxes, TEXT_PAD, cap=TEXT_PAD_MAX_PX)
    _erase_filled(out, component_boxes)
    _erase_border(out, container_boxes, thickness=int(max(3, 2 * stroke)))
    return out


def _stroke_width(mask: np.ndarray) -> float:
    """Median stroke thickness, from the distance transform of the ink."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    vals = dt[dt > 0.6]
    return float(max(ARROW_MIN_STROKE, 2.0 * np.median(vals))) if vals.size else 2.0


# ── Segment detection and merging ─────────────────────────────────────────────
def _detect_segments(mask: np.ndarray, min_len: float) -> list[tuple[float, float, float, float]]:
    """LSD, with HoughLinesP as a fallback if the build lacks it."""
    try:
        lsd = cv2.createLineSegmentDetector()
        raw = lsd.detect(mask)[0]
        segs = [] if raw is None else [tuple(map(float, s[0])) for s in raw]
    except Exception as exc:
        print(f"[connection_detector] LSD unavailable ({exc}); falling back to HoughLinesP")
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=45,
                                minLineLength=int(min_len), maxLineGap=12)
        segs = [] if lines is None else [tuple(map(float, l[0])) for l in lines]

    return [s for s in segs if math.dist(s[:2], s[2:]) >= min_len]


def _merge_collinear(segs: list[tuple[float, float, float, float]]
                     ) -> list[tuple[float, float, float, float]]:
    """Join segments lying on the same line with a small axial gap.

    Connectors are routinely broken by a crossing line, an edge label, or an
    anti-aliased gap. Left unmerged, each fragment is too short to reach a
    component and the connection is lost entirely.
    """
    buckets: dict[tuple[int, int], list] = {}
    for (x1, y1, x2, y2) in segs:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        # Perpendicular distance from origin identifies which parallel line it is
        rad = math.radians(angle)
        offset = -math.sin(rad) * x1 + math.cos(rad) * y1
        key = (int(angle / MERGE_ANGLE_DEG), int(offset / MERGE_OFFSET_PX))
        buckets.setdefault(key, []).append((x1, y1, x2, y2))

    merged: list[tuple[float, float, float, float]] = []
    for group in buckets.values():
        x1, y1, x2, y2 = group[0]
        rad = math.atan2(y2 - y1, x2 - x1)
        ux, uy = math.cos(rad), math.sin(rad)

        # Project every endpoint onto the shared direction, then sweep
        spans = []
        for (ax, ay, bx, by) in group:
            ta, tb = ax * ux + ay * uy, bx * ux + by * uy
            spans.append((min(ta, tb), max(ta, tb), (ax, ay, bx, by)))
        spans.sort()

        cur_lo, cur_hi, ref = spans[0]
        for lo, hi, seg in spans[1:]:
            if lo - cur_hi <= MERGE_GAP_PX:
                cur_hi = max(cur_hi, hi)
            else:
                merged.append(_span_to_seg(ref, ux, uy, cur_lo, cur_hi))
                cur_lo, cur_hi, ref = lo, hi, seg
        merged.append(_span_to_seg(ref, ux, uy, cur_lo, cur_hi))
    return merged


def _span_to_seg(ref, ux: float, uy: float, lo: float, hi: float
                 ) -> tuple[float, float, float, float]:
    """Rebuild a segment from an axial span, anchored on a reference point."""
    ax, ay = ref[0], ref[1]
    t0 = ax * ux + ay * uy
    return (ax + (lo - t0) * ux, ay + (lo - t0) * uy,
            ax + (hi - t0) * ux, ay + (hi - t0) * uy)


# ── Endpoint snapping ─────────────────────────────────────────────────────────
def _dist_to_box(px: float, py: float, box: tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    dx = max(x - px, 0.0, px - (x + w))
    dy = max(y - py, 0.0, py - (y + h))
    return math.hypot(dx, dy)


def _snap(px: float, py: float, boxes: list[tuple[int, int, int, int]],
          radius: float) -> int | None:
    """Nearest component by distance to its EDGE, not its centroid. Centroid
    distance systematically mis-assigns endpoints on large boxes, where the
    edge may be adjacent while the centre is far away."""
    best_idx, best = None, radius
    for i, box in enumerate(boxes):
        d = _dist_to_box(px, py, box)
        if d <= best:
            best, best_idx = d, i
    return best_idx


# ── Line style (B4) ───────────────────────────────────────────────────────────
def _line_style(ink: np.ndarray, seg: tuple[float, float, float, float]) -> str:
    """Solid vs dashed from the ink duty cycle along the segment. Pure
    geometry — the ink is already computed, so this is nearly free."""
    x1, y1, x2, y2 = seg
    length = math.dist((x1, y1), (x2, y2))
    if length < 8:
        return "unknown"

    h, w = ink.shape[:2]
    samples = int(min(length, 400))
    hits = 0
    for i in range(samples):
        t = i / (samples - 1)
        # Sample a small perpendicular window so a 1px wobble is not read as a gap
        cx, cy = int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)
        x0, y0 = max(0, cx - 1), max(0, cy - 1)
        patch = ink[y0:min(h, cy + 2), x0:min(w, cx + 2)]
        if patch.size and patch.max() > 0:
            hits += 1

    duty = hits / samples
    if duty >= DASH_DUTY_SOLID:
        return "solid"
    if duty >= DASH_DUTY_MIN:
        return "dashed"
    return "unknown"


# ── Arrowhead decoration (B3) ─────────────────────────────────────────────────
def _widthmap(arrows: np.ndarray, stroke: float) -> np.ndarray:
    """Distance transform dilated by roughly one stroke width.

    LSD detects the EDGES of a stroke, not its centreline, so a detected
    segment lies a pixel or two off the middle where the raw distance
    transform reads ~0. Taking a local maximum recovers the stroke's true
    half-width at each point along the segment.
    """
    dt = cv2.distanceTransform(arrows, cv2.DIST_L2, 3)
    r = int(max(1, round(stroke)))
    return cv2.dilate(dt, np.ones((2 * r + 1, 2 * r + 1), np.uint8))


def _sample_halfwidths(dt: np.ndarray, seg: tuple[float, float, float, float]) -> np.ndarray:
    """Stroke half-width sampled ALONG the segment.

    Sampling on the line rather than counting ink in a box window is what makes
    this usable on dense diagrams: a window near a component border or a
    crossing connector is full of unrelated ink, but the width at a point on
    the line reports only that stroke's own thickness.
    """
    x1, y1, x2, y2 = seg
    h, w = dt.shape[:2]
    n = int(max(8, min(math.dist((x1, y1), (x2, y2)), 400)))
    ts = np.linspace(0.0, 1.0, n)
    xs = np.clip((x1 + (x2 - x1) * ts).astype(int), 0, w - 1)
    ys = np.clip((y1 + (y2 - y1) * ts).astype(int), 0, h - 1)
    return dt[ys, xs]


def _head_shape(mask: np.ndarray, tip: tuple[float, float],
                head_halfwidth: float) -> str:
    """Name the decoration once widening has already established one is there.

    Best-effort. Arrowhead PRESENCE drives `directed` and is the Tier A signal;
    telling a filled triangle from a hollow diamond only feeds the optional
    `relationship` field, which is reported qualitatively.
    """
    h, w = mask.shape[:2]
    size = int(max(9, 4 * head_halfwidth))
    cx, cy = int(round(tip[0])), int(round(tip[1]))
    win = mask[max(0, cy - size):min(h, cy + size),
               max(0, cx - size):min(w, cx + size)]
    if win.size == 0:
        return "open_arrow"

    contours, _ = cv2.findContours(win, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "open_arrow"
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0 or area <= 0:
        return "open_arrow"

    fill = area / hull_area
    peri = cv2.arcLength(hull, True)
    verts = len(cv2.approxPolyDP(hull, 0.05 * peri, True)) if peri > 0 else 3
    peri_ratio = cv2.arcLength(contour, True) / max(1.0, peri)

    if fill >= 0.72:                                  # solid head
        return "filled_diamond" if verts == 4 else "filled_arrow"
    if peri_ratio < 1.35:                             # outline with a closed base
        return "hollow_diamond" if verts == 4 else "hollow_triangle"
    return "open_arrow"                               # two open barbs


def _endpoint_heads(arrows: np.ndarray, dt: np.ndarray,
                    seg: tuple[float, float, float, float],
                    stroke: float) -> tuple[str, str]:
    """Decoration at each end of a segment, from stroke widening.

    An arrowhead is the stroke getting locally fatter near its tip. Comparing
    each end against the SAME line's mid-section width makes the test
    self-normalising, so it holds for a hairline connector and a bold one alike.
    """
    x1, y1, x2, y2 = seg
    length = math.dist((x1, y1), (x2, y2))
    if length < 4:
        return "none", "none"

    # Probe PAST each tip. LSD traces the shaft and stops at the arrowhead's
    # base, so the head lies beyond the segment — sampling only within the span
    # measures the shaft twice and never sees the decoration.
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    ext = HEAD_SAMPLE_PX * stroke
    tip_s = (x1 - ux * ext, y1 - uy * ext)
    tip_t = (x2 + ux * ext, y2 + uy * ext)

    widths = _sample_halfwidths(dt, (*tip_s, *tip_t))
    n = widths.size
    if n < 6:
        return "none", "none"

    mid = float(np.median(widths[n // 3:2 * n // 3]))
    if mid <= 0.1:
        mid = max(0.5, stroke / 2.0)

    # Span BOTH sides of each endpoint. Whether LSD stops at the arrowhead's
    # base or traces on to its tip varies with the head's shape, so the
    # decoration can sit just inside or just outside the detected endpoint.
    k = max(2, int(n * ext / (length + 2 * ext)))
    k = min(2 * k, max(2, n // 3))
    near_start = float(widths[:k].max())
    near_end   = float(widths[-k:].max())

    # Aim the shape window at the middle of the decoration, not the shaft end.
    head_s = (_head_shape(arrows, (x1 - ux * ext * 0.5, y1 - uy * ext * 0.5), near_start)
              if near_start >= HEAD_WIDEN_RATIO * mid else "none")
    head_t = (_head_shape(arrows, (x2 + ux * ext * 0.5, y2 + uy * ext * 0.5), near_end)
              if near_end >= HEAD_WIDEN_RATIO * mid else "none")
    return head_s, head_t


def detect_connector_segments(
    img_rgb: np.ndarray,
    shape_boxes: list[tuple[int, int, int, int]],
    container_boxes: list[tuple[int, int, int, int]],
    text_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[float, float, float, float]]:
    """Connector segments derived from SHAPES alone, before components exist.

    The component builder needs these to recognise edge labels — text sitting
    on a connector (a UML interface name, a C4 "Sends e-mail using", an AWS
    protocol tag) is a label on that link, not a component in its own right.
    Shapes are known at that point; final component boxes are not.
    """
    try:
        ink = _ink_mask(img_rgb)
        stroke = _stroke_width(ink)
        mask = _connector_mask(ink, shape_boxes, container_boxes, text_boxes, stroke)
        diag = math.hypot(*ink.shape[:2])
        return _merge_collinear(_detect_segments(mask, MIN_SEG_FRACTION * diag))
    except Exception as exc:
        print(f"[connection_detector] segment pre-pass failed: {exc}")
        return []


MIDSPAN_LO, MIDSPAN_HI = 0.25, 0.75


def segment_midspan_to_box_distance(seg: tuple[float, float, float, float],
                                    box: tuple[int, int, int, int],
                                    samples: int = 10) -> float:
    """Closest approach between a segment's MIDDLE and a box.

    Two reasons for the mid-span restriction:

    Centre-to-segment distance is wrong for edge labels — "«API» HASP .Net" is
    a wide label whose centre sits ~50px from its connector while its nearest
    edge is ~10px away — so the measurement is box-to-segment.

    But box-to-whole-segment then flags real components, because every
    component has a connector ENDING at it. Measured: it cut recall from 0.73
    to 0.58. A link label sits alongside the middle of a line; a component sits
    at its end. Sampling only the middle separates the two.
    """
    x1, y1, x2, y2 = seg
    best = float("inf")
    span = MIDSPAN_HI - MIDSPAN_LO
    for i in range(samples):
        t = MIDSPAN_LO + span * (i / (samples - 1))
        d = _dist_to_box(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, box)
        if d < best:
            best = d
            if best == 0.0:
                break
    return best


def point_to_segment_distance(px: float, py: float,
                              seg: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 0:
        return math.dist((px, py), (x1, y1))
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
    return math.dist((px, py), (x1 + t * dx, y1 + t * dy))


# ── Entry point ───────────────────────────────────────────────────────────────
def detect_connections(
    img_rgb: np.ndarray,
    component_boxes: list[tuple[int, int, int, int]],
    component_ids: list[str],
    text_boxes: list[tuple[int, int, int, int]] | None = None,
    notation: str = "informal",
    container_boxes: list[tuple[int, int, int, int]] | None = None,
) -> list[DetectedConnection]:
    """Detect connections between components. Never raises — returns whatever
    was found."""
    if len(component_boxes) < 2:
        return []

    try:
        ink = _ink_mask(img_rgb)
        stroke = _stroke_width(ink)
        connectors = _connector_mask(ink, component_boxes,
                                     container_boxes or [], text_boxes or [], stroke)

        h, w = ink.shape[:2]
        diag = math.hypot(h, w)
        dt = _widthmap(connectors, stroke)

        segs = _merge_collinear(_detect_segments(connectors, MIN_SEG_FRACTION * diag))
        if not segs:
            return []

        radius = SNAP_BASE_PX + SNAP_DIAG_FRACTION * diag

        # Keep the longest evidence per component pair: a pair joined by one
        # long connector plus a stray fragment should not be scored twice.
        best: dict[tuple[int, int], DetectedConnection] = {}
        for seg in segs:
            x1, y1, x2, y2 = seg
            s = _snap(x1, y1, component_boxes, radius)
            t = _snap(x2, y2, component_boxes, radius)
            if s is None or t is None or s == t:
                continue

            head_s, head_t = _endpoint_heads(connectors, dt, seg, stroke)
            style  = _line_style(ink, seg)
            length = math.dist((x1, y1), (x2, y2))

            # Arrowhead at exactly one end defines the direction. Heads at both
            # ends or neither leaves the pair undirected rather than guessed.
            if head_t != "none" and head_s == "none":
                src, tgt, head = s, t, head_t
            elif head_s != "none" and head_t == "none":
                src, tgt, head = t, s, head_s
            else:
                src, tgt, head = s, t, "none"

            conn = DetectedConnection(
                source_idx=src,
                target_idx=tgt,
                directed=head != "none",
                line_style=style,
                arrowhead_source=head_s if src == s else head_t,
                arrowhead_target=head,
                relationship=relationship_for(notation, head, style),
                length=length,
            )

            key = (min(src, tgt), max(src, tgt))
            if key not in best or length > best[key].length:
                best[key] = conn

        return list(best.values())

    except Exception as exc:
        print(f"[connection_detector] failed: {exc}")
        return []
