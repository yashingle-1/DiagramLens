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

# ── Skeleton tracing (replaces straight-segment detection) ────────────────────
DASH_BRIDGE_RATIO  = 2.2    # closing kernel = this × stroke ...
DASH_BRIDGE_DIAG   = 0.012  # ... or this × image diagonal, whichever is longer.
                            # Swept over {0.004, 0.012, 0.02} x snap radius on
                            # the full 37-diagram set. Larger values win on
                            # sparse C4/UML diagrams and lose on dense cloud
                            # ones, where they weld neighbouring connectors
                            # into one blob; 0.012 is the joint optimum.
TRACE_TURN_MAX_DEG = 72.0   # at a junction, continue only through a branch whose
                            # heading deviates less than this a crossing line is
                            # ~90deg away a routed elbow is 90deg but reached as
                            # a degree-2 bend, not a junction
TRACE_LOOKAHEAD    = 6      # pixels of branch followed before judging its heading
TRACE_MAX_STEPS    = 20000
SNAP_PATH_MIN_PX   = 12.0   # path-end snap radius, floor ...
SNAP_PATH_DIAG     = 0.05   # ... and this × image diagonal. Swept over
                            # {0.02, 0.035, 0.05}: connectors stop at the ICON
                            # while a component's box is often just its caption,
                            # so the gap to close is a layout distance, not a
                            # stroke-scaled one.
HEAD_OPEN_RATIO    = 1.7    # opening kernel = this × stroke; erases shafts, keeps heads
HEAD_MIN_AREA_R    = 1.1    # head blob area between (this × stroke)^2 ...
HEAD_MAX_AREA_R    = 12.0   # ... and (this × stroke)^2
HEAD_SNAP_RATIO    = 5.0    # head blob within this × stroke of a path end = that end's


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


# ── Skeleton path tracing ─────────────────────────────────────────────────────
# Straight-segment detection (LSD/Hough) cannot represent the connectors these
# diagrams actually use. AWS, C4 and UML connectors are orthogonally routed
# polylines: an L-shaped link is two collinear-incompatible segments, so one leg
# ends at a component and the other ends in whitespace at the elbow, and the
# link is lost. Measured on the hybrid arm: 138 merged segments for ~10 real
# connectors, of which 40 snapped both ends to the SAME component (box outlines
# and icon internals) and only 6 spanned a component pair.
#
# Tracing the skeleton instead follows a connector wherever it goes — elbows,
# curves, dashes bridged beforehand — and yields exactly two endpoints per
# connector, which is what endpoint snapping wants.

_NEIGHBOURS = ((-1, -1), (0, -1), (1, -1), (-1, 0),
               (1, 0), (-1, 1), (0, 1), (1, 1))


def _bridge_dashes(mask: np.ndarray, stroke: float, diag: float) -> np.ndarray:
    """Close dash gaps so a dashed dependency traces as one path.

    Closing is ORIENTED, not isotropic: four one-pixel-wide linear kernels at
    0/45/90/135 degrees, OR-ed together. A round kernel big enough to span a
    dash gap is also big enough to weld neighbouring parallel connectors into
    one blob; a linear kernel reaches along the line and nowhere else.

    Length is tied to the image diagonal because dash gaps scale with the
    rendering resolution, not with the stroke: on a 5000px C4 export the gaps
    are ~20px while the stroke is ~3px, so a stroke-scaled kernel bridged
    nothing and every dashed link fragmented into sub-minimum-length pieces.
    """
    L = int(max(5, round(max(DASH_BRIDGE_RATIO * stroke, DASH_BRIDGE_DIAG * diag))))
    L += 1 - (L % 2)
    eye = np.eye(L, dtype=np.uint8)
    kernels = (np.ones((1, L), np.uint8), np.ones((L, 1), np.uint8),
               eye, np.fliplr(eye).copy())
    out = np.zeros_like(mask)
    for k in kernels:
        out = cv2.bitwise_or(out, cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k))
    return out


def _skeletonise(mask: np.ndarray) -> np.ndarray | None:
    """One-pixel-wide centreline. Returns None if the build lacks ximgproc, in
    which case the caller falls back to the segment detector."""
    try:
        return cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except Exception as exc:
        print(f"[connection_detector] thinning unavailable ({exc}); using segments")
        return None


def _pixel_graph(skel: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    ys, xs = np.nonzero(skel)
    pts = set(zip(xs.tolist(), ys.tolist()))
    return {p: [q for q in ((p[0] + dx, p[1] + dy) for dx, dy in _NEIGHBOURS) if q in pts]
            for p in pts}


def _heading(a: tuple[int, int], b: tuple[int, int]) -> tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def _branch_heading(nbrs, cur, first, steps: int) -> tuple[float, float]:
    """Heading of a branch, judged `steps` pixels in rather than from the first
    pixel — 8-connected neighbours only resolve 45 degrees, which is not enough
    to tell a crossing line from a continuing one."""
    prev, node = cur, first
    for _ in range(steps):
        onward = [q for q in nbrs.get(node, ()) if q != prev]
        if len(onward) != 1:
            break
        prev, node = node, onward[0]
    return _heading(cur, node)


def _trace_paths(nbrs, min_len: float) -> list[list[tuple[int, int]]]:
    """Every skeleton path running between two free ends.

    Junctions are crossings far more often than they are real forks — connectors
    routinely cross each other and clip component borders — so a walk continues
    through the branch that best preserves its heading, and gives up if nothing
    continues straight enough. That keeps a crossed connector whole instead of
    cutting it into four stubs at the intersection.
    """
    ends = [p for p, n in nbrs.items() if len(n) == 1]
    paths: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for start in ends:
        prev, cur = start, nbrs[start][0]
        path = [start, cur]
        steps = 0
        while steps < TRACE_MAX_STEPS:
            steps += 1
            onward = [q for q in nbrs.get(cur, ()) if q != prev]
            if not onward:
                break
            if len(onward) == 1:
                nxt = onward[0]
            else:
                back = path[-min(len(path), TRACE_LOOKAHEAD)]
                hx, hy = _heading(back, cur)
                best, best_dot = None, math.cos(math.radians(TRACE_TURN_MAX_DEG))
                for q in onward:
                    qx, qy = _branch_heading(nbrs, cur, q, TRACE_LOOKAHEAD)
                    dot = hx * qx + hy * qy
                    if dot > best_dot:
                        best, best_dot = q, dot
                if best is None:
                    break
                nxt = best
            if nxt in path[-3:]:
                break
            path.append(nxt)
            prev, cur = cur, nxt

        if len(path) < 3:
            continue
        key = (min(path[0], path[-1]), max(path[0], path[-1]))
        if key in seen:          # same path walked from its other end
            continue
        # Path length as travelled, not end-to-end: an L-shaped connector is
        # long even when its endpoints are close in Euclidean terms.
        travelled = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
        if travelled < min_len:
            continue
        seen.add(key)
        paths.append(path)
    return paths


# ── Arrowheads as objects (replaces stroke-width sampling) ────────────────────
def _head_blobs(connectors: np.ndarray, stroke: float) -> list[tuple[float, float, tuple]]:
    k = int(max(3, round(HEAD_OPEN_RATIO * stroke)))
    k += 1 - (k % 2)
    opened = cv2.morphologyEx(connectors, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(opened, 8)

    lo, hi = (HEAD_MIN_AREA_R * stroke) ** 2, (HEAD_MAX_AREA_R * stroke) ** 2
    out = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if not (lo <= area <= hi):
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        # A head is compact. A long thin survivor is a thick border, not a head.
        if max(w, h) > 4.0 * max(1, min(w, h)):
            continue
        out.append((float(cents[i][0]), float(cents[i][1]), (x, y, w, h)))
    return out


def _head_at(point: tuple[int, int], blobs, radius: float,
             connectors: np.ndarray) -> str:
    """Decoration at a path end, or "none"."""
    best, best_d = None, radius
    for (cx, cy, box) in blobs:
        d = math.dist(point, (cx, cy))
        if d < best_d:
            best, best_d = (cx, cy, box), d
    if best is None:
        return "none"
    return _head_shape_box(connectors, best[2])


def _head_shape_box(mask: np.ndarray, box: tuple[int, int, int, int]) -> str:
    x, y, w, h = box
    pad = 2
    win = mask[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
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
    if fill >= 0.72:
        return "filled_diamond" if verts == 4 else "filled_arrow"
    return "hollow_diamond" if verts == 4 else "hollow_triangle"


def _path_style(ink: np.ndarray, path: list[tuple[int, int]]) -> str:
    """Solid vs dashed, measured along the traced path on the ORIGINAL ink.

    Sampling the pre-bridging ink is the point: the closing that makes a dashed
    line traceable would otherwise erase the very gaps being measured.
    """
    if len(path) < 8:
        return "unknown"
    h, w = ink.shape[:2]
    step = max(1, len(path) // 400)
    pts = path[::step]
    hits = 0
    for (px, py) in pts:
        x0, y0 = max(0, px - 1), max(0, py - 1)
        patch = ink[y0:min(h, py + 2), x0:min(w, px + 2)]
        if patch.size and patch.max() > 0:
            hits += 1
    duty = hits / len(pts)
    if duty >= DASH_DUTY_SOLID:
        return "solid"
    if duty >= DASH_DUTY_MIN:
        return "dashed"
    return "unknown"


def _connections_from_paths(
    ink: np.ndarray,
    connectors: np.ndarray,
    bridged: np.ndarray,
    component_boxes: list[tuple[int, int, int, int]],
    stroke: float,
    diag: float,
    notation: str,
) -> list[DetectedConnection] | None:
    skel = _skeletonise(bridged)
    if skel is None:
        return None

    paths = _trace_paths(_pixel_graph(skel), MIN_SEG_FRACTION * diag)
    if not paths:
        return []

    radius = max(SNAP_PATH_MIN_PX, SNAP_PATH_DIAG * diag)
    blobs = _head_blobs(connectors, stroke)
    head_radius = HEAD_SNAP_RATIO * stroke

    best: dict[tuple[int, int], DetectedConnection] = {}
    for path in paths:
        a, b = path[0], path[-1]
        s = _snap(a[0], a[1], component_boxes, radius)
        t = _snap(b[0], b[1], component_boxes, radius)
        if s is None or t is None or s == t:
            continue

        head_s = _head_at(a, blobs, head_radius, connectors)
        head_t = _head_at(b, blobs, head_radius, connectors)
        style = _path_style(ink, path)
        length = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))

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


# ── Entry point ───────────────────────────────────────────────────────────────
def detect_connections(
    img_rgb: np.ndarray,
    component_boxes: list[tuple[int, int, int, int]],
    component_ids: list[str],
    text_boxes: list[tuple[int, int, int, int]] | None = None,
    notation: str = "informal",
    container_boxes: list[tuple[int, int, int, int]] | None = None,
) -> list[DetectedConnection]:
    """Detect connections between components. Never raises returns whatever
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

        # Primary path: bridge dashes, skeletonise, trace. Falls back to the
        # straight-segment detector only if the OpenCV build has no thinning.
        bridged = _bridge_dashes(connectors, stroke, diag)
        traced = _connections_from_paths(ink, connectors, bridged, component_boxes,
                                         stroke, diag, notation)
        if traced is not None:
            return traced

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
