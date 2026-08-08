"""
Proposer P2 — geometric shape detection.

Carries the box-centric notations (C4, UML, informal), where the component
identity lives in a drawn rectangle with its label INSIDE. Icon-centric cloud
diagrams get little from this and are carried by P1 (icon bank) instead.

Replaces SAM's automatic mask generator in the hybrid arm. SAM is
class-agnostic: on synthetic diagrams it over-segments decorative gradients and
icon sub-parts while under-distinguishing semantic units, which is why the old
pipeline needed MAX_REGIONS / DUPLICATE_IOU / CONTAINMENT_RATIO filtering and
still returned ~6 components in ~137s. Contour analysis answers the actual
question — "is this a drawn box?" — in milliseconds.

Contour hierarchy is preserved so nesting (VPC > subnet > instance) can be
recovered rather than discarded.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

MIN_AREA_FRACTION = 0.0008   # smaller than this fraction of the image = noise
MAX_AREA_FRACTION = 0.60     # larger = page background
MIN_SIDE_PX       = 18
MAX_ASPECT        = 14.0     # thin slivers are rules/dividers, not components
DUPLICATE_IOU     = 0.80
CONTAINER_MIN_CHILDREN = 2   # encloses this many others = group boundary


@dataclass
class Shape:
    box:   tuple[int, int, int, int]     # x, y, w, h
    kind:  str                           # rect|rounded_rect|cylinder|ellipse|diamond|hexagon
    area:  float
    depth: int                           # contour nesting depth (0 = outermost)


def _kind_of(contour, box: tuple[int, int, int, int]) -> str | None:
    """Classify by polygon approximation. Returns None for shapes that are not
    plausible component outlines."""
    x, y, w, h = box
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    verts  = len(approx)
    area   = cv2.contourArea(contour)
    rect_fill = area / float(w * h) if w * h else 0.0

    # Circularity separates ellipses/cylinders from angular shapes
    circularity = 4 * np.pi * area / (peri * peri)

    if verts == 4:
        # Diamonds sit corner-up: their vertices cluster at the edge midpoints
        pts = approx.reshape(-1, 2)
        cx, cy = x + w / 2.0, y + h / 2.0
        on_axis = sum(
            1 for px, py in pts
            if abs(px - cx) < 0.18 * w or abs(py - cy) < 0.18 * h
        )
        return "diamond" if on_axis >= 3 else "rect"
    if verts == 6:
        return "hexagon"
    if verts > 6:
        if circularity > 0.78:
            return "ellipse"
        # Rounded rectangles approximate to many short edges but still fill
        # their bounding box like a rectangle.
        if rect_fill > 0.80:
            return "rounded_rect"
        # A cylinder (database) is a rectangle capped by ellipse arcs: it fills
        # most of its box and is markedly taller than a pure ellipse.
        if rect_fill > 0.68 and h > 0.6 * w:
            return "cylinder"
    return None


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int],
             ratio: float = 0.80) -> bool:
    """True if `ratio` of inner's area lies inside outer."""
    ox, oy, ow, oh = outer
    ix_, iy_, iw, ih = inner
    ix = max(0, min(ox + ow, ix_ + iw) - max(ox, ix_))
    iy = max(0, min(oy + oh, iy_ + ih) - max(oy, iy_))
    inner_area = iw * ih
    return bool(inner_area) and (ix * iy) / inner_area >= ratio


def detect_shapes(img_rgb: np.ndarray) -> list[Shape]:
    """Detect drawn component outlines. Never raises."""
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        if float(gray.mean()) < 110:
            gray = cv2.bitwise_not(gray)

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )
        # Close 1px gaps so a box with an anti-aliased corner still forms a
        # closed contour; without this many outlines break into arcs.
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            return []
        hierarchy = hierarchy[0]

        img_area = img_rgb.shape[0] * img_rgb.shape[1]
        depths = _depths(hierarchy)

        found: list[Shape] = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if not (MIN_AREA_FRACTION * img_area <= area <= MAX_AREA_FRACTION * img_area):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
                continue
            if max(w / h, h / w) > MAX_ASPECT:
                continue
            kind = _kind_of(contour, (x, y, w, h))
            if kind is None:
                continue
            found.append(Shape((x, y, w, h), kind, area, depths[i]))

        # Largest first, so an outer boundary is kept and its near-duplicate
        # inner outline (double-stroked borders) is dropped.
        found.sort(key=lambda s: -s.area)
        kept: list[Shape] = []
        for shape in found:
            if any(_iou(shape.box, k.box) >= DUPLICATE_IOU for k in kept):
                continue
            kept.append(shape)
        return kept

    except Exception as exc:
        print(f"[shape_detector] failed: {exc}")
        return []


DASH_CLOSE_KERNEL   = 11      # joins dashes into a continuous outline
DASH_MIN_FRACTION   = 0.02    # a group boundary is a substantial region
DASH_MAX_FRACTION   = 0.75


def detect_dashed_boundaries(img_rgb: np.ndarray) -> list[Shape]:
    """Find DASHED group boundaries (VPC, subnet, availability zone, C4
    boundary).

    detect_shapes cannot see these: findContours on a dashed outline returns
    one small contour per dash, never a rectangle. Closing with a large kernel
    first bridges the gaps. Run separately rather than raising the main close
    kernel, which would also fuse genuinely adjacent components.
    """
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        if float(gray.mean()) < 110:
            gray = cv2.bitwise_not(gray)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (DASH_CLOSE_KERNEL, DASH_CLOSE_KERNEL))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        img_area = img_rgb.shape[0] * img_rgb.shape[1]

        found: list[Shape] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            frac = (w * h) / img_area
            if not (DASH_MIN_FRACTION <= frac <= DASH_MAX_FRACTION):
                continue
            if max(w / h, h / w) > 8:
                continue
            # Boundaries are hollow: the closed outline covers only a small
            # part of the bounding box. A filled panel would cover most of it.
            if cv2.contourArea(contour) / float(w * h) > 0.55:
                continue
            found.append(Shape((x, y, w, h), "boundary", float(w * h), 0))

        found.sort(key=lambda s: -s.area)
        kept: list[Shape] = []
        for shape in found:
            if any(_iou(shape.box, k2.box) >= 0.55 for k2 in kept):
                continue
            kept.append(shape)
        return kept

    except Exception as exc:
        print(f"[shape_detector] dashed boundary detection failed: {exc}")
        return []


def _depths(hierarchy: np.ndarray) -> list[int]:
    """Nesting depth per contour, from the [next, prev, child, parent] table."""
    depths = []
    for i in range(len(hierarchy)):
        depth, parent, guard = 0, hierarchy[i][3], 0
        while parent != -1 and guard < 64:
            depth += 1
            parent = hierarchy[parent][3]
            guard += 1
        depths.append(depth)
    return depths


COMPARTMENT_WIDTH_RATIO = 0.85   # a compartment spans nearly its parent's width
COMPARTMENT_COVERAGE    = 0.70   # ...and stacked they fill most of its height


def children_of(shape: Shape, shapes: list[Shape]) -> list[Shape]:
    return [o for o in shapes if o is not shape and contains(shape.box, o.box)]


def are_compartments(shape: Shape, children: list[Shape]) -> bool:
    """True if these children are COMPARTMENTS of one component, not members of
    a group.

    A UML/C4 box is divided by horizontal rules, so every compartment is itself
    a detected rectangle. Structurally they tile the parent: each spans nearly
    its full width, and stacked they fill nearly its full height. Members of a
    group boundary (an Auto Scaling Group's servers, a VPC's subnets) sit inset
    with padding and rarely span the parent's width.
    """
    if not children:
        return False
    _, _, pw, ph = shape.box
    if pw <= 0 or ph <= 0:
        return False
    if not all(cw >= COMPARTMENT_WIDTH_RATIO * pw for (_, _, cw, _) in
               (c.box for c in children)):
        return False
    covered = sum(ch for (_, _, _, ch) in (c.box for c in children))
    return covered >= COMPARTMENT_COVERAGE * ph


COMPARTMENT_X_TOL     = 5     # px of horizontal offset allowed between stacked compartments
COMPARTMENT_W_RATIO   = 0.90  # ...and how closely their widths must agree
COMPARTMENT_GAP_PX    = 5     # compartments share a divider line, so the gap is ~0


def merge_compartments(shapes: list[Shape]) -> list[Shape]:
    """Fuse a component's compartments back into one shape.

    The nesting test cannot be used here, because a UML component box is often
    never detected at all: boxes are joined to each other by connectors, so the
    outermost contour spans most of the diagram and is rejected on area. What
    survives are the compartments, found as HOLES inside that blob. On the HASP
    licensing diagram that left 17 compartment rectangles and zero parents.

    Compartments are recoverable from geometry alone: they share a left edge
    and width, and stack with no gap because they share a divider line.
    Components in a group (AWS icons in an availability zone) sit apart with
    padding, so a near-zero gap threshold keeps them separate.
    """
    if not shapes:
        return []

    order = sorted(range(len(shapes)), key=lambda i: (shapes[i].box[0], shapes[i].box[1]))
    parent = list(range(len(shapes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_i, a in enumerate(order):
        ax, ay, aw, ah = shapes[a].box
        for b in order[a_i + 1:]:
            bx, by, bw, bh = shapes[b].box
            if abs(bx - ax) > COMPARTMENT_X_TOL:
                continue
            if min(aw, bw) < COMPARTMENT_W_RATIO * max(aw, bw):
                continue
            # Vertically adjacent in either order
            gap = by - (ay + ah) if by >= ay else ay - (by + bh)
            if -COMPARTMENT_GAP_PX <= gap <= COMPARTMENT_GAP_PX:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for i in range(len(shapes)):
        groups.setdefault(find(i), []).append(i)

    merged: list[Shape] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(shapes[members[0]])
            continue
        boxes = [shapes[i].box for i in members]
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        # Keep the topmost compartment's kind — the header carries the icon
        # and best represents the component.
        top = min(members, key=lambda i: shapes[i].box[1])
        merged.append(Shape((x0, y0, x1 - x0, y1 - y0), shapes[top].kind,
                            float((x1 - x0) * (y1 - y0)), shapes[top].depth))
    return merged


def compartment_ids(shapes: list[Shape]) -> set[int]:
    """ids of shapes that are a compartment of some other shape, so they are
    never proposed as components in their own right."""
    out: set[int] = set()
    for shape in shapes:
        if are_compartments(shape, children_of(shape, shapes)):
            out.update(id(c) for c in children_of(shape, shapes))
    return out


def split_containers(shapes: list[Shape]) -> tuple[list[Shape], list[Shape]]:
    """Partition into (components, containers).

    A shape enclosing CONTAINER_MIN_CHILDREN or more other shapes is a group
    boundary — VPC, subnet, availability zone, C4 system boundary. The old
    pipeline discarded these outright; they are retained here because the
    hierarchy is exactly what a knowledge graph or RAG index needs.

    A box whose children are its own COMPARTMENTS is not a container. Without
    that exception every UML component box (header + artifacts = 2 children)
    was reclassified as a boundary, and its compartments became components —
    turning a 9-component diagram into 22.
    """
    containers, components = [], []
    for shape in shapes:
        children = children_of(shape, shapes)
        is_container = (len(children) >= CONTAINER_MIN_CHILDREN
                        and not are_compartments(shape, children))
        (containers if is_container else components).append(shape)
    return components, containers
