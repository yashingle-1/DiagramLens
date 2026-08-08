"""
Maps (endpoint decoration, line style) to a relationship label, per notation.

Topology and direction are notation-independent — a line is a line, and an
arrowhead points the same way in UML as in an AWS diagram. What a decoration
MEANS is notation-dependent, so that part is a lookup rather than logic.

Unknown combinations return None. Never guess a relationship: the field is
reported qualitatively in the dissertation (n is far too small for a
per-notation relationship F1) and a fabricated value would be worse than a
null one.
"""

from __future__ import annotations

# (arrowhead_at_target, line_style) -> relationship
_UML = {
    ("hollow_triangle", "solid"):  "inheritance",
    ("hollow_triangle", "dashed"): "realisation",
    ("filled_diamond",  "solid"):  "composition",
    ("hollow_diamond",  "solid"):  "aggregation",
    ("open_arrow",      "dashed"): "dependency",
    ("open_arrow",      "solid"):  "association",
    ("filled_arrow",    "solid"):  "association",
    ("none",            "solid"):  "association",
}

_C4 = {
    ("open_arrow",   "dashed"): "async",
    ("filled_arrow", "dashed"): "async",
    ("open_arrow",   "solid"):  "uses",
    ("filled_arrow", "solid"):  "uses",
    ("none",         "solid"):  "uses",
}

# Cloud vendor notations carry protocol semantics in the line LABEL, not in the
# arrowhead shape, so every arrowed variant collapses to one relationship.
_FLOW = {
    ("open_arrow",      "solid"):  "data_flow",
    ("filled_arrow",    "solid"):  "data_flow",
    ("hollow_triangle", "solid"):  "data_flow",
    ("filled_diamond",  "solid"):  "data_flow",
    ("hollow_diamond",  "solid"):  "data_flow",
    ("open_arrow",      "dashed"): "async",
    ("filled_arrow",    "dashed"): "async",
    ("none",            "solid"):  "data_flow",
    ("none",            "dashed"): "async",
}

RELATIONSHIP_TABLE: dict[str, dict[tuple[str, str], str]] = {
    "uml":      _UML,
    "c4":       _C4,
    "aws":      _FLOW,
    "azure":    _FLOW,
    "gcp":      _FLOW,
    "informal": _FLOW,
}


def relationship_for(notation: str, arrowhead: str, line_style: str) -> str | None:
    table = RELATIONSHIP_TABLE.get(notation or "informal", _FLOW)
    return table.get((arrowhead or "none", line_style or "solid"))
