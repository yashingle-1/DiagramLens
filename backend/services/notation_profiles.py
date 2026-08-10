"""
Per-notation extraction rules, declared as data rather than code.

Each notation draws components the same way structurally (a shape with a label)
but decorates them differently. Those decorations are what a general extractor
mistakes for components:

  UML  every box has compartments — a header («stereotype» + name) and a body
       ("artifacts" + a file list). Read naively that is 2-4 components per box.
       Interface names sit on the connectors as lollipop labels.
  C4   boxes carry name / [Container: Java] / a description sentence.
  AWS  the label sits OUTSIDE the glyph, below it.

Adding a notation means adding a dict entry here, not a new code path.

Rules are DECLARATIVE, not learned. Training a classifier for this would need a
labelled corpus that does not exist at this scale (13 annotated diagrams), and
would be trained on the same diagrams used for evaluation. A rule also states
its own reason — "rejected: matched a UML compartment keyword" — which serves
the explainability criterion the project is evaluated on.

Notation-specific rules apply ONLY when the classifier is confident. A
misclassified diagram would otherwise get the wrong rules applied silently.
Universal rules (watermarks, edge labels) always apply.
"""

from __future__ import annotations

import re

# Confidence the notation classifier must reach before notation-specific rules
# are used. classify_notation returns STRONG=0.9 for unambiguous evidence
# (guillemet stereotypes, C4 bracket tags, an icon-bank consensus).
PROFILE_MIN_CONFIDENCE = 0.7

# «application», <<interface>> — a UML stereotype, never part of the name.
# OCR rarely returns true guillemets: PP-OCRv5 reads them as CJK double angle
# brackets (《》), and other engines emit 〈〉, ‹›, or a doubled ASCII <<>>.
# Missing these meant the stereotype stayed glued to the name AND the notation
# classifier never fired, so no UML rules were applied at all.
GUILLEMET_OPEN  = "«《〈‹"
GUILLEMET_CLOSE = "»》〉›"
# Two alternatives, because the content class differs. Between real guillemets
# a stray ASCII '>' must be tolerated — OCR produced "《Provided component>》" —
# but inside <<...>> that character is the delimiter itself.
_STEREOTYPE = re.compile(
    rf"[{GUILLEMET_OPEN}]\s*([^{GUILLEMET_OPEN}{GUILLEMET_CLOSE}]+?)\s*[{GUILLEMET_CLOSE}]"
    rf"|<<\s*([^<>]+?)\s*>>"
)

# C4 element tag: "[Container: Java and Spring MVC]"
_C4_TAG = re.compile(r"\[\s*([^\]]+?)\s*\]")

# Attribution, not architecture. Deliberately keyed on the copyright marker
# rather than on a domain name: "yourApp.com" is a real component in the AWS
# reference diagrams, so a bare-domain rule would delete real data.
_WATERMARK = re.compile(
    r"(©|\(c\)\s|copyright|all rights reserved|creative commons)", re.IGNORECASE
)

# Compartment headers. These label a section of a box, never the box itself.
_UML_COMPARTMENTS = {
    "artifacts", "artifact", "attributes", "operations", "methods",
    "provided interfaces", "required interfaces", "realizations",
    "responsibilities", "tagged values",
}
# C4 prints the element kind as its own line under the name. Left in, it
# becomes the component name ("Amazon RDS Deployment Node").
_C4_COMPARTMENTS = {
    "description", "technology", "deployment node", "infrastructure node",
    "software system", "container", "component", "person", "database",
    "external system",
}


class Profile:
    __slots__ = ("strip_stereotype", "strip_bracket_tag", "compartments",
                 "stereotyped_text_outside_is_edge_label", "merge_compartments",
                 "geometric_edge_labels")

    def __init__(self, strip_stereotype=False, strip_bracket_tag=False,
                 compartments=frozenset(),
                 stereotyped_text_outside_is_edge_label=False,
                 merge_compartments=False, geometric_edge_labels=False):
        self.strip_stereotype = strip_stereotype
        self.strip_bracket_tag = strip_bracket_tag
        self.compartments = compartments
        self.stereotyped_text_outside_is_edge_label = (
            stereotyped_text_outside_is_edge_label)
        # Fusing stacked same-width rectangles is correct for compartmented
        # notations and harmful elsewhere: on cloud and informal diagrams it
        # merged genuinely separate components that happened to be drawn in a
        # column, cutting recall from 0.73 to 0.58.
        self.merge_compartments = merge_compartments
        # Treating any text near a connector's mid-span as that link's label.
        # Ablated per notation: it gains UML +0.10 F1 (interface names sit off
        # to one side of short stubs) but costs uber_system_design 0.72->0.54,
        # c4_big_bank 0.59->0.40 and aws_cicd 0.38->0.24, because in those
        # notations component labels also sit beside long connectors. It is
        # also the pipeline's worst latency outlier (21.8s vs 6.3s on uber).
        self.geometric_edge_labels = geometric_edge_labels


PROFILES: dict[str, Profile] = {
    # In UML a stereotype outside a box marks an interface on a connector
    # ("«API» HASP Java"), never a component.
    "uml": Profile(strip_stereotype=True, compartments=_UML_COMPARTMENTS,
                   stereotyped_text_outside_is_edge_label=True,
                   merge_compartments=True, geometric_edge_labels=True),
    "c4":  Profile(strip_stereotype=True, strip_bracket_tag=True,
                   compartments=_C4_COMPARTMENTS, merge_compartments=True),
    # Cloud vendor notations put the label outside the glyph and carry no
    # compartments; the defaults already handle them.
    "aws":      Profile(),
    "azure":    Profile(),
    "gcp":      Profile(),
    "informal": Profile(),
}

_DEFAULT = Profile()


def profile_for(notation: str, confidence: float = 1.0) -> Profile:
    if confidence < PROFILE_MIN_CONFIDENCE:
        return _DEFAULT
    return PROFILES.get(notation or "informal", _DEFAULT)


# ── Universal rules ───────────────────────────────────────────────────────────
def is_watermark(text: str) -> bool:
    return bool(_WATERMARK.search(text or ""))


# ── Profile-driven helpers ────────────────────────────────────────────────────
def split_stereotype(text: str) -> tuple[str, str | None]:
    """Return (name_without_stereotype, stereotype). Ground truth annotates
    "License Status", not "«application» License Status", so leaving the
    stereotype in the name costs a match."""
    found = _STEREOTYPE.search(text or "")
    if not found:
        return (text or "").strip(), None
    cleaned = _STEREOTYPE.sub(" ", text)
    marker = (found.group(1) or found.group(2) or "").strip(" <>")
    return re.sub(r"\s+", " ", cleaned).strip(), marker


def strip_bracket_tag(text: str) -> tuple[str, str | None]:
    """C4: "API Application [Container: Java]" -> ("API Application", "Container: Java")."""
    found = _C4_TAG.search(text or "")
    if not found:
        return (text or "").strip(), None
    cleaned = _C4_TAG.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip(), found.group(1).strip()


def is_compartment_header(text: str, profile: Profile) -> bool:
    return (text or "").strip().lower().rstrip(":") in profile.compartments


def strip_trailing_compartment(text: str, profile: Profile) -> str:
    """Drop a compartment header glued onto the end of a name.

    OCR often returns the header on the same line as the name it follows
    ("License Status artifacts"), so filtering whole header lines is not
    enough on its own.
    """
    words = (text or "").split()
    while words and words[-1].lower().rstrip(":") in profile.compartments:
        words.pop()
    return " ".join(words)


def clean_name(text: str, profile: Profile) -> tuple[str, str | None]:
    """Apply the profile's name rules. Returns (name, stereotype_or_tag)."""
    marker: str | None = None
    if profile.strip_stereotype:
        text, marker = split_stereotype(text)
    if profile.strip_bracket_tag:
        text, tag = strip_bracket_tag(text)
        marker = marker or tag
    return strip_trailing_compartment(text, profile), marker
