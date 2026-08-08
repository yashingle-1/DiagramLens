"""
Fuzzy-matching metrics for DiagramLens benchmarking.

Matching is normalised and token-aware, NOT raw character similarity.
Raw SequenceMatcher systematically punishes correct extractions that differ
only by vendor prefix or acronym:

    "Application Load Balancer" vs "ALB"        -> 0.14  (wrong: same component)
    "Amazon S3 Artifacts"       vs "S3"         -> 0.19  (wrong: same component)
    "Amazon ECS"                vs "ECS Cluster"-> 0.38  (wrong: same component)

Gemini renames and normalises far more than the OCR-based pipelines do, so a
character-level metric penalises it for being verbose rather than for being
wrong. match_ratio() fixes that with acronym expansion, vendor-token removal,
and token-set comparison.

Both the raw and normalised scorers are exposed: report both in the
dissertation so the effect of normalisation is visible, not hidden.
"""

import re
from difflib import SequenceMatcher

FUZZY_THRESHOLD = 0.75
CONTAINMENT_SCORE = 0.90    # score when one token set fully contains the other
DISTINCT_TOKEN_MIN = 0.50   # similarity the disagreeing tokens must reach to still match

# Vendor / filler tokens carrying no discriminative meaning. Deliberately
# short: words like "server", "cluster", "service" DO distinguish components
# ("Web Server" vs "Web App") and must not be stripped.
_NOISE_TOKENS = {"aws", "amazon", "azure", "microsoft", "google", "gcp",
                 "the", "a", "an", "of"}

# Bidirectional acronym expansion. Applied to the whole normalised string, so
# "alb" -> "application load balancer" matches the spelled-out ground truth.
_ACRONYMS = {
    "alb": "application load balancer",
    "nlb": "network load balancer",
    "elb": "elastic load balancer",
    "lb":  "load balancer",
    "s3":  "simple storage service",
    "ec2": "elastic compute cloud",
    "ecs": "elastic container service",
    "eks": "elastic kubernetes service",
    "ecr": "elastic container registry",
    "rds": "relational database service",
    "sqs": "simple queue service",
    "sns": "simple notification service",
    "cdn": "content delivery network",
    "api gw": "api gateway",
    "db":  "database",
}


# ── Normalisation ─────────────────────────────────────────────────────────────
def _normalise(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def _tokens(name: str) -> set[str]:
    return {t for t in _normalise(name).split() if t not in _NOISE_TOKENS}


def _expand(tokens: set[str]) -> set[str]:
    """Expand acronyms per token, not per whole string. Whole-string expansion
    fires on "S3" but not on "Amazon S3 Artifacts", which leaves the two sides
    written in different vocabularies and guarantees a miss."""
    out: set[str] = set()
    for t in tokens:
        out.update(_ACRONYMS.get(t, t).split())
    return out - _NOISE_TOKENS


def _acronyms_in(tokens: set[str]) -> set[str]:
    return tokens & _ACRONYMS.keys()


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if (a or b) else 0.0


def _token_set_ratio(ta: set[str], tb: set[str]) -> float:
    """token_set_ratio on stdlib difflib: compare the shared-token string
    against each side's full token string, so extra words on one side cost far
    less than under plain character ratio."""
    if not ta or not tb:
        return 0.0
    inter  = " ".join(sorted(ta & tb))
    only_a, only_b = ta - tb, tb - ta
    s_a = " ".join(sorted(ta & tb) + sorted(only_a)).strip()
    s_b = " ".join(sorted(ta & tb) + sorted(only_b)).strip()

    # Both sides carry a token the other lacks, so the names genuinely
    # disagree. Anchoring on the shared tokens here would score "Auth Service"
    # against "User Service" as a match on the strength of "Service" alone —
    # compare the full strings instead.
    if only_a and only_b:
        return _ratio(s_a, s_b)
    return max(_ratio(inter, s_a), _ratio(inter, s_b))


def match_ratio(name_a: str, name_b: str) -> float:
    """Normalised similarity in [0, 1]. Use this everywhere, not SequenceMatcher."""
    ta, tb = _tokens(name_a), _tokens(name_b)

    # Different known acronyms are discriminative, not noise: "ECS Cluster" and
    # "EKS Cluster" share every other token and would otherwise match.
    acr_a, acr_b = _acronyms_in(ta), _acronyms_in(tb)
    if acr_a and acr_b and acr_a != acr_b:
        return 0.0

    score = max(_ratio(_normalise(name_a), _normalise(name_b)),
                _token_set_ratio(ta, tb))

    # Containment: every meaningful token of one name appears in the other
    # ("S3" inside "Amazon S3 Artifacts"). Checked on both the literal tokens
    # and their acronym expansions, so "ALB" reaches "Application Load
    # Balancer". Guarded on non-empty sets — a name that normalises to nothing
    # would otherwise contain, and match, everything.
    for sa, sb in ((ta, tb), (_expand(ta), _expand(tb))):
        if sa and sb and (sa <= sb or sb <= sa):
            return max(score, CONTAINMENT_SCORE)

    # Neither name contains the other, so both carry tokens the other lacks.
    # Judge them on the parts where they disagree: "Auth Service" and "User
    # Service" are 0.75 similar as raw strings purely because of the shared
    # word, but "auth" and "user" are unrelated. Without this, a long shared
    # suffix lets any two sibling components match.
    only_a, only_b = ta - tb, tb - ta
    if only_a and only_b:
        distinct = _ratio(" ".join(sorted(only_a)), " ".join(sorted(only_b)))
        if distinct < DISTINCT_TOKEN_MIN:
            return min(score, FUZZY_THRESHOLD - 0.01)
    return score


def raw_ratio(name_a: str, name_b: str) -> float:
    """Un-normalised character similarity — the original metric. Kept so the
    dissertation can report normalised vs raw side by side."""
    return SequenceMatcher(None, (name_a or "").lower(), (name_b or "").lower()).ratio()


# ── Matching primitives ───────────────────────────────────────────────────────
def fuzzy_match(name_a: str, name_b: str) -> bool:
    return match_ratio(name_a, name_b) >= FUZZY_THRESHOLD


def find_best_match(name: str, candidates: list[str]) -> tuple[str | None, float]:
    """Return (best_candidate, ratio) if ratio >= threshold, else (None, best_ratio)."""
    best, best_ratio = None, 0.0
    for c in candidates:
        r = match_ratio(name, c)
        if r > best_ratio:
            best, best_ratio = c, r
    return (best, best_ratio) if best_ratio >= FUZZY_THRESHOLD else (None, best_ratio)


# ── Component scoring ─────────────────────────────────────────────────────────
def score_components(
    extracted: list[str],
    ground_truth: list[str],
    ratio_fn=match_ratio,
) -> dict:
    """
    Fuzzy precision/recall/F1 for component names, with ONE-TO-ONE assignment.

    Each ground-truth name can be claimed by at most one extracted name.
    Without this, three extracted variants of one component ("Web Server",
    "Web Server 1", "Web Server 2") all count as true positives against a
    single ground-truth entry, letting recall exceed 1.0 and leaving
    tp/fp/fn mutually inconsistent.

    ratio_fn is injectable so the same scorer can produce raw-metric numbers
    for comparison (pass raw_ratio).
    """
    if not ground_truth:
        return _zero_metrics()

    def _best(name: str, pool: list[str]) -> tuple[str | None, float]:
        best, best_ratio = None, 0.0
        for c in pool:
            r = ratio_fn(name, c)
            if r > best_ratio:
                best, best_ratio = c, r
        return (best, best_ratio) if best_ratio >= FUZZY_THRESHOLD else (None, best_ratio)

    # Strongest matches claim their ground-truth entry first, so a weak
    # near-duplicate cannot steal the slot belonging to an exact match.
    ranked = sorted(
        ((_best(name, ground_truth)[1], name) for name in extracted),
        key=lambda x: -x[0],
    )

    unmatched_gt = list(ground_truth)
    tp_names: list[str] = []
    fp: list[str] = []

    for _, name in ranked:
        match, _r = _best(name, unmatched_gt)
        if match:
            unmatched_gt.remove(match)
            tp_names.append(name)
        else:
            fp.append(name)

    tp = len(tp_names)
    precision = tp / len(extracted) if extracted else 0.0
    recall    = tp / len(ground_truth)

    return {
        "precision":          round(precision, 4),
        "recall":             round(recall, 4),
        "f1":                 round(_f1(precision, recall), 4),
        "tp":                 tp,
        "fp":                 len(fp),
        "fn":                 len(unmatched_gt),
        "hallucinated_names": fp,
        "missed_names":       unmatched_gt,
    }


# ── Connection scoring ────────────────────────────────────────────────────────
def score_connections(
    extracted_conns: list[dict],
    ground_truth_conns: list[dict],
    extracted_components: list[str],
    ground_truth_components: list[str],
) -> dict:
    """
    Precision/recall/F1 for connections, scored two ways.

    IMPORTANT: extracted_conns must already carry component NAMES in
    source/target, not ids. The caller (routers/benchmark.py) resolves ids to
    names first — ground truth stores names, and fuzzy-matching "c1" against
    "CloudFront" never succeeds.

    Returns both:
      - directed   ("precision"/"recall"/"f1")  — topology AND arrow direction
      - undirected ("undirected_*")             — topology only
    Reporting them separately isolates whether a pipeline failed to see the
    line at all, or saw it but read the arrowhead wrong.
    """
    if not ground_truth_conns:
        return _zero_conn_metrics()

    gt_directed   = [(c["source"], c["target"]) for c in ground_truth_conns]
    gt_undirected = [tuple(sorted(p)) for p in gt_directed]

    resolved: list[tuple[str, str]] = []
    for conn in extracted_conns:
        src, _ = find_best_match(conn.get("source", ""), ground_truth_components)
        tgt, _ = find_best_match(conn.get("target", ""), ground_truth_components)
        if src and tgt and src != tgt:
            resolved.append((src, tgt))

    def _consume(pairs: list[tuple], gt_pairs: list[tuple]) -> int:
        """One-to-one: each ground-truth pair can be matched only once."""
        remaining = list(gt_pairs)
        tp = 0
        for p in pairs:
            if p in remaining:
                remaining.remove(p)
                tp += 1
        return tp

    d_tp = _consume(resolved, gt_directed)
    u_tp = _consume([tuple(sorted(p)) for p in resolved], gt_undirected)

    # Denominator is every emitted connection, including ones whose endpoints
    # never resolved — those are false positives, not free passes.
    n_ext, n_gt = len(extracted_conns), len(ground_truth_conns)

    d_p = d_tp / n_ext if n_ext else 0.0
    d_r = d_tp / n_gt
    u_p = u_tp / n_ext if n_ext else 0.0
    u_r = u_tp / n_gt

    return {
        "precision":            round(d_p, 4),
        "recall":               round(d_r, 4),
        "f1":                   round(_f1(d_p, d_r), 4),
        "tp":                   d_tp,
        "fp":                   n_ext - d_tp,
        "fn":                   n_gt - d_tp,
        "undirected_precision": round(u_p, 4),
        "undirected_recall":    round(u_r, 4),
        "undirected_f1":        round(_f1(u_p, u_r), 4),
        "undirected_tp":        u_tp,
    }


# ── Tier B: optional-field scoring ────────────────────────────────────────────
def score_optional_field(
    extracted: list[dict],
    ground_truth: list[dict],
    field: str,
    key: str = "name",
) -> dict | None:
    """
    Accuracy for an optional field (parent, relationship, stereotype, ...).

    Returns None when the ground truth does not annotate the field at all.

    A null ground-truth value means NOT ANNOTATED — never "annotated as
    absent". Treating null as a real empty value would make every unannotated
    optional field score as a false positive and would wreck the precision of
    all three pipelines the moment richer fields are emitted.
    """
    annotated = [g for g in ground_truth if g.get(field) is not None]
    if not annotated:
        return None

    gt_names = [g.get(key, "") for g in ground_truth]
    by_name = {g.get(key, ""): g for g in annotated}

    correct = 0
    for item in extracted:
        match, _ = find_best_match(item.get(key, ""), gt_names)
        gt_item = by_name.get(match) if match else None
        if gt_item is None:
            continue   # not annotated for this component — skip, do not penalise
        if fuzzy_match(str(item.get(field) or ""), str(gt_item.get(field) or "")):
            correct += 1

    return {
        "accuracy": round(correct / len(annotated), 4),
        "correct":  correct,
        "n":        len(annotated),   # always report n — subsets here are small
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _zero_metrics() -> dict:
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0,
            "hallucinated_names": [], "missed_names": []}


def _zero_conn_metrics() -> dict:
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0,
            "undirected_precision": 0.0, "undirected_recall": 0.0,
            "undirected_f1": 0.0, "undirected_tp": 0}
