"""
Fuzzy-matching metrics for DiagramLens benchmarking.

Uses difflib.SequenceMatcher at threshold 0.75 — never exact string matching.
Exact matching is wrong: "API Gateway" vs "api gateway" = miss (incorrect).
"""

from difflib import SequenceMatcher

FUZZY_THRESHOLD = 0.75


def fuzzy_match(name_a: str, name_b: str) -> bool:
    return SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio() >= FUZZY_THRESHOLD


def find_best_match(name: str, candidates: list[str]) -> tuple[str | None, float]:
    """Return (best_candidate, ratio) if ratio >= threshold, else (None, best_ratio)."""
    best, best_ratio = None, 0.0
    for c in candidates:
        r = SequenceMatcher(None, name.lower(), c.lower()).ratio()
        if r > best_ratio:
            best, best_ratio = c, r
    return (best, best_ratio) if best_ratio >= FUZZY_THRESHOLD else (None, best_ratio)


def score_components(
    extracted: list[str],
    ground_truth: list[str],
) -> dict:
    """
    Fuzzy precision/recall/F1 for component names.

    extracted:    list of component names from the pipeline
    ground_truth: list of component names from the annotation file
    """
    if not ground_truth:
        return _zero_metrics()

    tp_extracted: list[str] = []   # extracted names that matched a GT name
    fp: list[str] = []             # extracted names with no GT match (hallucinated)
    matched_gt: set[str] = set()   # GT names that were matched (to detect FN)

    for name in extracted:
        match, _ = find_best_match(name, ground_truth)
        if match:
            tp_extracted.append(name)
            matched_gt.add(match)
        else:
            fp.append(name)

    fn: list[str] = [gt for gt in ground_truth if gt not in matched_gt]

    tp = len(tp_extracted)
    precision = tp / len(extracted) if extracted else 0.0
    recall    = tp / len(ground_truth) if ground_truth else 0.0
    f1        = _f1(precision, recall)

    return {
        "precision":            round(precision, 4),
        "recall":               round(recall, 4),
        "f1":                   round(f1, 4),
        "tp":                   tp,
        "fp":                   len(fp),
        "fn":                   len(fn),
        "hallucinated_names":   fp,
        "missed_names":         fn,
    }


def score_connections(
    extracted_conns: list[dict],
    ground_truth_conns: list[dict],
    extracted_components: list[str],
    ground_truth_components: list[str],
) -> dict:
    """
    Fuzzy precision/recall/F1 for connections.

    Connection matching works by:
    1. Map each extracted source/target name to its nearest GT component name (fuzzy)
    2. Compare (mapped_source, mapped_target) pairs against GT (source, target) pairs
    """
    if not ground_truth_conns:
        return _zero_metrics()

    def resolve_name(name: str, candidates: list[str]) -> str | None:
        match, _ = find_best_match(name, candidates)
        return match

    # Build GT pair set using GT component names
    gt_pairs: set[tuple[str, str]] = {
        (c["source"], c["target"]) for c in ground_truth_conns
    }

    matched = 0
    for conn in extracted_conns:
        # Map extracted component names → nearest GT component names
        src = resolve_name(conn["source"], ground_truth_components)
        tgt = resolve_name(conn["target"], ground_truth_components)
        if src and tgt and (src, tgt) in gt_pairs:
            matched += 1

    precision = matched / len(extracted_conns) if extracted_conns else 0.0
    recall    = matched / len(ground_truth_conns) if ground_truth_conns else 0.0
    f1        = _f1(precision, recall)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp":        matched,
        "fp":        len(extracted_conns) - matched,
        "fn":        len(ground_truth_conns) - matched,
    }


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _zero_metrics() -> dict:
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0,
            "hallucinated_names": [], "missed_names": []}
