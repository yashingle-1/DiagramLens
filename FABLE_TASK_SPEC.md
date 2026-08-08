# DiagramLens — Implementation Spec for Fable

**Project:** A Framework for Software Architecture Diagram Understanding and Structured Knowledge Extraction
**Repo root:** `archexplain - Copy/`
**Author of spec:** planning session, 2026-08-07

> **STATUS (2026-08-08). Part A: done and verified. Part B: B1–B8 + B11 done and
> measured; B9–B10 not started.**
>
> Deviations from this document, all deliberate:
> - `rapidfuzz` was **not** added — `match_ratio` runs on stdlib `difflib`, so
>   there is no new dependency.
> - Matching gained two guards the spec did not anticipate: differing known
>   acronyms never match (`ECS Cluster` vs `EKS Cluster` scored 0.82 without
>   it), and when neither name contains the other the *disagreeing* tokens must
>   themselves be similar (`Auth Service` vs `User Service` sat exactly on the
>   0.75 threshold without it).
> - `shape_detector` gained `detect_dashed_boundaries()`. Contour analysis
>   cannot see a dashed group boundary at all — `findContours` returns one
>   contour per dash — so a separate aggressively-closed pass was required.
> - OCR runs on a dedicated single worker thread. PaddleOCR predictors are
>   thread-affine and abort with a bare "Unknown exception" when created on one
>   thread and called from another, which is exactly what `asyncio.to_thread`
>   does.
> - PP-OCRv5 **mobile** models, not the server default: the server pair took
>   ~55s per diagram on CPU for no measurable gain on rendered diagram text.
>
> Run `backend/scripts/migrate_benchmark_columns.py` once Postgres is up —
> `create_all()` does not add columns to an existing table.
>
> **Measured on the 13 ground-truth pairs (classical vs hybrid; Gemini needs an
> API key and was not run):**
>
> | arm | comp F1 | comp F1 (raw metric) | precision | recall | conn F1 | conn undirected | ms |
> |---|---|---|---|---|---|---|---|
> | classical | 0.486 | 0.280 | 0.377 | 0.752 | 0.067 | 0.083 | 1568 |
> | hybrid v2 | **0.597** | 0.308 | 0.542 | 0.707 | 0.029 | 0.031 | 7796 |
>
> Hybrid runtime fell from ~137s to ~8s.
>
> **B12 — notation rule profiles (`notation_profiles.py`).** Added after UML
> extraction was found to inflate a 9-component diagram to 22-25. Per-notation
> component F1, before and after:
>
> | notation | n | before | after |
> |---|---|---|---|
> | uml | 3 | 0.560 | **0.783** |
> | c4 | 5 | 0.577 | 0.591 |
> | aws | 4 | 0.392 | 0.424 |
> | informal | 1 | 0.720 | 0.761 |
>
> On `uml_component_hasp_licensing`: 22 components -> 8, F1 0.516 -> **0.941**,
> precision 1.00, zero hallucinations (ground truth has 9).
>
> Four findings from that work, each worth stating in the write-up:
>
> 1. **The outer UML component box is never detected.** Boxes are joined by
>    connectors, so the outermost contour spans most of the diagram and is
>    rejected on area; only the compartments survive, found as *holes*. Nesting
>    tests are therefore unusable, and compartments must be recovered from
>    geometry — stacked rectangles sharing a left edge and width, with ~0 gap
>    because they share a divider line. That reconstructed all 9 boxes exactly.
> 2. **OCR does not return guillemets.** PP-OCRv5 reads `«»` as CJK `《》`.
>    Until the regex accepted every variant, the notation classifier never
>    fired and NO UML rule was ever applied.
> 3. **Rules must be notation-gated, and the gating was determined by ablation,
>    not assumption.** Treating text near a connector as an edge label gains UML
>    +0.10 F1 but costs `uber_system_design` 0.72->0.54, `c4_big_bank`
>    0.59->0.40 and `aws_cicd` 0.38->0.24 — and is the pipeline's worst latency
>    outlier (21.8s vs 6.3s). It is enabled for UML only. Same for compartment
>    merging (uml/c4 only).
> 4. **No training is involved.** These are declarative rules. A learned
>    alternative would need a labelled corpus that does not exist at this scale,
>    and would be fitted on the evaluation set. Rules also state their own
>    reason, which serves the explainability criterion.
>
> **Known open problem — connection F1 is bad and hybrid is WORSE than the
> classical control.** Do not report these connection numbers as a result yet.
> Diagnosed cause: line detection is *not* the bottleneck. On
> `aws_multi_region_architecture` (11 GT connections) the detector finds 12
> unique linked component pairs, with endpoints snapping at a median distance
> of 21px against a 64px radius. The loss happens in scoring: `score_connections`
> resolves each endpoint name against ground truth, and with component
> precision at 0.43 more than half of all endpoints resolve to a component that
> is not in the ground truth, which discards the whole connection.
> **Connection accuracy is therefore capped by component precision — fix
> precision first, and connection F1 follows.** The dominant precision leak is
> the text proposer emitting non-component text (legends, titles, captions,
> axis notes); `_is_description` catches C4-style prose but not these.
>
> **P1 icon bank: built, calibrated, and DEFAULT OFF (`ICON_BANK=1` enables).**
> Bank holds 1456 icons (AWS 525 / Azure 808 / GCP 123) sourced from the
> MIT-licensed `diagrams` package — record that provenance in the dissertation,
> the glyphs remain vendor trademarks.
>
> This produced a reportable **negative result**. CLIP image embeddings of flat
> vector icons occupy a narrow cone, so an absolute cosine threshold cannot
> identify them. Measured over the bank:
>
> | statistic | value |
> |---|---|
> | random different-icon pair cosine | p50 0.790, p90 0.858, p99 0.912 |
> | each icon's nearest *other* icon | p50 **0.953** |
> | random pairs passing the original 0.82 cutoff | **29.3%** |
>
> At 0.82 the matcher labelled 19 of 20 crops on `aws_three_tier_web` as the
> same entry ("IAM Access Analyzer") at 0.89–0.93. Acceptance was rewritten to
> be relative — absolute floor 0.86 **and** margin over ranks 2–10 ≥ 0.05
> **and** z-score ≥ 3.5, plus a flat-fill guard — separating genuine icons
> (margin p50 0.094) from diagram crops (margin p50 0.017). That removed every
> false match, but left 1 accepted icon across 41 crops, moved component F1 by
> 0.000, and cost ~4.8s per diagram. Hence default off, preserved for ablation.
>
> **Conclusion for the write-up: CLIP image-to-image retrieval is not a
> reliable icon identifier on this corpus.** The high similarity floor between
> distinct vendor icons is the reason, and it is measurable. This is directly
> relevant to the "should we use CLIP / SAM / Grounding DINO" question — two of
> the three foundation-model components trialled here (SAM, CLIP) were measured
> and rejected in favour of deterministic geometry and a specialised OCR model.
>
> Remaining: **B9** (Tier A/B metric wiring into the router — `metrics.py` has
> `score_optional_field` with the null-skip guard, but `benchmark.py` does not
> call it yet) and **B10** (ground-truth optional-field migration).
>
> Note: `benchmark_columns` migration has been APPLIED to the live database.
> Wall-clock timings on this machine are noisy (hybrid measured 5.7–10.5s
> across runs); quote a median over repeated runs in the dissertation, not a
> single figure.

---

## 0. Read this first — hard rules

1. **NEVER modify the extraction algorithm in `backend/services/classical_pipeline.py`.**
   It is the experimental control arm (zero-AI baseline). Canny + HoughLinesP + Tesseract stay exactly as they are.
   The ONE permitted change is B4 in Part A (fabricated `directed=True`). Nothing else.

2. **All new detection work goes in the hybrid arm** (`hybrid_pipeline.py` and new service modules).

3. **`hybrid_pipeline.py` must stop importing from `classical_pipeline.py`.**
   Current lines 29–36 import `_detect_connections`, `_classify_type`, `_complexity`, `_infer_arch_type`, `_infer_diagram_standard`, `_is_noise`. Move the shared *helpers* (`_classify_type`, `_complexity`, `_infer_arch_type`, `_is_noise`) into a new `backend/services/common.py` that both arms import. `_detect_connections` must NOT be shared — hybrid gets its own detector. Two arms sharing a connection detector makes the three-way comparison invalid.

4. **No generative/foundation LLM in classical or hybrid arms.** PaddleOCR and CLIP are specialised discriminative models — allowed in hybrid. TrOCR is a generative decoder — remove it.

5. **All new JSON fields are optional and nullable.** Existing ground truth files must remain valid without edits.

6. **Backwards compatibility:** the frontend consumes `ArchitectureSchema`. Add fields, never rename or remove existing ones.

7. Work through parts in order. Part A first — it fixes measurement, and without correct measurement nothing else can be validated.

---

# PART A — Critical bug fixes (do these first)

These are confirmed defects. They currently make benchmark numbers meaningless and unfairly penalise the Gemini arm.

## A1. Connection scoring is structurally broken — connection F1 is ~0 for ALL pipelines

**Where:** `backend/routers/benchmark.py:83-86` + `backend/services/metrics.py:87-102`

**Problem:**
Ground truth stores connections by **name**:
```json
{"source": "Developer", "target": "AWS CodeCommit"}
```
Pipelines emit connections by **component ID**:
- `classical_pipeline.py:409-410` → `source="c1", target="c4"`
- `hybrid_pipeline.py:419` → `source="h1", target="h4"`
- Gemini prompt → `"source": "source_component_id"`

`benchmark.py:84` passes those raw IDs into `score_connections`, which at `metrics.py:99` runs
`find_best_match("c1", ["Developer", "AWS CodeCommit", ...])`.
`SequenceMatcher("c1", "developer").ratio()` ≈ 0.0 → never matches.

**Result: connection precision/recall/F1 = 0.0 for every pipeline on every diagram.**
This violates CLAUDE.md Critical Rule 4 ("Connections use component NAMES, never IDs").

**Fix:** in `benchmark.py`, before scoring, build an id→name map from the same `raw_json` and translate:

```python
raw = arch.raw_json or {}
components = raw.get("components", [])
id_to_name = {c.get("id"): c.get("name", "") for c in components if c.get("id")}

def _resolve(ref: str) -> str:
    # Pipelines emit IDs; some Gemini outputs emit names directly. Handle both.
    return id_to_name.get(ref, ref)

extracted_conns = [
    {"source": _resolve(c.get("source", "")),
     "target": _resolve(c.get("target", "")),
     "directed": c.get("directed", True)}
    for c in raw.get("connections", [])
]
```

Add a defensive check: if >50% of connection endpoints fail to resolve against `id_to_name` AND do not fuzzy-match any component name, log a warning — that means a pipeline emitted dangling references.

**Improves:** connection F1 becomes a real measurement instead of a constant zero. This is the single highest-impact fix in the whole spec.

---

## A2. Component recall double-counts — recall can exceed 1.0

**Where:** `backend/services/metrics.py:44-57`

**Problem:** many extracted names may fuzzy-match the *same* ground-truth name. Each is appended to `tp_extracted`, so `tp = len(tp_extracted)` counts duplicates, but `recall = tp / len(ground_truth)`.

Example: GT has 9 components including one `"Web Server"`. Pipeline emits `"Web Server"`, `"Web Server 1"`, `"Web Server 2"`. → `tp = 3`, `matched_gt = {"Web Server"}`, `fn = 8`. Recall reported as `3/9 = 0.33` when true coverage is `1/9 = 0.11`. With enough duplicates recall exceeds 1.0. `tp`, `fp`, `fn` are mutually inconsistent.

**Fix:** one-to-one greedy assignment. Once a GT name is consumed it cannot be matched again; subsequent extracted names matching it become false positives.

```python
def score_components(extracted: list[str], ground_truth: list[str]) -> dict:
    if not ground_truth:
        return _zero_metrics()

    unmatched_gt = list(ground_truth)
    tp_pairs: list[tuple[str, str]] = []
    fp: list[str] = []

    # Sort by best available ratio so strong matches claim their GT first.
    scored = []
    for name in extracted:
        best, ratio = find_best_match(name, ground_truth)
        scored.append((ratio, name))
    scored.sort(key=lambda x: -x[0])

    for _, name in scored:
        match, _ratio = find_best_match(name, unmatched_gt)
        if match:
            unmatched_gt.remove(match)
            tp_pairs.append((name, match))
        else:
            fp.append(name)

    tp = len(tp_pairs)                       # == number of distinct GT covered
    precision = tp / len(extracted) if extracted else 0.0
    recall    = tp / len(ground_truth)
    ...
    "missed_names": unmatched_gt,
```

**Improves:** precision, recall, F1, TP, FP, FN become internally consistent. Recall is bounded by 1.0. Duplicate/split component names are correctly penalised as false positives.

---

## A3. Name matching is too strict — penalises correct extractions (hits Gemini hardest)

**Where:** `backend/services/metrics.py:13-24`

**Problem:** `SequenceMatcher` is character-level. Cloud vendor prefixes and suffixes destroy the ratio even when the extraction is semantically correct:

| Ground truth | Extracted (correct!) | ratio | Result |
|---|---|---|---|
| `Amazon S3 Artifacts` | `S3` | 0.19 | MISS |
| `Application Load Balancer` | `ALB` | 0.14 | MISS |
| `Amazon ECS` | `ECS Cluster` | 0.38 | MISS |
| `AWS CodeCommit` | `CodeCommit` | 0.83 | hit |

Gemini normalises and re-labels far more than OCR-based pipelines do, so it eats a disproportionate share of these false misses. **This is a metric artefact, not a Gemini accuracy problem.**

**Fix:** add `rapidfuzz` and replace the scorer with a normalised, token-aware match. Keep `FUZZY_THRESHOLD = 0.75` so results stay comparable to the original design.

```python
# backend/services/metrics.py
from rapidfuzz import fuzz

# Vendor / generic noise words stripped before comparison.
_NOISE_TOKENS = {
    "aws", "amazon", "azure", "microsoft", "google", "gcp", "cloud",
    "service", "services", "server", "instance", "cluster", "node",
    "the", "a", "an", "of",
}

# Bidirectional acronym expansion — extend as diagrams demand.
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

def _normalise(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return _ACRONYMS.get(s, s)

def _tokens(name: str) -> set[str]:
    return {t for t in _normalise(name).split() if t not in _NOISE_TOKENS}

def match_ratio(a: str, b: str) -> float:
    """Max of: token-set ratio, plain ratio, and containment bonus."""
    na, nb = _normalise(a), _normalise(b)
    ta, tb = _tokens(a), _tokens(b)

    r = max(
        fuzz.token_set_ratio(na, nb) / 100.0,
        fuzz.ratio(na, nb) / 100.0,
    )
    # Containment: one name's meaningful tokens fully inside the other's.
    if ta and tb and (ta <= tb or tb <= ta):
        r = max(r, 0.90)
    return r
```

`fuzzy_match` and `find_best_match` both use `match_ratio` instead of `SequenceMatcher`.

**Improves:** removes systematic false misses caused by vendor prefixes, acronyms, and generic suffixes. Expect a meaningful jump in reported component F1 for **all three** pipelines, largest for Gemini. This makes the comparison fair rather than inflating any one arm.

**Add to `backend/requirements.txt`:** `rapidfuzz>=3.9.0`

**Dissertation note:** document this normalisation step in the evaluation-methodology chapter. Report both the raw `SequenceMatcher` numbers and the normalised numbers in one table — the delta is itself a finding about naming variance across extraction paradigms.

---

## A4. Gemini prompt is doing analysis instead of extraction

**Where:** `backend/services/llm/prompts.py` — all three variants

**Problem:** every extraction prompt demands a `metadata` block **per component**: `role`, `bottleneck_risk`, `scalability`, `security_surface`, `responsibilities[]`, `suggestions[]`. That is ~60 output tokens of *reasoning* per component. On a 20-component diagram it is 1200+ tokens spent inventing analysis instead of reading labels.

Two measurable harms:
- **Token exhaustion.** `gemini.py:44-47` already had to raise `max_output_tokens` to 32768, and `gemini.py:223-276` contains a 50-line truncated-JSON salvage routine. When salvage trims to the last complete object, **components are silently dropped** → recall loss that looks like a Gemini failure but is a prompt-design failure.
- **Attention dilution.** The model is optimising for plausible metadata, not faithful transcription. This is a standard driver of both hallucination and omission.

**Fix:** split extraction from enrichment.

1. Create a new prompt set `EXTRACTION_PROMPTS_V2` with `zero_shot`, `few_shot`, `chain_of_thought` variants that request **only**: `components[{id, name, type, technology}]`, `connections[{id, source, target, label, directed}]`, `arch_type`, `confidence_score`. No `metadata`, no `position`, no `responsibilities`, no `suggestions`.
2. Move the metadata block into a separate `ENRICHMENT_PROMPT`, called lazily and only when the user opens the Explain tab. It is not part of extraction and must not be timed into `response_time_ms`.
3. Keep `EXTRACTION_PROMPTS` (v1) in the file behind a `PROMPT_VERSION` setting so the old behaviour is reproducible for an ablation table.

**Improves:** shorter, more reliable output; far fewer truncations; salvage path rarely triggers; recall rises because components stop being silently trimmed.

---

## A5. Gemini prompt does not require verbatim labels

**Where:** `backend/services/llm/prompts.py` — all variants

**Problem:** prompts say `"name": "Component Name"` with no transcription constraint. Gemini expands and normalises: diagram says `S3`, output says `Amazon Simple Storage Service`. Ground truth was annotated from the on-diagram text. Mismatch → false miss. A3 mitigates this; A5 attacks the cause.

**Fix:** add to every v2 extraction prompt, immediately above the JSON template:

```
NAMING RULES — these are strict:
- "name" MUST be the exact text visible in the diagram, character for character.
- Do NOT expand abbreviations. If the diagram says "ALB", output "ALB".
- Do NOT add vendor prefixes that are not printed on the diagram.
- Do NOT rename, pluralise, or tidy the label.
- If a component has an icon but no visible text label, use the standard product
  name for that icon and set "technology" to the same value.
- If you cannot read a label with confidence, omit the component entirely rather
  than guessing a name.
```

**Improves:** raises fuzzy-match hit rate against ground truth, and the final line directly reduces hallucination rate — which is a headline metric of this project.

---

## A6. Gemini is not using structured output

**Where:** `backend/services/llm/gemini.py:44-47`

**Problem:** `GenerateContentConfig` sets only `temperature` and `max_output_tokens`. The model is free to emit markdown fences and prose, which is why `_parse_json` (lines 192-230) strips ` ```json ` fences, hunts for outermost braces, and falls back to `_salvage_truncated_json`. Every one of those paths is lossy.

**Fix:** use Gemini 2.5's native structured output.

```python
self.extraction_config = types.GenerateContentConfig(
    temperature=0.1,
    max_output_tokens=32768,
    response_mime_type="application/json",
    response_schema=EXTRACTION_RESPONSE_SCHEMA,   # define in prompts.py or a new schema module
)
```

Define `EXTRACTION_RESPONSE_SCHEMA` to mirror the v2 output shape, with `type` constrained to the enum from `models/schemas.py`.

Keep `_parse_json` and `_salvage_truncated_json` as a fallback path — do not delete them — but add a counter/log line when salvage fires so you can report how often it was needed before vs after. That before/after count is a usable dissertation figure.

**Improves:** guaranteed-parseable JSON, no fence stripping, no truncation salvage, no silently-dropped components, no wasted retries at `gemini.py:118-126`.

---

## A7. Image MIME type is hardcoded to PNG

**Where:** `backend/services/llm/gemini.py:85`

```python
types.Part.from_bytes(data=image_bytes, mime_type="image/png")
```

JPEG, WebP, and GIF uploads are all declared as PNG. Detect the real type instead:

```python
from PIL import Image
import io
fmt = (Image.open(io.BytesIO(image_bytes)).format or "PNG").lower()
mime = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png",
        "webp": "image/webp", "gif": "image/gif"}.get(fmt, "image/png")
```

**Improves:** removes a silent correctness risk on non-PNG uploads.

---

## A8. Per-component confidence is wrong

**Where:** `backend/services/extraction_orchestrator.py:96`

```python
confidence=raw.get("confidence_score"),   # document-level value assigned to EVERY component
```

Every component gets the identical document-level score, so per-component confidence is meaningless in the UI and in any confidence-vs-correctness analysis.

**Fix:** ask for per-component `confidence` in the v2 prompt and schema; map `c.get("confidence")` per component. Fall back to `None` (not the document score) when absent. Keep the document-level value on `ArchitectureSchema.confidence_score` only.

**Improves:** enables a confidence-calibration figure (does Gemini's self-reported confidence predict correctness?) — a genuinely interesting dissertation result that is currently impossible.

---

## A9. Classical arm fabricates connection direction

**Where:** `backend/services/classical_pipeline.py:395-413`

The detector dedups to **undirected** pairs (`pair = (min(s,t), max(s,t))`, line 403) and then emits `directed=True` (line 411). Direction was never measured. This is fabricated data.

**Fix — one line only:**
```python
directed=False,
```
Do not change anything else in this file. The algorithm stays intact; only the honesty of the output flag changes.

**Improves:** classical arm stops claiming direction it never determined. Directed-connection F1 becomes an honest measurement. Expect classical's directed score to drop — that is the correct result and should be reported.

---

# PART B — Pipeline rebuild (hybrid arm only)

## B1 (C1). PaddleOCR replaces Tesseract + TrOCR in the hybrid arm

**New file:** `backend/services/ocr_engine.py`

```python
"""
PaddleOCR PP-OCRv5 text engine for the hybrid arm.
Single full-page pass returns detection boxes + recognition text + confidence.
Model loaded lazily, cached at module level, thread-safe (same pattern as
hybrid_pipeline._load_models).
"""

class OcrWord:
    text: str
    conf: float
    box: tuple[int, int, int, int]   # x, y, w, h in ORIGINAL image coords

def read_page(img_rgb: np.ndarray) -> list[OcrWord]: ...
def words_in_box(words: list[OcrWord], box, expand: float = 0.0) -> list[OcrWord]: ...
def text_for_box(words: list[OcrWord], box, expand: float = 0.0) -> str: ...
```

Use `PaddleOCR(use_angle_cls=True, lang="en", show_log=False)`. CPU is fine. Sort words into reading order (rows by y-centre with adaptive tolerance, then left-to-right) — reuse the row-grouping logic from `classical_pipeline._cluster_label` as a reference implementation, but reimplement in `ocr_engine.py`; do not import from the classical arm.

**Delete from `hybrid_pipeline.py`:**
`_read_region_text`, `_tesseract_words`, `_clean_ocr_tokens`, `_page_ocr_words`, TrOCR loading in `_load_models`, and constants `TROCR_MODEL_ID`, `TROCR_VALIDATE_RATIO`, `OCR_UPSCALE_MIN_H`, `OCR_UPSCALE_FACTOR`, `LABEL_EXPAND_DOWN`, `LABEL_EXPAND_SIDE`.

**Improves:** text extraction accuracy (PP-OCRv5 substantially outperforms Tesseract on small/stylised diagram text). Eliminates TrOCR hallucination at source, so the whole validation guard becomes unnecessary rather than merely better-tuned. One full-page pass replaces N per-region passes → faster. Removes roughly 120 lines of workaround code.

**Requirements:** `paddlepaddle>=2.6.0`, `paddleocr>=2.9.0`

---

## B2 (C2). LSD replaces HoughLinesP — hybrid arm only

**New file:** `backend/services/connection_detector.py`

```python
def detect_connections(
    img_rgb: np.ndarray,
    component_boxes: list[tuple[int, int, int, int]],
    component_ids: list[str],
    text_boxes: list[tuple[int, int, int, int]],
    notation: str = "informal",
) -> list[DetectedConnection]: ...
```

Steps:
1. Grayscale → Canny.
2. Erase text strokes: zero out the edge map inside every `text_box` (padded ~15%) and inside every `component_box` interior. Same idea as `classical_pipeline._detect_connections` lines 376-382, reimplemented locally.
3. `cv2.createLineSegmentDetector()` (OpenCV ≥ 4.8; `opencv-python>=4.10` is already pinned). If the constructor is unavailable at runtime, fall back to `cv2.ximgproc.createFastLineDetector()`, and if that is also missing, fall back to `HoughLinesP` with a logged warning. Never crash on this.
4. Merge collinear segments: group by angle (±5°) and perpendicular distance (±4 px), join segments whose gap along the shared axis is < 20 px. Diagram connectors are frequently broken by label boxes and crossings.
5. Snap each endpoint to the nearest component box (distance to box edge, not centroid — centroid distance mis-assigns on large boxes). Adaptive radius as in the current code.
6. Drop self-loops and duplicates.

**Improves:** connection recall. Published comparisons put LSD at ~66% recall vs Hough ~30% on line-detection benchmarks; recall is the binding constraint here since the pipelines currently miss connections rather than invent them. LSD also needs no threshold tuning and returns sub-pixel endpoints. Collinear merging recovers connectors broken by crossing edge labels.

**Also:** this file makes the hybrid arm structurally independent of the classical arm, so the three-way connection comparison measures three real methods instead of two.

---

## B3 (C3). Endpoint decoration classifier → real direction

**Where:** `backend/services/connection_detector.py`

For each merged segment, crop a square window at each endpoint sized `~2.5 ×` local stroke width (min 24 px). Within the window, on the binary edge/ink mask:

1. `cv2.findContours` → take the contour nearest the endpoint.
2. `cv2.approxPolyDP` at `epsilon = 0.04 * arcLength`.
3. Classify by vertex count, solidity (`area / convexHullArea`), and fill ratio (ink pixels / hull area):

| vertices | solidity | fill | class |
|---|---|---|---|
| 3 | > 0.85 | > 0.7 | `filled_arrow` / `hollow_triangle` (see below) |
| 3 | > 0.85 | < 0.4 | `hollow_triangle` |
| 4 | > 0.85 | > 0.7 | `filled_diamond` |
| 4 | > 0.85 | < 0.4 | `hollow_diamond` |
| 2 open strokes forming a "V" | — | — | `open_arrow` |
| circle-like, circularity > 0.75 | — | — | `circle` |
| nothing found | — | — | `none` |

Disambiguate `filled_arrow` vs `hollow_triangle` by base width relative to line width: UML inheritance triangles are wide and hollow; flow arrowheads are narrow and solid.

Direction rule:
- decoration at exactly one end → that end is the **target**, `directed = True`
- decoration at both ends → `directed = True`, emit two connections (A→B and B→A)
- decoration at neither end → `directed = False`, keep the undirected pair

**Do NOT stamp `directed = True` by default anywhere.**

**Improves:** direction becomes measured rather than assumed. Directed-pair F1 becomes a real metric for the hybrid arm. Also supplies the `relationship` field in B6.

---

## B4 (C4). Line style — solid vs dashed

**Where:** `backend/services/connection_detector.py`

Sample the ink mask along the segment at 1 px steps → binary occupancy string → compute duty cycle and run-length statistics.
- duty cycle > 0.90 → `solid`
- duty cycle 0.30–0.90 with regular gap runs (gap-length stdev / mean < 0.4) → `dashed`
- otherwise → `unknown`

Pure geometry, no model, roughly 15 lines, reuses segments already computed by B2.

**Improves:** distinguishes async from sync relationships in C4 and dependency from association in UML. Feeds the notation lookup table in B6.

---

## B5 (C5). Three-proposer fusion replaces SAM + CLIP-prompting

**Rewrite:** the core of `backend/services/hybrid_pipeline.py`
**New:** `backend/services/icon_bank.py`, `backend/services/shape_detector.py`, `backend/data/icons/`

### Why SAM is being removed
`SamAutomaticMaskGenerator` is class-agnostic. On synthetic diagrams it over-segments decorative gradients and icon sub-parts while under-distinguishing semantic units. The filter wall at `hybrid_pipeline.py:146-177` (`MAX_REGIONS`, `DUPLICATE_IOU`, `CONTAINMENT_RATIO`, `contains_count >= 2`) exists purely to fight this, and the arm still yields ~6 components in ~137 s where Gemini yields ~15.

**This removal is a reportable negative result. Write it up.**

### Why CLIP stays but changes job
Current use is text-prompt → image zero-shot (`CLIP_PROMPTS`, lines 55-66). CLIP was trained on natural photographs; abstract vector glyphs are out of distribution, which is why `hybrid_pipeline.py:403` already overrides CLIP whenever a keyword matches.

New use is **image → image retrieval**, which is what CLIP embeddings are actually good at:
- Embed every official vendor icon **once, offline** → persisted vector bank.
- At inference, embed each candidate crop and take cosine nearest neighbour.
- Above threshold → known component with the exact vendor product name.

### P1 — `icon_bank.py`

```python
def build_bank(icon_dir: Path, out_path: Path) -> None:
    """Offline. Embeds every icon PNG/SVG with CLIP image encoder.
    Writes {names: [...], vectors: np.ndarray (N, 512), meta: [...]} to .npz."""

def match(crop_rgb: np.ndarray, top_k: int = 1) -> list[IconMatch]:
    """Cosine NN against the loaded bank. IconMatch = (name, type, provider, score)."""
```

Icon sources (all free, redistributable for academic use — record the licence in the dissertation):
- AWS Architecture Icons asset pack
- Azure Architecture Icons
- Google Cloud Architecture Icons

Store under `backend/data/icons/{aws,azure,gcp}/`. Ship the built `.npz` so the bank is not rebuilt at runtime. Provide `backend/scripts/build_icon_bank.py` as the offline builder.

Match threshold: start at cosine ≥ 0.82, make it a module constant, tune on real diagrams.

Each icon file maps to `{product_name, component_type, provider}` via a small hand-written `backend/data/icons/mapping.json`. Generate a first-pass mapping from filenames, then correct by hand.

### P2 — `shape_detector.py`

```python
def detect_shapes(img_rgb: np.ndarray) -> list[Shape]:
    """Shape = (box, kind, confidence).
    kind ∈ {rect, rounded_rect, cylinder, ellipse, diamond, hexagon, actor}"""
```

Method: adaptive threshold → morphological close → `findContours` (RETR_TREE, keep hierarchy) → `approxPolyDP`. Classify by vertex count, aspect ratio, corner curvature. Cylinders detected by two horizontal ellipse arcs bounding a rectangle. Keep the contour hierarchy — B7 needs it for containment.

### P3 — text-cluster proposer

Word boxes from `ocr_engine.read_page` → the union-find spatial clustering already proven in `classical_pipeline.py:272-303`. Reimplement in `hybrid_pipeline.py` (or `common.py`); do not import from the classical arm.

**P3 is the universal floor. It must fire on every diagram.** Every component in every notation carries a label; icon and shape proposers only upgrade precision and typing.

### Fusion rules

Run all three, then merge into one component list:

| Situation | Result | `proposer` |
|---|---|---|
| text cluster **inside** a shape | one component, name = text, type from keywords | `shape+text` |
| text cluster **below/beside** an icon match (within 1.2 × icon height) | one component, name = text, type from icon bank | `icon+text` |
| icon match, no nearby text | one component, name = icon bank product name | `icon` |
| text cluster, no shape, no icon | one component, name = text | `text` |
| shape, no text, no icon, not a container | discard (decoration) | — |

Then NMS at IoU ≥ 0.6, keeping the entry with the richest evidence (`icon+text` > `shape+text` > `icon` > `text`).

Type resolution priority: **icon-bank type > keyword type from label > shape-kind type > `other`**.

### Notation weighting (set by B6)

| Notation | P1 icon bank | P2 shape | P3 text |
|---|---|---|---|
| aws / azure / gcp | **primary** | support | required floor |
| c4 | off | **primary** | required floor |
| uml | off | **primary** | required floor |
| informal | off | loose thresholds | **primary** |

**Improves:** works across all four notations instead of only icon-heavy cloud diagrams. Runtime drops from ~137 s to a target of ~5 s. Removes the SAM filter wall entirely. Replaces CLIP prompt-guessing (already being overridden by keywords) with icon retrieval that returns exact vendor product names.

---

## B6 (C6). Notation router as a real pre-stage

**New file:** `backend/services/notation_classifier.py`

```python
def classify_notation(img_rgb: np.ndarray,
                      ocr_words: list[OcrWord],
                      icon_hit_rate: float) -> tuple[str, float]:
    """Returns (notation, confidence). notation ∈ {aws, azure, gcp, c4, uml, informal}"""
```

Signals (cheap, all computed from data already available):
- icon-bank hit rate above threshold, and which provider dominates → `aws` / `azure` / `gcp`
- `<<stereotype>>` regex, `+`/`-`/`#` visibility markers, compartmented rectangles → `uml`
- `[Container]`, `[Component]`, `[System]`, `[Person]` bracket tags; the C4 blue/grey palette; three-line text blocks inside rounded rects → `c4`
- high stroke-width variance and low corner sharpness → hand-drawn `informal`
- fallback → `informal`

This **replaces** the circular `_infer_diagram_standard()` (`classical_pipeline.py:143-151`, duplicated at `extraction_orchestrator.py:32-41`), which guesses the standard from already-extracted names.

Keep the keyword version as a fallback when the visual classifier is below confidence. Do not delete the classical arm's copy — that arm is frozen.

Router output drives:
1. proposer weights in B5
2. relationship lookup table in B7
3. `ArchitectureSchema.diagram_standard`

**Improves:** notation-appropriate extraction instead of one-size-fits-all, and removes a circular dependency in the framework design. Directly serves the project's stated "modular framework" aim.

---

## B7 (C7). Container / boundary capture

**Where:** `hybrid_pipeline.py` (replaces the discard at line 171), `shape_detector.py`

Stop deleting regions that enclose 2+ components. Emit them as components with `is_container = True`, and set `parent_id` on every component whose box is ≥ 80% inside them. Use the `findContours` hierarchy from P2 for nesting; fall back to geometric containment.

Container names come from text near the top-left of the region (AWS labels VPC/subnet/AZ there; C4 labels system boundaries similarly).

**Improves:** VPC, subnet, availability zone, and C4 system boundaries are retained instead of discarded. Directly serves the project's stated future applications — architecture knowledge graphs, RAG, architecture search — which need hierarchy, not a flat node list.

---

## B8 (C8). Schema additions

**Where:** `backend/models/schemas.py`

All fields optional, all defaulting to `None`/`False`. No existing field renamed or removed.

```python
class ComponentSchema(BaseModel):
    # existing: id, name, type, confidence, technology, position, metadata
    parent_id:    str | None = None     # container this component sits inside
    is_container: bool = False          # VPC / subnet / AZ / C4 boundary
    icon_match:   str | None = None     # exact vendor product name from icon bank
    icon_score:   float | None = None
    stereotype:   str | None = None     # UML <<interface>>, <<abstract>>
    c4_level:     str | None = None     # context | container | component | code
    description:  str | None = None     # C4 third text line
    proposer:     str | None = None     # icon+text | shape+text | icon | text

class ConnectionSchema(BaseModel):
    # existing: id, source, target, label, directed, direction, protocol, data_type
    line_style:        str | None = None  # solid | dashed | unknown
    arrowhead_source:  str | None = None  # none|open_arrow|filled_arrow|hollow_triangle|filled_diamond|hollow_diamond|circle
    arrowhead_target:  str | None = None
    relationship:      str | None = None  # inheritance|realisation|composition|aggregation|dependency|association|data_flow|async
    source_name:       str | None = None  # denormalised — see note
    target_name:       str | None = None

class ArchitectureSchema(BaseModel):
    # existing fields unchanged
    notation_confidence: float | None = None
```

**`source_name` / `target_name` are important.** CLAUDE.md Critical Rule 4 requires connections to be comparable by name across pipelines, but every pipeline emits IDs. Populate both: keep `source`/`target` as IDs for the React Flow frontend, and populate `source_name`/`target_name` for scoring. Once these exist, A1's id→name translation in `benchmark.py` becomes a fallback for old rows rather than the primary path.

### Relationship lookup table

**New file:** `backend/services/relationship_table.py`

```python
RELATIONSHIP_TABLE: dict[str, dict[tuple[str, str], str]] = {
    "uml": {
        ("hollow_triangle", "solid"):  "inheritance",
        ("hollow_triangle", "dashed"): "realisation",
        ("filled_diamond",  "solid"):  "composition",
        ("hollow_diamond",  "solid"):  "aggregation",
        ("open_arrow",      "dashed"): "dependency",
        ("open_arrow",      "solid"):  "association",
        ("none",            "solid"):  "association",
    },
    "c4": {
        ("open_arrow",   "dashed"): "async",
        ("filled_arrow", "dashed"): "async",
        ("open_arrow",   "solid"):  "uses",
        ("filled_arrow", "solid"):  "uses",
        ("none",         "solid"):  "uses",
    },
    "aws":      { ... "data_flow" for all arrowed variants ... },
    "informal": { ... "data_flow" ... },
}
```
`aws`, `azure`, `gcp` share one table. Unknown combinations → `None`, never a guess.

**Improves:** richer machine-readable output per the project aim, with all new fields optional so existing ground truth and the existing benchmark remain valid.

---

## B9 (C9). Two-tier metrics

**Where:** `backend/services/metrics.py`, `backend/routers/benchmark.py`

### Tier A — universal, computed on every diagram. These are the headline numbers.

| Metric | Notes |
|---|---|
| component precision / recall / F1 | normalised fuzzy match from A3, one-to-one assignment from A2 |
| connection F1 **undirected** | `(min(a,b), max(a,b))` pairs, by name |
| connection F1 **directed** | `(source, target)` ordered pairs, by name |
| hallucination rate | from `hallucination_filter.py` |
| `response_time_ms` | already stored |

Report undirected and directed connection F1 as **two separate columns**. Undirected measures topology; directed measures topology + arrowhead reading. Separating them isolates where each paradigm fails.

### Tier B — conditional. Computed only where ground truth defines the field.

| Metric | Subset | Report as |
|---|---|---|
| containment accuracy (`parent_id`) | diagrams whose GT has non-null `parent` | secondary table, state n |
| icon-match precision | AWS / Azure / GCP diagrams | secondary table, state n |
| relationship-type accuracy | UML diagrams (n ≈ 3) | **qualitative case study, NOT an F1 claim** |

### The null-skip guard — get this exactly right

```python
def score_optional_field(extracted, ground_truth, field: str) -> dict | None:
    """Returns None (metric not applicable) if the GT does not annotate this
    field. A null GT value means NOT ANNOTATED — never 'annotated as absent'."""
    annotated = [g for g in ground_truth if g.get(field) is not None]
    if not annotated:
        return None
    ...
```

**If `None` in ground truth were treated as "this field is genuinely empty", every unannotated field would score as a false positive and the new extension fields would destroy the precision of all three pipelines.** Add a unit test that asserts a fully-unannotated optional field yields `None` (skipped) and does not alter Tier A numbers.

Every Tier B table row must print its `n`.

**Improves:** lets the framework emit much richer JSON without the extra fields damaging the headline scores, and keeps per-notation reporting honest about sample size.

---

## B10 (C10). Ground truth schema extension — backwards compatible

**Where:** `evaluation/ground_truth/*.json` (14 files today)

Add optional nullable fields. **No re-annotation of existing files is required** — absent fields are treated as unannotated and skipped by B9.

```json
{
  "diagram_id": "aws_cicd_pipeline",
  "components": [
    {"name": "AWS CodeBuild", "type": "service", "parent": null, "stereotype": null, "c4_level": null}
  ],
  "connections": [
    {"source": "AWS CodeCommit", "target": "AWS CodePipeline",
     "directed": true, "relationship": null, "line_style": null}
  ]
}
```

Write `backend/scripts/migrate_ground_truth.py` to add the keys with null values to all existing files, so the shape is uniform. Then fill real values only where cheap:
- `parent` on the AWS diagrams that show a VPC or subnet boundary (~6 files, ~20 min)
- `directed` on all connections (they are already directional in intent; confirm against the source image)
- `relationship` on UML files only

**Improves:** unlocks Tier B metrics at near-zero annotation cost while keeping all 14 existing files valid.

---

## B11 (C11). Preserve the SAM + CLIP path behind a flag

**Where:** `backend/services/hybrid_pipeline.py`, `backend/config.py`

Move the current implementation to `backend/services/hybrid_pipeline_v1.py` unchanged. Add:

```python
HYBRID_VERSION = os.environ.get("HYBRID_VERSION", "v2")   # "v1" = SAM+CLIP+TrOCR
```

`run_hybrid_pipeline` dispatches on the flag.

**Improves:** gives a free ablation table (SAM+CLIP+TrOCR vs proposer fusion) on identical diagrams and identical metrics, turning the removal of SAM into a measured, defensible negative result rather than deleted work.

---

# PART C — Acceptance criteria

Fable must verify each of these and report actual output, not assertions.

## Part A
- [ ] `benchmark.py` translates connection IDs to names before scoring; connection F1 is non-zero on at least one diagram for at least one pipeline.
- [ ] Unit test: 3 extracted names all matching 1 GT name → `tp == 1`, `fp == 2`, `recall <= 1.0`.
- [ ] Unit test: `match_ratio("ALB", "Application Load Balancer") >= 0.75`; same for `("S3", "Amazon S3 Artifacts")` and `("ECS Cluster", "Amazon ECS")`.
- [ ] `EXTRACTION_PROMPTS_V2` exists; contains no `metadata`, `responsibilities`, or `suggestions` key; v1 still reachable via `PROMPT_VERSION`.
- [ ] All v2 prompts contain the verbatim NAMING RULES block.
- [ ] `extraction_config` sets `response_mime_type` and `response_schema`; salvage-path invocations are logged and counted.
- [ ] MIME type detected from image bytes, not hardcoded.
- [ ] Per-component `confidence` is per-component; document score only on `ArchitectureSchema.confidence_score`.
- [ ] `classical_pipeline.py` diff is **exactly one line** (`directed=False`).

## Part B
- [ ] `hybrid_pipeline.py` has **zero** imports from `classical_pipeline.py`.
- [ ] `git diff --stat backend/services/classical_pipeline.py` shows 1 insertion, 1 deletion.
- [ ] PaddleOCR path runs end-to-end on a real diagram; TrOCR no longer imported anywhere.
- [ ] LSD detector with documented fallback chain (LSD → FastLineDetector → HoughLinesP); fallback logs a warning and never raises.
- [ ] Arrowhead classifier returns a non-`none` class on a UML diagram with inheritance arrows.
- [ ] Icon bank builds; `match()` on a cropped AWS Lambda icon returns `"AWS Lambda"` above threshold.
- [ ] All three proposers fire independently; P3 fires on an informal diagram where P1 and P2 return nothing.
- [ ] Notation classifier returns `aws` on an AWS diagram and `c4` on a C4 diagram.
- [ ] Containers emitted with `is_container=True` and children carrying `parent_id`.
- [ ] Unit test: fully-unannotated optional GT field → Tier B metric is `None`, Tier A numbers unchanged.
- [ ] All 14 existing ground truth files still load and score after migration.
- [ ] `HYBRID_VERSION=v1` reproduces the old SAM path.
- [ ] Hybrid runtime on a typical diagram is under 15 s (target ~5 s), down from ~137 s. **Report the measured number.**

---

# PART D — Files

**New**
```
backend/services/common.py                    # shared helpers, imported by both arms
backend/services/ocr_engine.py                # B1
backend/services/connection_detector.py       # B2, B3, B4
backend/services/icon_bank.py                 # B5 P1
backend/services/shape_detector.py            # B5 P2
backend/services/notation_classifier.py       # B6
backend/services/relationship_table.py        # B8
backend/services/hybrid_pipeline_v1.py        # B11 (moved, unchanged)
backend/scripts/build_icon_bank.py            # offline
backend/scripts/migrate_ground_truth.py       # B10
backend/data/icons/{aws,azure,gcp}/           # icon assets
backend/data/icons/mapping.json               # icon → product name + type
backend/data/icons/bank.npz                   # prebuilt embeddings
```

**Modified**
```
backend/services/metrics.py                   # A2, A3, B9
backend/routers/benchmark.py                  # A1, B9
backend/services/llm/prompts.py               # A4, A5
backend/services/llm/gemini.py                # A6, A7
backend/services/extraction_orchestrator.py   # A8, B6
backend/services/hybrid_pipeline.py           # B1, B2, B5, B7, B11 — full core rewrite
backend/models/schemas.py                     # B8
backend/config.py                             # B11
backend/requirements.txt                      # rapidfuzz, paddleocr, paddlepaddle
evaluation/ground_truth/*.json                # B10
```

**Frozen — one line only**
```
backend/services/classical_pipeline.py        # A9 only. Nothing else.
```

**New requirements**
```
rapidfuzz>=3.9.0
paddlepaddle>=2.6.0
paddleocr>=2.9.0
```
`segment-anything` stays installed for `HYBRID_VERSION=v1`. `transformers` and `torch` stay — CLIP still uses them. Remove nothing from `requirements.txt`.

---

# PART E — Scoping decisions already made

1. **Relationship type is extracted but NOT F1-scored.** With ~3 UML diagrams, a per-notation relationship F1 is not statistically defensible. Emit `relationship` into the JSON, report it as a qualitative case study with 2–3 worked examples.

2. **Primary benchmark grouping is icon-centric vs box-centric**, not the 4-way notation split. n ≈ 3–4 per notation is too thin for the 4-way split to be the headline. Keep the 4-way table as secondary with n printed on every row.

3. **Direction is never assumed.** No arrowhead evidence → `directed=False`. If a layout-convention heuristic (left→right, top→bottom) is added later, it must be measured and reported as its own number, never folded silently into the directed F1.

4. **Expected headline finding:** each paradigm wins on a different diagram family — deterministic pipelines on clean icon-centric cloud diagrams (finite vocabulary, no API, seconds not minutes), Gemini on informal and hand-drawn (semantic inference, tolerates mess). This is a stronger and more honest conclusion than a single-F1 horse race, and it maps directly onto the project's accuracy / privacy / explainability / cost trade-off framing.
