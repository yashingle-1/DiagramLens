# DiagramLens — Project Briefing
**Student:** Yash Rajabhau Ingle | MSc Advanced Computer Science (AI) | University of Leeds
**Working title:** *Beyond VLMs: Benchmarking Rule-Based, Specialized-ML, and Generative Approaches for Software Architecture Diagram Extraction*

> This document is structured in the order I present it: motivation → research questions → system design → backend walkthrough (step by step) → evaluation method → current results → contributions → limitations → future work.

---

## 1. Motivation

Software architecture diagrams are everywhere — design docs, wikis, onboarding material — but they are **images**, not data. To search them, validate them, or feed them into tooling, you must first extract their structure: *which components exist, and how are they connected?*

The obvious modern answer is a Vision-Language Model (VLM) such as Gemini. But VLMs have three costs that matter in practice:

1. **Money** — every extraction is a paid API call.
2. **Privacy** — architecture diagrams are often confidential. Many enterprises cannot send internal system designs to a third-party cloud API at all.
3. **Hallucination** — generative models invent components that are not in the image, and they do so confidently.

This raises the question nobody has answered empirically for this domain: **how far can you get *without* a generative model?** If a free, local, non-generative pipeline achieves, say, 80% of a VLM's accuracy, that changes the deployment decision for a privacy-constrained organisation.

## 2. Research Questions

| RQ | Question |
|----|----------|
| RQ1 | How do the three paradigms compare on **component extraction F1**, across diagram complexity levels? |
| RQ2 | How do they compare on **connection (edge) extraction F1**? |
| RQ3 | Can **OCR cross-validation reliably detect VLM hallucinations** — and which component types hallucinate most? |
| RQ4 | What is the **cost-accuracy frontier**? At what accuracy gap does the free/local pipeline become the rational choice? |
| RQ5 | Does **notation standard** (AWS, C4, UML, informal) significantly affect relative performance? |

Each RQ maps to a dissertation chapter and to a specific chart produced by the system's dashboard.

## 3. The Three Paradigms (core design decision)

All three pipelines take the same input (a PNG upload) and must return the **identical Pydantic schema** — components with `{id, name, type, confidence}` and connections with `{source, target, label}`. This is what makes the comparison fair: same input, same output contract, same scorer.

### Pipeline 1 — Classical CV (rule-based baseline, zero AI)
- OpenCV: grayscale → Gaussian blur → Canny edge detection → contour detection → bounding boxes.
- Tesseract OCR per region for component names.
- HoughLinesP line detection; line endpoints snapped to nearest bounding box = connections.
- **Zero AI, zero API calls, zero cost, fully local.** Runs in ~3 seconds. This is the floor.

### Pipeline 2 — Specialized-ML Hybrid (the novel pipeline)
Three *specialized, non-generative* models chained, each doing exactly one job:
- **SAM** (Meta, `sam_vit_b`) — automatic mask generation → candidate component regions. Filtered smallest-first so leaf components beat enclosing containers (VPC boxes, availability zones).
- **CLIP** (OpenAI, `clip-vit-base-patch32`) — zero-shot classification of each region against text prompts ("a database or data storage component", "a message queue or event bus", …). A "text label" verdict discards non-component regions — but only when CLIP is confident (≥0.5), otherwise the runner-up type is used.
- **TrOCR** (Microsoft, `trocr-base-printed`) + Tesseract — text extraction for the component name. Crops are expanded (labels sit below icons) and upscaled 3× before OCR. TrOCR output is cross-validated against full-page OCR because a generative *decoder* will invent words when shown pure iconography.
- Connections: HoughLinesP endpoints mapped to SAM boxes.
- **No LLM, no generative AI, 100% local, zero API cost.** ~2 minutes on CPU. Key research object: how close can specialized models get to a VLM?

### Pipeline 3 — Generative VLM (the upper bound)
- **Gemini 2.5 Flash** with a chain-of-thought extraction prompt, structured JSON output.
- Followed by the **hallucination filter**: full-image Tesseract OCR → every Gemini component name is fuzzy-matched (SequenceMatcher ≥ 0.8 per significant word) against the OCR text. No OCR evidence → flagged as hallucinated. Tracked per component type.
- Paid API call (~fractions of a cent per image), cloud-dependent, ~30–60 seconds.

**The framing:** classical = floor, VLM = ceiling, hybrid = the interesting middle. The dissertation quantifies the gaps and the conditions under which each paradigm wins.

## 4. System Architecture

```
┌──────────────┐        ┌───────────────────────────────────────┐
│  Frontend    │  HTTP  │  Backend (FastAPI, async)              │
│  Next.js 16  │───────▶│                                        │
│  React Flow  │        │  POST /api/analyze                     │
│  canvas +    │        │    └─ extraction_orchestrator          │
│  Recharts    │        │        ├─ classical_pipeline  ┐        │
│  dashboard   │        │        ├─ hybrid_pipeline     ├─ asyncio.gather (parallel)
└──────────────┘        │        └─ gemini (google.genai)┘       │
                        │        └─ hallucination_filter         │
                        │  POST /api/benchmark ── metrics.py     │
                        │  GET  /api/dashboard ── aggregates     │
                        │  POST /api/chat ── Claude explainer    │
                        ├────────────────────────────────────────┤
                        │  PostgreSQL (sessions, architectures,  │
                        │  benchmarks) · Redis (24h result cache)│
                        └────────────────────────────────────────┘
```

- **Backend:** Python, FastAPI, async SQLAlchemy, PostgreSQL, Redis.
- **Frontend:** Next.js 16, React 19, Tailwind 4, Zustand state, React Flow diagram canvas with Dagre auto-layout, Recharts dashboard, animated packet-flow visualisation on edges.
- **AI is used for extraction only in pipeline 3.** The chat "explain this architecture" feature (Claude/Gemini) is a UX add-on, deliberately outside the research comparison.

## 5. Backend Walkthrough — Step by Step

This is the exact life of one request.

### Step 1 — Upload (`POST /api/analyze`)
1. User drops an image; frontend sends `multipart/form-data` with the file and a `prompt_variant` (default `chain_of_thought`).
2. `services/storage.py` saves the file under a UUID name; a static URL is returned for the frontend to display.

### Step 2 — Orchestration (`services/extraction_orchestrator.py`)
3. The image is SHA-hashed; Redis is checked for a cached Gemini result (24 h TTL) so repeated uploads of the same diagram don't re-bill.
4. All three pipelines launch **concurrently** via `asyncio.gather`. CPU-bound pipelines (classical, hybrid) run in worker threads so they don't block the event loop.
5. Failures are isolated: if one pipeline throws, the other two still return. A failed pipeline yields an **empty component list plus an `extraction_error` string** — deliberately *not* a fake placeholder component, because a fake component would count as a false positive and corrupt the benchmark's precision scores. An empty list is the honest datum (recall = 0).

### Step 3 — Classical pipeline (`services/classical_pipeline.py`)
6. Decode → grayscale → blur → Canny → `findContours` → area-filtered bounding boxes.
7. Tesseract `image_to_data` per box; words below confidence 40 discarded; noise-filtered text becomes the component name; keyword rules assign the type ("redis" → cache, "queue" → queue, …).
8. `HoughLinesP` on the edge image; each line endpoint is snapped to the nearest box within threshold → connection pairs.

### Step 4 — Hybrid pipeline (`services/hybrid_pipeline.py`)
9. SAM generates masks (image downscaled to 1024px long side for CPU speed). Masks filtered by area fraction; duplicates removed by IoU; regions containing ≥2 kept regions rejected as group containers.
10. All crops classified in **one batched CLIP pass** (softmax over 10 type prompts).
11. Per region: crop expanded down/sideways (labels sit outside shapes), upscaled if small, OCR'd — Tesseract first, TrOCR fallback (validated against page OCR), low-confidence Tesseract as last resort. Junk edge tokens (truncated fragments like `'ce'`, stray punctuation) stripped.
12. Near-identical names on **overlapping** boxes deduplicated (same shape segmented twice) — but identical names on distant boxes are kept, because replicated components (multi-AZ web servers) are real.
13. Models are loaded once and cached at module level — loading per request would add ~30 s latency.

### Step 5 — Gemini pipeline (`services/llm/gemini.py`)
14. Image + chain-of-thought prompt sent via the `google.genai` SDK, `temperature=0.1`, `max_output_tokens=32768` (large diagrams previously truncated JSON at 16384).
15. Response text is read defensively (candidate parts, finish-reason checked — `response.text` can raise when the model returns no text part).
16. The JSON parser strips markdown fences, and if the output was truncated mid-object it **salvages** the longest valid prefix by cutting at the last complete object boundary and closing open brackets. Up to 3 attempts with backoff on transport errors or unparseable output.
17. Output is normalised into the same `ArchitectureSchema` as the other two pipelines.

### Step 6 — Hallucination filter (`services/hallucination_filter.py`)
18. Full-image Tesseract OCR → word set.
19. Every Gemini component name: significant words (len > 3) fuzzy-matched against OCR words at ratio ≥ 0.8. Any match → validated; none → hallucinated. Results attached to the Gemini schema (`hallucinated_components`, `hallucination_rate`) and grouped by component type for RQ3.

### Step 7 — Persistence & response
20. One `Session` row + three `Architecture` rows (one per pipeline, with `pipeline`, `response_time_ms`, `diagram_standard`, `complexity`) written to PostgreSQL.
21. Response: `{ session_id, classical, hybrid, gemini, image_url }`. Frontend renders all three on the React Flow canvas with a pipeline switcher; hallucinated nodes get a red dashed border.

### Step 8 — Benchmarking (`POST /api/benchmark`, `services/metrics.py`)
22. Request `{ session_id, diagram_id }` loads `evaluation/ground_truth/{diagram_id}.json` — a hand-annotated file with components, connections, notation standard, complexity level, and a `source_url` as annotation evidence.
23. **Fuzzy matching, never exact:** `SequenceMatcher` ratio ≥ 0.75. "API Gateway" vs "api gateway" must match; small OCR errors must not destroy scores.
24. Components: TP = extracted name matches a ground-truth name; FP = no match (hallucinated/wrong); FN = ground-truth name never matched (missed). Precision, recall, F1.
25. Connections: source/target names resolved to ground-truth names via fuzzy match, then compared as directed pairs. **Names, not IDs** — IDs differ across pipelines; names are the common key.
26. Three `Benchmark` rows stored (one per pipeline) with hallucinated/missed name lists, timing, standard, complexity.

### Step 9 — Dashboard (`GET /api/dashboard`)
27. Aggregates all benchmark rows: overall F1 per pipeline, F1 by complexity, F1 by notation standard, hallucination table, speed and cost comparison. Rendered as Recharts grouped bars — these are the dissertation figures.

## 6. Ground Truth Dataset (novel contribution)

- Target: **25 hand-annotated diagrams** (currently 13+ complete) sourced from official references — AWS Architecture Center, C4 model docs, UML references, plus informal/whiteboard-style diagrams.
- Split across 4–5 notation standards; complexity labelled by rule (low ≤ 7 components, medium 8–14, high 15+).
- Every file carries a `source_url` so the annotation is verifiable.
- Connections annotated by component **name**, making the file pipeline-agnostic.
- No comparable public dataset exists for software architecture diagrams with complexity + standard metadata.

## 7. Current Empirical Observations (pre-final-benchmark)

- **Speed:** classical ~3 s · hybrid ~2 min (CPU) · Gemini ~30–60 s. Classical always fastest — a legitimate finding, not a footnote.
- **Cost:** classical £0 · hybrid £0 · Gemini pay-per-call. Cost scales linearly with diagram count only for the VLM.
- **Robustness:** Gemini failed to return parseable JSON on ~14% of test images before hardening (output truncation on complex diagrams; occasional non-JSON responses). This *itself* is evidence for RQ4 — the VLM is the most accurate and the least reliable component in the system.
- **Notation sensitivity (early signal for RQ5):** the hybrid pipeline performs visibly worse on C4 diagrams (text-heavy boxes that CLIP classifies as "text annotation") than on icon-rich AWS diagrams.
- **Privacy:** two of three pipelines run fully offline. That is the deployment story.

## 8. Contributions

1. **First empirical 3-paradigm comparison** (rule-based CV vs specialized-ML ensemble vs generative VLM) on software architecture diagram extraction.
2. **Novel SAM + CLIP + TrOCR hybrid pipeline** — first application of this specialized-model ensemble to this task.
3. **Novel annotated dataset** — 25 architecture diagrams with notation standard + complexity metadata and verifiable sources.
4. **OCR-based hallucination filter** — a cheap, model-free cross-validation step that detects and type-classifies VLM hallucinations.
5. **Cost-accuracy-privacy frontier** — quantified tradeoffs for enterprise deployment decisions.

## 9. Limitations (stated up front)

- Dataset size (n = 25) limits statistical power; mitigated by controlled metadata and per-category analysis, and justified by manual annotation cost.
- Hybrid pipeline runs on CPU; GPU timing would change the speed comparison (but not accuracy or cost).
- Single VLM tested (Gemini 2.5 Flash); the final evaluation may add one frontier-model pass as an upper-bound reference row.
- Connection extraction for the non-generative pipelines relies on Hough line detection, which struggles with curved/routed edges — expected to depress RQ2 scores, and reported as such.

## 10. Future Work

- **Synthetic training data:** generate unlimited labelled diagrams from Mermaid/PlantUML source → fine-tune a YOLO/DETR detector, replacing SAM's generic segmentation.
- **Local VLM tier:** LLaVA / Qwen-VL as a fourth paradigm — separates "generative capability" from "cloud dependency" in the comparison.
- **Cascade router:** run classical first, escalate to the VLM only when confidence is low → a cost-optimal production system derived directly from RQ4.
- **Arrow-head direction detection** for reliable directed-edge extraction.
- **Round-trip validation:** re-render the extracted JSON and image-diff against the original — an accuracy signal that needs no ground truth.
- **Diagram-to-IaC:** emit a Terraform/infrastructure skeleton from the extracted schema.

---

*Prepared for first assessor presentation — July 2026.*
