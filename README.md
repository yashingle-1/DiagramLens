# DiagramLens

A framework for extracting structured, machine-readable representations of software
architecture diagrams, and for comparing three different ways of doing it.

MSc Advanced Computer Science (AI) project — University of Leeds.

---

## What it does

You give it an image of a software architecture diagram. It returns JSON describing
the components and the connections between them:

```json
{
  "pipeline": "hybrid",
  "diagram_standard": "uml",
  "components": [
    {"id": "h1", "name": "License Services Java", "type": "service",
     "stereotype": "component", "description": "license_service.jar"}
  ],
  "connections": [
    {"source": "h1", "target": "h2", "source_name": "License Services Java",
     "target_name": "HASP Java Native Interface Proxy",
     "directed": true, "line_style": "solid", "arrowhead_target": "open_arrow"}
  ]
}
```

The same image is processed by three independent pipelines, each representing a
different paradigm, and all three emit the same schema so their outputs are directly
comparable.

## The research question

Vision-Language Models can produce this JSON directly, so why build anything else?

Because a VLM is not always the right tool. It can hallucinate components that are not
in the image, it requires sending the diagram to a third-party API, it costs money per
call, and its output is hard to explain or reproduce. Whether those trade-offs matter
depends on the situation.

This project measures the alternatives under one benchmark rather than assuming an
answer.

## The three pipelines

| Pipeline | Approach | Runs where |
|---|---|---|
| **Classical** | OpenCV contour and edge analysis, Hough line detection, Tesseract OCR. No machine learning of any kind. | Local, ~1.5s |
| **Hybrid** | Specialised discriminative models — PaddleOCR PP-OCRv5 for text, CLIP image embeddings for icon retrieval — combined with deterministic geometry. No generative model. | Local, ~8s |
| **Gemini** | Gemini 2.5 Flash with structured output, prompted to extract the diagram directly. | Google API |

The classical pipeline is the experimental control and is deliberately frozen. The
hybrid pipeline has its own connection detector rather than sharing the classical one,
so the three-way comparison measures three genuinely different methods.

### How the hybrid pipeline works

Three proposers run over the same image and are merged into one component list:

- **Icon retrieval** — CLIP image embeddings matched against a bank of official
  AWS / Azure / GCP icons. Carries cloud diagrams, where the label sits outside the glyph.
- **Shape detection** — contour geometry. Carries C4, UML and informal diagrams, where
  the label sits inside a drawn box.
- **Text clustering** — PaddleOCR word boxes grouped spatially. This is the universal
  floor: every component in every notation carries a label, so this always fires. The
  other two only improve precision and typing.

A notation classifier runs first and selects a rule profile, because notations decorate
components differently. A UML box has compartments (`«stereotype» Name`, then
`artifacts`, then a file list) that a naive reader turns into four components. C4 boxes
carry a bracket tag and a description line. Those rules live in
`backend/services/notation_profiles.py` as data, not code — adding a notation means
adding a dictionary entry.

## Current results

Measured on 13 manually annotated ground-truth diagrams. Component names are compared
with normalised fuzzy matching (acronym expansion, vendor-prefix removal, token-set
similarity) at a 0.75 threshold.

| Pipeline | Component F1 | Precision | Recall | Time |
|---|---|---|---|---|
| Classical | 0.486 | 0.377 | 0.752 | 1.6s |
| Hybrid | **0.597** | 0.542 | 0.707 | 7.8s |
| Gemini | not yet measured | | | |

Component F1 by notation (hybrid):

| Notation | n | F1 |
|---|---|---|
| UML | 3 | 0.783 |
| C4 | 5 | 0.591 |
| Informal | 1 | 0.761 |
| AWS | 4 | 0.424 |

**Caveats, stated plainly:**

- The Gemini pipeline has not been benchmarked since the prompt and schema were
  rewritten. The three-way comparison is currently incomplete.
- n = 13, and the per-notation subsets are n = 1 to 5. These are indicative, not
  conclusive.
- Connection extraction scores poorly (F1 0.029 hybrid, 0.067 classical). Line detection
  is not the bottleneck — endpoints snap correctly and roughly the right number of links
  are found. The loss is in scoring: when component precision is 0.54, more than half of
  all connection endpoints resolve to a component absent from the ground truth, which
  discards the whole connection. Connection accuracy is capped by component precision.
- Reported F1 uses normalised name matching. The same extractions scored with plain
  character similarity give 0.308 (hybrid) and 0.280 (classical). Both figures are
  stored per benchmark run so the effect of normalisation is visible rather than hidden.

## Setup

Requires Python 3.11, Node 20, PostgreSQL and Redis.

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate on Unix
pip install -r requirements.txt
cp ../.env.example .env          # then fill in GEMINI_API_KEY
python scripts/migrate_benchmark_columns.py
uvicorn main:app --reload --port 8000
```

`.env` must live in `backend/`, and `uvicorn` must be started from there — settings are
read relative to the working directory.

```bash
# Frontend
cd frontend
npm install
npm run dev                      # http://localhost:3000
```

Tesseract must be installed and on `PATH` for the classical pipeline. PaddleOCR
downloads its PP-OCRv5 mobile models on first use (about 10s, then cached).

### Optional components

The icon-retrieval proposer is off by default. It is built and calibrated, but on this
dataset it accepted one icon across 41 candidate crops, changed component F1 by 0.000,
and added roughly 5s per diagram. See "What did not work" below.

```bash
pip install diagrams
# copy <site-packages>/resources/{aws,azure,gcp} into backend/data/icons/
python backend/scripts/build_icon_bank.py
ICON_BANK=1 uvicorn main:app        # to enable it
HYBRID_VERSION=v1 uvicorn main:app  # run the earlier SAM+CLIP+TrOCR pipeline
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Upload an image; runs all three pipelines in parallel |
| `POST /api/benchmark` | Score a session against a ground-truth file |
| `GET /api/dashboard` | Aggregate benchmark statistics |
| `GET /api/sessions` | Recent uploads |
| `POST /api/chat` | Ask questions about an extracted architecture |

## Repository layout

```
backend/
  services/
    classical_pipeline.py     Control arm — frozen, do not modify
    hybrid_pipeline.py        Three-proposer fusion
    hybrid_pipeline_v1.py     Earlier SAM+CLIP+TrOCR arm, kept for ablation
    ocr_engine.py             PaddleOCR PP-OCRv5, Tesseract fallback
    shape_detector.py         Contour geometry, compartment handling
    connection_detector.py    LSD segments, arrowheads, line style
    icon_bank.py              CLIP image-to-image icon retrieval
    notation_classifier.py    Which notation is this?
    notation_profiles.py      Per-notation rules, as data
    metrics.py                Fuzzy matching and scoring
    llm/gemini.py             Gemini 2.5 Flash with structured output
  scripts/                    Icon bank builder, database migration
evaluation/ground_truth/      Annotated diagrams (JSON + source image)
frontend/                     Next.js interface
```

## What did not work

Two foundation-model components were implemented, measured and then rejected. Both are
kept in the repository behind flags so the comparison can be reproduced.

**Segment Anything (SAM).** Its automatic mask generator is class-agnostic. On synthetic
diagrams it over-segments decorative gradients and icon sub-parts while failing to
distinguish semantic units, so it needed an extensive filter stack and still produced
about 6 components in roughly 137 seconds. Replaced by contour analysis, which answers
the actual question — "is this a drawn box?" — in milliseconds.

**CLIP for icon identification.** CLIP embeddings of flat vector icons occupy a narrow
cone, so an absolute cosine threshold cannot separate them. Across a 1456-icon bank,
random pairs of *different* icons score a median of 0.790, and the median icon's nearest
*other* icon scores 0.953. At a 0.82 threshold, 29.3% of unrelated pairs pass. In
practice this labelled 19 of 20 crops on one AWS diagram as the same service. Acceptance
was rewritten to be relative (margin over ranks 2–10, plus a z-score), which removed
every false match but left almost no true ones.

CLIP is still used, but for what it is actually good at: image-to-image retrieval
against a known vocabulary, rather than matching text prompts to abstract glyphs.

## Licensing note

The icon bank is derived from vendor icon sets redistributed by the MIT-licensed
`diagrams` package. The glyphs themselves remain AWS, Microsoft and Google trademarks.
Only the derived embeddings are committed to this repository; the raw images are
excluded and can be regenerated with the script above.

---

Yash Rajabhau Ingle — MSc Advanced Computer Science (AI), University of Leeds, 2026.
