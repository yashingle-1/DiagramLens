# DiagramLens

### From software architecture diagrams to structured, queryable system graphs.

**DiagramLens** is a framework for extracting structured software architecture information from diagram images.

Upload an architecture diagram and DiagramLens identifies **components and connections**, converts them into a common JSON schema, visualises the resulting architecture graph, and allows the outputs of three different extraction approaches to be compared.

**MSc Advanced Computer Science (Artificial Intelligence) University of Leeds, 2026**

**Dissertation:** *A Framework for Software Architecture Diagram Understanding and Structured Knowledge Extraction*

---

## Overview

Architecture diagrams contain valuable information about the structure of a software system, but that information is usually locked inside an image.

DiagramLens converts that visual representation into structured data that can be:

* visualised as a graph
* benchmarked against ground truth
* queried using natural language
* reused by other software systems

The project compares three approaches using the **same output schema and evaluation process**:

| Pipeline      | Approach                   | Runs on    | Component F1 |
| ------------- | -------------------------- | ---------- | -----------: |
| **Classical** | OCR + Computer Vision      | Local      |        0.558 |
| **Hybrid**    | OCR + ML + Computer Vision | Local      |        0.713 |
| **Gemini**    | Vision-Language Model      | Google API |    **0.911** |

### Benchmark

**37 diagrams · 6 notation families · 458 annotated components**

AWS · Azure · GCP · C4 · UML · Informal

---

## How it works
An uploaded diagram is processed by all three pipelines independently.

Each pipeline produces the same `ArchitectureSchema`, making the results directly comparable.

The extracted architecture can then be:

1. **Visualised** as an interactive graph
2. **Benchmarked** against annotated ground truth
3. **Queried** through the architecture chat interface

---

## The three pipelines

### Classical

A deterministic computer-vision baseline using:

* Tesseract OCR
* text clustering
* Canny edge detection
* connector endpoint matching

**Local · No network · 2.5 s median**

---

### Hybrid

A local ML/CV pipeline combining:

* PaddleOCR
* shape detection
* spatial text clustering
* notation-specific rules
* connector skeleton tracing

It adapts its extraction rules to AWS, Azure, GCP, C4, UML and informal diagrams.

**Local · No network · 5.9 s median**

---

### Gemini

A Vision-Language Model pipeline using **Gemini 2.5 Flash** with schema-constrained output.

The model receives the complete diagram and extracts components and relationships directly into the common schema.

The application supports zero-shot, few-shot and chain-of-thought prompt strategies.

**Google API · 9.3 s median**

---

## Structured output

All pipelines produce the same architecture representation:

```json
{
  "pipeline": "hybrid",
  "diagram_standard": "uml",
  "components": [
    {
      "id": "h1",
      "name": "License Services Java",
      "type": "service",
      "stereotype": "component"
    }
  ],
  "connections": [
    {
      "source": "License Services Java",
      "target": "HASP Java Native Interface Proxy",
      "directed": true,
      "label": null
    }
  ]
}
```

This common schema allows outputs from different extraction approaches to be compared and consumed by the rest of the application.

---

# Results

## Component extraction

| Pipeline   |        F1 | Precision |    Recall |
| ---------- | --------: | --------: | --------: |
| Classical  |     0.558 |     0.476 |     0.817 |
| Hybrid     |     0.713 |     0.659 |     0.812 |
| **Gemini** | **0.911** | **0.950** | **0.897** |

Gemini extracted 444 components compared with 458 ground-truth components.

The classical pipeline has higher recall but substantially more over-extraction, while the hybrid pipeline improves precision through notation-aware processing.

---

## Results by notation

| Notation | Classical | Hybrid |    Gemini |
| -------- | --------: | -----: | --------: |
| AWS      |     0.581 |  0.592 | **0.872** |
| Azure    |     0.680 |  0.691 | **0.960** |
| GCP      |     0.543 |  0.562 | **0.923** |
| C4       |     0.434 |  0.768 | **0.909** |
| UML      |     0.654 |  0.773 | **0.873** |
| Informal |     0.612 |  0.882 | **0.938** |

---

## Connection extraction

| Pipeline   | Directed F1 | Undirected F1 |
| ---------- | ----------: | ------------: |
| Classical  |       0.058 |         0.095 |
| Hybrid     |       0.052 |         0.115 |
| **Gemini** |   **0.652** |     **0.695** |

Connection extraction is considerably more difficult for the local pipelines because connector endpoints must be associated with correctly detected components.

---

## Speed

| Pipeline  | Median | Network |
| --------- | -----: | ------- |
| Classical |  2.5 s | No      |
| Hybrid    |  5.9 s | No      |
| Gemini    |  9.3 s | Yes     |

All measurements were performed on CPU.

---

# Application

The DiagramLens application provides an interactive interface for exploring the extracted architectures.

## Interactive graph

The extracted architecture is rendered as an interactive graph using React Flow and Dagre.

![DiagramLens architecture graph](evaluation/figures/Screenshot(89).png)

## Benchmark dashboard

Compare component and connection metrics for the different pipelines.

![DiagramLens benchmark dashboard](evaluation/figures/Screenshot(90).png)

## Architecture chat

Select a component and ask questions about its relationships, or ask questions about the complete architecture.

![DiagramLens architecture chat](evaluation/figures/Screenshot(91).png)

---

# Benchmark

The benchmark contains **37 manually annotated architecture diagrams**.

| Notation  | Diagrams | Components |
| --------- | -------: | ---------: |
| AWS       |        7 |        107 |
| Azure     |        5 |         70 |
| GCP       |        5 |         71 |
| C4        |       11 |         81 |
| UML       |        4 |         41 |
| Informal  |        5 |         88 |
| **Total** |   **37** |    **458** |

Each annotation includes the source image, ground-truth structure, source URL and licence information.

The benchmark can be reproduced without running the web application or database.

---

# Reproducing the benchmark

Run the complete offline benchmark:

```bash
python backend/scripts/run_offline_benchmark.py
```

Generate the figures:

```bash
python evaluation/figures/make_results_figures.py
```

Run stability analysis:

```bash
python backend/scripts/result_stability.py
```

Validate the ground truth:

```bash
python backend/scripts/validate_ground_truth.py
```

To run only the local pipelines:

```bash
python backend/scripts/run_offline_benchmark.py --arms classical hybrid
```

Benchmark results are written to:

```text
evaluation/results/offline_benchmark.json
```

---

# Setup

## Requirements

* Python 3.11
* Node.js 20
* PostgreSQL
* Redis
* Tesseract OCR
* CPU-compatible PyTorch

A Gemini API key is required for the Gemini pipeline.

---

## Backend

```bash
cd backend

python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
cp ../.env.example .env
```

Add your Gemini API key:

```text
GEMINI_API_KEY=your_key_here
```

Run the database migration:

```bash
python scripts/migrate_benchmark_columns.py
```

Start the backend:

```bash
uvicorn main:app --reload --port 8000
```

> `.env` should be located inside `backend/`, and the backend should be started from that directory.

### Additional dependencies

Tesseract must be installed and available on `PATH`.

PaddleOCR downloads its PP-OCRv5 mobile weights on first use.

Install only one OpenCV distribution. The project requires the contrib build:

```bash
pip install opencv-contrib-python
```

For CPU-only PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

---

# API

| Endpoint              | Description                                      |
| --------------------- | ------------------------------------------------ |
| `POST /api/analyze`   | Upload an image and run the extraction pipelines |
| `POST /api/benchmark` | Benchmark an extraction against ground truth     |
| `GET /api/dashboard`  | Retrieve aggregate benchmark statistics          |
| `GET /api/sessions`   | Retrieve recent sessions                         |
| `POST /api/chat`      | Ask questions about an extracted architecture    |

---

# Repository structure

```text
DiagramLens/
├── backend/
│   ├── services/          # Extraction pipelines and supporting modules
│   ├── routers/            # API endpoints
│   └── scripts/            # Benchmark and utility scripts
│
├── frontend/               # Next.js application
│
├── evaluation/
│   ├── ground_truth/       # Annotated diagrams
│   ├── figures/            # Result figures
│   └── results/            # Benchmark results
│
├── docs/                   # Project documentation
│
├── .env.example
└── README.md
```

---

# Contributions

The main contributions of DiagramLens are:

* A common structured schema for architecture-diagram extraction.
* Three independent extraction paradigms: Classical, Hybrid and VLM.
* A hand-annotated benchmark of 37 diagrams across six notation families.
* Evaluation of component and connection extraction.
* An interactive application for visualising and querying extracted architectures.
* A reproducible offline benchmark and evaluation pipeline.

---

# Licensing

The project code is provided under the licence included in this repository.

The benchmark diagrams remain subject to their **original source licences**. Source URLs and licence information are recorded with the corresponding annotations.

The icon bank is derived from vendor icon sets distributed through the MIT-licensed `diagrams` package. AWS, Microsoft and Google Cloud icons remain the property/trademarks of their respective owners.

Raw vendor icon images are not committed to the repository; derived CLIP embeddings can be regenerated locally.

Users are responsible for complying with the licences and terms applicable to source diagrams and vendor assets.

---

# Author

**Yash Rajabhau Ingle**

MSc Advanced Computer Science (AI)
University of Leeds · 2026

**Dissertation:**
*A Framework for Software Architecture Diagram Understanding and Structured Knowledge Extraction*

**Repository:**
https://github.com/yashingle-1/DiagramLens
