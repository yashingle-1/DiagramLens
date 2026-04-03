# ArchExplain 🏗️

> **AI-Powered Architecture Diagram Analyser**
> Upload a software architecture diagram and get instant, structured explanations powered by LLMs (Gemini, Claude, GPT-4V).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Environment Variables](#environment-variables)
   - [Running with Docker (Recommended)](#running-with-docker-recommended)
   - [Running Manually (Development)](#running-manually-development)
5. [API Reference](#api-reference)
6. [Key Concepts](#key-concepts)
7. [LLM Provider Support](#llm-provider-support)
8. [Development Notes](#development-notes)

---

## Project Overview

ArchExplain analyses architecture diagrams by:

1. **Uploading** an image (PNG/JPG) of a system architecture.
2. **Extracting** components (services, databases, gateways, queues, etc.) and their connections via a Vision LLM.
3. **Exposing** an interactive chat interface to ask follow-up questions about the diagram.
4. **Providing** pre-loaded case studies (Netflix, Uber, etc.) for learning.
5. **Benchmarking** multiple LLM providers and prompt strategies (zero-shot, few-shot, chain-of-thought) against each other.

---

## Tech Stack

| Layer        | Technology                                      |
|--------------|-------------------------------------------------|
| **Frontend** | Next.js 16 · React 19 · TypeScript · Tailwind CSS · shadcn/ui · Zustand · TanStack Query · React Flow |
| **Backend**  | FastAPI · Python · Pydantic v2 · Uvicorn        |
| **Database** | PostgreSQL 15 (via SQLAlchemy async + asyncpg)  |
| **Cache**    | Redis 7                                         |
| **LLMs**     | Google Gemini (primary) · Anthropic Claude · OpenAI GPT-4V |
| **DevOps**   | Docker · Docker Compose                         |

---

## Project Structure

```
archexplain/
├── docker-compose.yml          # Orchestrates all services
├── .env                        # Active environment config (not committed)
├── .env.example                # Template for environment variables
│
├── backend/                    # FastAPI application
│   ├── main.py                 # App entry-point, middleware, router mounts
│   ├── config.py               # Pydantic-settings config (singleton)
│   ├── requirements.txt        # Python dependencies
│   │
│   ├── routers/                # API route handlers
│   │   ├── analyze.py          # POST /api/analyze — image upload & extraction
│   │   ├── chat.py             # POST /api/chat  — conversational Q&A
│   │   ├── cases.py            # GET  /api/cases — pre-built case studies
│   │   ├── session.py          # GET  /api/session/{id} — session retrieval
│   │   └── benchmark.py        # POST /api/benchmark — multi-LLM comparison
│   │
│   ├── services/               # Business logic
│   │   ├── extraction.py       # Diagram → ArchitectureSchema using LLM
│   │   ├── cache.py            # Redis read/write helpers
│   │   ├── storage.py          # File upload & static serving helpers
│   │   └── llm/                # LLM provider adapters (Gemini, Claude, OpenAI)
│   │
│   ├── models/
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── database.py         # SQLAlchemy ORM models & table definitions
│   │
│   ├── db/
│   │   └── connection.py       # Async database connection & table creation
│   │
│   ├── data/                   # Static case study JSON files
│   ├── scripts/                # One-off helper scripts (seeding, migrations)
│   └── uploads/                # Uploaded diagram images (auto-created)
│
└── frontend/                   # Next.js application
    ├── app/                    # Next.js App Router pages
    ├── components/             # Reusable React components
    ├── lib/                    # API client, utilities
    ├── store/                  # Zustand state management
    └── types/                  # Shared TypeScript types
```

---

## Getting Started

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Docker Desktop | ≥ 24 | Recommended for full stack |
| Node.js | ≥ 20 | Frontend dev only |
| Python | ≥ 3.11 | Backend dev only |
| Git | any | — |

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Get free key at [aistudio.google.com](https://aistudio.google.com) |
| `LLM_PROVIDER` | ✅ | `gemini` \| `claude` \| `openai` (default: `gemini`) |
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `REDIS_URL` | ✅ | Redis connection string |
| `CLAUDE_API_KEY` | ❌ | Only needed for benchmarking |
| `OPENAI_API_KEY` | ❌ | Only needed for benchmarking |
| `MAX_UPLOAD_SIZE_MB` | ❌ | Max image size (default: `10`) |

> [!CAUTION]
> Never commit your `.env` file. It is already listed in `.gitignore`.
> The `config.py` file contains a placeholder API key — remove it and rely on `.env`.

---

### Running with Docker (Recommended)

```bash
# 1. Clone and enter project
git clone <repo-url>
cd archexplain

# 2. Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Start all services
docker compose up --build

# 4. Open the app
#    Frontend → http://localhost:3000
#    Backend API → http://localhost:8000
#    API Docs (Swagger) → http://localhost:8000/docs
```

To stop:
```bash
docker compose down
```

To reset everything (including volumes):
```bash
docker compose down -v
```

---

### Running Manually (Development)

**Backend**

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the API server (with auto-reload)
uvicorn main:app --reload --port 8000
```

> Make sure PostgreSQL and Redis are running locally, or use Docker for just the services:
> ```bash
> docker compose up postgres redis
> ```

**Frontend**

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend will be available at `http://localhost:3000`.

---

## API Reference

All endpoints are prefixed with `/api`. Full interactive docs available at `/docs` (Swagger UI) and `/redoc`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/analyze` | Upload diagram image → returns `ArchitectureSchema` |
| `POST` | `/api/chat` | Send a chat message about the analysed diagram |
| `GET` | `/api/cases` | List all pre-loaded case studies |
| `GET` | `/api/cases/{id}` | Get full details of a specific case study |
| `GET` | `/api/session/{id}` | Retrieve a previous analysis session |
| `POST` | `/api/benchmark` | Run multi-LLM / multi-prompt benchmarking |

### Example: Analyse a Diagram

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@/path/to/diagram.png"
```

Response:
```json
{
  "session_id": "uuid",
  "architecture": {
    "components": [...],
    "connections": [...],
    "arch_type": "microservices",
    "confidence_score": 0.92
  },
  "image_url": "/uploads/diagram.png",
  "cached": false,
  "llm_provider": "gemini"
}
```

---

## Key Concepts

### ArchitectureSchema

The core data model returned by every analysis:

```
ArchitectureSchema
├── components[]          # nodes in the diagram
│   ├── id, name, type    # type: service | database | gateway | queue | ...
│   ├── technology         # e.g. "NGINX", "PostgreSQL 15"
│   └── metadata
│       ├── role
│       ├── bottleneck_risk    # low | medium | high
│       ├── scalability        # horizontal | vertical | both
│       ├── security_surface   # low | medium | high
│       ├── responsibilities[]
│       └── suggestions[]
└── connections[]         # edges between components
    ├── source, target    # component ids
    ├── label             # e.g. "REST", "gRPC", "SQL"
    ├── direction         # unidirectional | bidirectional
    ├── protocol
    └── data_type         # JSON | binary | stream
```

### Session Model

Each diagram upload creates a **session** (stored in PostgreSQL). Redis caches results for identical image hashes. Sessions can be retrieved via `/api/session/{id}`.

### Benchmarking

The `/api/benchmark` endpoint runs the same diagram through multiple LLM providers (`gemini`, `claude`, `openai`) and prompt variants:

| Variant | Description |
|---------|-------------|
| `zero_shot` | No examples, direct extraction prompt |
| `few_shot` | Includes example JSON in the prompt |
| `chain_of_thought` | Asks the LLM to reason step-by-step before extracting |

---

## LLM Provider Support

| Provider | Model | Status |
|----------|-------|--------|
| Google Gemini | `gemini-1.5-flash` / `gemini-pro-vision` | ✅ Default |
| Anthropic Claude | `claude-3-opus` | ✅ Optional (benchmarking) |
| OpenAI | `gpt-4-vision-preview` | ✅ Optional (benchmarking) |

Set `LLM_PROVIDER` in `.env` to switch the primary provider.

---

## Development Notes

- **Hot-reload**: Both frontend (`next dev`) and backend (`uvicorn --reload`) support hot-reload out of the box. The Docker volumes in `docker-compose.yml` mount source directories to enable this in containers too.
- **Database migrations**: Use [Alembic](https://alembic.sqlalchemy.org/) for schema migrations. The `scripts/` folder contains seeding helpers.
- **Uploads**: Files are stored in `backend/uploads/` and served statically at `/uploads/<filename>`.
- **CORS**: The backend allows requests from `http://localhost:3000` by default. Change `FRONTEND_URL` in `.env` for other origins.
- **Linting**: Run `npm run lint` in the frontend directory for ESLint checks.

---

## MSc Project Context

This project is developed as part of an MSc dissertation at the **University of Leeds**. It explores the use of Vision-Language Models for automated software architecture understanding, with a focus on structured extraction, multi-LLM comparison, and educational tooling for system design interview preparation.

---

*Last updated: April 2026*
