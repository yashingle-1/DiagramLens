from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
import os

from config import settings
from db.connection import create_tables, get_db
from fastapi import HTTPException
from models.database import Session as SessionModel, Architecture
from models.schemas import SessionListItem
from routers import analyze, chat, benchmark, dashboard


# ── Lifespan ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting DiagramLens API...")
    os.makedirs(settings.upload_dir, exist_ok=True)
    await create_tables()
    print("Database tables ready")
    yield
    print("Shutting down DiagramLens API...")


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="DiagramLens API",
    description="Classical CV vs Gemini architecture diagram extraction — MSc research",
    version="2.0.0",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static Files ──────────────────────────────────────────
app.mount(
    "/uploads",
    StaticFiles(directory=settings.upload_dir),
    name="uploads",
)


# ── Routers ───────────────────────────────────────────────
app.include_router(analyze.router,   prefix="/api", tags=["analyze"])
app.include_router(chat.router,      prefix="/api", tags=["chat"])
app.include_router(benchmark.router,  prefix="/api", tags=["benchmark"])
app.include_router(dashboard.router,  prefix="/api", tags=["dashboard"])


# ── Sessions list ─────────────────────────────────────────
@app.get("/api/sessions", response_model=list[SessionListItem])
async def list_sessions(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Returns the most recent upload sessions."""
    result = await db.execute(
        select(SessionModel)
        .order_by(desc(SessionModel.created_at))
        .limit(limit)
    )
    sessions = result.scalars().all()
    return [
        SessionListItem(
            session_id=s.id,
            original_filename=s.original_filename,
            created_at=s.created_at.isoformat() if s.created_at else "",
        )
        for s in sessions
    ]


# ── Single session (dual pipeline) ────────────────────────
@app.get("/api/sessions/{session_id}")
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Reconstruct a session's three-pipeline result for refresh / shared links."""
    session_result = await db.execute(
        select(SessionModel).where(SessionModel.id == session_id)
    )
    session = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    arch_result = await db.execute(
        select(Architecture).where(Architecture.session_id == session_id)
    )
    architectures = arch_result.scalars().all()

    classical = None
    hybrid = None
    gemini = None
    for arch in architectures:
        if arch.pipeline == "classical":
            classical = arch.raw_json
        elif arch.pipeline == "hybrid":
            hybrid = arch.raw_json
        elif arch.pipeline == "gemini":
            gemini = arch.raw_json

    return {
        "session_id": session_id,
        "classical":  classical,
        "hybrid":     hybrid,
        "gemini":     gemini,
        "image_url":  session.image_url,
        "created_at": session.created_at.isoformat() if session.created_at else "",
    }


# ── Ground truth list ─────────────────────────────────────
@app.get("/api/ground-truth")
async def list_ground_truth():
    """Lists available ground truth diagram IDs for benchmarking."""
    import json
    from pathlib import Path

    gt_dir = Path(__file__).resolve().parent.parent / "evaluation" / "ground_truth"
    if not gt_dir.exists():
        return []

    items = []
    for path in sorted(gt_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append({
                "diagram_id":       data.get("diagram_id", path.stem),
                "diagram_standard": data.get("diagram_standard", "unknown"),
                "complexity":       data.get("complexity", "unknown"),
                "component_count":  len(data.get("components", [])),
            })
        except Exception:
            continue
    return items


# ── Health Check ──────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
