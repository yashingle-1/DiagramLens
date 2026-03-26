from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from config import settings
from db.connection import create_tables
from routers import analyze, chat, cases, session, benchmark


# ── Lifespan ─────────────────────────────────────────────
# Runs on startup and shutdown of the app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting ArchExplain API...")

    # Create uploads folder if it doesn't exist
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Create all database tables
    await create_tables()
    print("✅ Database tables ready")

    yield  # App runs here

    # Shutdown
    print("👋 Shutting down ArchExplain API...")


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="ArchExplain API",
    description="AI-Powered Architecture Diagram Analyzer",
    version="1.0.0",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────
# Allows Next.js frontend (port 3000) to call this backend (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static Files ──────────────────────────────────────────
# Serves uploaded images at /uploads/filename.png
app.mount(
    "/uploads",
    StaticFiles(directory=settings.upload_dir),
    name="uploads",
)


# ── Routers ───────────────────────────────────────────────
# Each router handles a group of related endpoints
app.include_router(analyze.router,   prefix="/api", tags=["analyze"])
app.include_router(chat.router,      prefix="/api", tags=["chat"])
app.include_router(cases.router,     prefix="/api", tags=["cases"])
app.include_router(session.router,   prefix="/api", tags=["session"])
app.include_router(benchmark.router, prefix="/api", tags=["benchmark"])


# ── Health Check ──────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}