from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from db.connection import Base


# ── Session Table ─────────────────────────────────────────
class Session(Base):
    __tablename__ = "sessions"

    id                = Column(String, primary_key=True)
    image_path        = Column(String, nullable=False)
    image_url         = Column(String, nullable=False)
    image_hash        = Column(String, nullable=True, index=True)
    original_filename = Column(String, nullable=True)
    status            = Column(String, default="processing")   # processing | done | failed
    created_at        = Column(DateTime(timezone=True), server_default=func.now())
    updated_at        = Column(DateTime(timezone=True), onupdate=func.now())

    # One session → two architectures (classical + gemini)
    architectures = relationship("Architecture", back_populates="session")
    chat_messages = relationship("ChatMessage", back_populates="session")


# ── Architecture Table ────────────────────────────────────
# One row per pipeline per session (so two rows per upload)
class Architecture(Base):
    __tablename__ = "architectures"

    id               = Column(String, primary_key=True)
    session_id       = Column(String, ForeignKey("sessions.id"), nullable=False)
    raw_json         = Column(JSON, nullable=False)
    component_count  = Column(Integer, default=0)
    connection_count = Column(Integer, default=0)
    arch_type        = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    llm_provider     = Column(String, nullable=True)
    prompt_variant   = Column(String, nullable=True)
    # DiagramLens research fields
    pipeline         = Column(String, nullable=True)   # "classical" | "gemini"
    diagram_standard = Column(String, nullable=True)   # "aws" | "c4" | "uml" | "informal"
    complexity       = Column(String, nullable=True)   # "low" | "medium" | "high"
    response_time_ms = Column(Integer, nullable=True)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="architectures")


# ── ChatMessage Table ─────────────────────────────────────
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id             = Column(String, primary_key=True)
    session_id     = Column(String, ForeignKey("sessions.id"), nullable=False)
    role           = Column(String, nullable=False)   # user | assistant
    content        = Column(Text, nullable=False)
    interview_mode = Column(Boolean, default=False)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="chat_messages")


# ── Benchmark Table ───────────────────────────────────────
# One row per pipeline per benchmark run
class Benchmark(Base):
    __tablename__ = "benchmarks"

    id                   = Column(String, primary_key=True)
    session_id           = Column(String, ForeignKey("sessions.id"), nullable=True)
    llm_provider         = Column(String, nullable=False)    # "gemini" | "classical"
    prompt_variant       = Column(String, nullable=False)
    diagram_type         = Column(String, nullable=True)

    # Accuracy metrics
    component_precision  = Column(Float, nullable=True)
    component_recall     = Column(Float, nullable=True)
    component_f1         = Column(Float, nullable=True)
    connection_precision = Column(Float, nullable=True)
    connection_recall    = Column(Float, nullable=True)
    connection_f1        = Column(Float, nullable=True)

    # Hallucination tracking — stored as JSON lists of names, not counts
    hallucinated_components = Column(JSON, default=list)
    missed_components       = Column(JSON, default=list)

    # Performance
    response_time_ms = Column(Integer, nullable=True)
    tokens_used      = Column(Integer, nullable=True)

    # DiagramLens research fields
    diagram_id       = Column(String, nullable=True)   # links to ground truth filename
    diagram_standard = Column(String, nullable=True)
    complexity       = Column(String, nullable=True)

    # Raw data
    extracted_json   = Column(JSON, nullable=True)
    ground_truth_json = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
