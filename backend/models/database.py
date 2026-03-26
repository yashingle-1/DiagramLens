from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from db.connection import Base


# ── Session Table ─────────────────────────────────────────
# Created every time a user uploads a diagram
# This is the root record everything else links to
class Session(Base):
    __tablename__ = "sessions"

    id              = Column(String, primary_key=True)          # UUID
    image_path      = Column(String, nullable=False)            # local file path
    image_url       = Column(String, nullable=False)            # URL to serve image
    image_hash      = Column(String, nullable=True, index=True) # for Redis cache lookup
    original_filename = Column(String, nullable=True)           # original upload name
    status          = Column(String, default="processing")      # processing | done | failed
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships — one session has one architecture, many chat messages
    architecture    = relationship("Architecture", back_populates="session", uselist=False)
    chat_messages   = relationship("ChatMessage", back_populates="session")


# ── Architecture Table ────────────────────────────────────
# Stores the extracted JSON from Gemini Vision
# One architecture per session
class Architecture(Base):
    __tablename__ = "architectures"

    id              = Column(String, primary_key=True)          # UUID
    session_id      = Column(String, ForeignKey("sessions.id"), nullable=False)
    raw_json        = Column(JSON, nullable=False)              # full extracted JSON
    component_count = Column(Integer, default=0)                # how many components found
    connection_count = Column(Integer, default=0)               # how many connections found
    arch_type       = Column(String, nullable=True)             # microservices | monolith | etc
    confidence_score = Column(Float, nullable=True)             # 0.0 - 1.0 extraction confidence
    llm_provider    = Column(String, nullable=True)             # which model extracted this
    prompt_variant  = Column(String, nullable=True)             # which prompt was used
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship back to session
    session         = relationship("Session", back_populates="architecture")


# ── ChatMessage Table ─────────────────────────────────────
# Stores every message in the AI chat panel
# Keeps full conversation history per session
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id              = Column(String, primary_key=True)          # UUID
    session_id      = Column(String, ForeignKey("sessions.id"), nullable=False)
    role            = Column(String, nullable=False)            # user | assistant
    content         = Column(Text, nullable=False)              # message text
    interview_mode  = Column(Boolean, default=False)            # was interview mode on?
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship back to session
    session         = relationship("Session", back_populates="chat_messages")


# ── CaseStudy Table ───────────────────────────────────────
# Pre-loaded industry architectures (Netflix, Uber, etc)
# These are seeded once and never change
class CaseStudy(Base):
    __tablename__ = "case_studies"

    id              = Column(String, primary_key=True)          # e.g. "netflix"
    title           = Column(String, nullable=False)            # "Netflix Streaming"
    company         = Column(String, nullable=False)            # "Netflix"
    description     = Column(Text, nullable=True)               # short description
    difficulty      = Column(String, default="intermediate")    # beginner | intermediate | advanced
    tags            = Column(JSON, default=list)                # ["microservices", "cdn", "scale"]
    architecture_json = Column(JSON, nullable=False)            # nodes and edges
    hld_content     = Column(Text, nullable=True)               # High Level Design markdown
    lld_content     = Column(Text, nullable=True)               # Low Level Design markdown
    flashcards      = Column(JSON, default=list)                # interview Q&A cards
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())


# ── Benchmark Table ───────────────────────────────────────
# MSc Research — stores results of LLM comparison experiments
# Every time you run a benchmark test, one row is created
class Benchmark(Base):
    __tablename__ = "benchmarks"

    id                  = Column(String, primary_key=True)      # UUID
    session_id          = Column(String, ForeignKey("sessions.id"), nullable=True)
    llm_provider        = Column(String, nullable=False)        # gemini | claude | openai
    prompt_variant      = Column(String, nullable=False)        # zero_shot | few_shot | cot
    diagram_type        = Column(String, nullable=True)         # professional | handdrawn | uml
    
    # Accuracy metrics — compared against ground truth labels
    component_precision = Column(Float, nullable=True)          # correctly found / total found
    component_recall    = Column(Float, nullable=True)          # correctly found / total actual
    component_f1        = Column(Float, nullable=True)          # balance of precision and recall
    connection_precision = Column(Float, nullable=True)
    connection_recall   = Column(Float, nullable=True)
    connection_f1       = Column(Float, nullable=True)
    
    # Hallucination tracking
    hallucinated_components = Column(Integer, default=0)        # components that don't exist
    missed_components   = Column(Integer, default=0)            # components that were missed
    
    # Performance
    response_time_ms    = Column(Integer, nullable=True)        # how long it took
    tokens_used         = Column(Integer, nullable=True)        # token consumption
    
    # Raw data for analysis
    extracted_json      = Column(JSON, nullable=True)           # what the LLM returned
    ground_truth_json   = Column(JSON, nullable=True)           # what was actually there
    
    created_at          = Column(DateTime(timezone=True), server_default=func.now())
