from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any
from enum import Enum


# ── Enums ─────────────────────────────────────────────────
class ComponentType(str, Enum):
    SERVICE       = "service"
    DATABASE      = "database"
    GATEWAY       = "gateway"
    QUEUE         = "queue"
    CACHE         = "cache"
    CDN           = "cdn"
    LOAD_BALANCER = "load_balancer"
    CLIENT        = "client"
    STORAGE       = "storage"
    MONITORING    = "monitoring"
    NOTIFICATION  = "notification"
    OTHER         = "other"

class RiskLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"

class PromptVariant(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT  = "few_shot"
    COT       = "chain_of_thought"


# ── Sub-schemas (kept for canvas compatibility) ───────────
class ComponentPosition(BaseModel):
    x: float = 0.0
    y: float = 0.0

class ComponentMetadata(BaseModel):
    role:             Optional[str]       = None
    bottleneck_risk:  Optional[RiskLevel] = None
    scalability:      Optional[str]       = None
    security_surface: Optional[RiskLevel] = None
    responsibilities: Optional[List[str]] = []
    suggestions:      Optional[List[str]] = []


# ── Canonical DiagramLens Component Schema ────────────────
# Both pipelines MUST return this. Classical sets confidence=None.
class ComponentSchema(BaseModel):
    id:   str
    name: str
    type: str = "other"   # gateway|service|database|cache|queue|cdn|load_balancer|client|storage|other
    confidence: Optional[float] = None   # None for classical; 0.0-1.0 for Gemini
    # Optional richer fields — classical omits, Gemini may supply
    technology: Optional[str]               = None
    position:   Optional[ComponentPosition] = None
    metadata:   Optional[ComponentMetadata] = None

    @field_validator("id")
    @classmethod
    def id_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Component id cannot be empty")
        return v


# ── Canonical DiagramLens Connection Schema ───────────────
class ConnectionSchema(BaseModel):
    id:      str
    source:  str          # component id
    target:  str          # component id
    label:   str = ""     # "REST"|"gRPC"|"HTTP"|"TCP"|""
    directed: bool = True
    # Optional legacy fields kept for canvas compatibility
    direction: Optional[str] = None
    protocol:  Optional[str] = None
    data_type: Optional[str] = None


# ── Canonical DiagramLens Architecture Schema ─────────────
# Returned by BOTH pipelines. Every field is required except confidence_score.
class ArchitectureSchema(BaseModel):
    session_id:       str
    pipeline:         str    # "classical" | "hybrid" | "gemini"
    diagram_standard: str    # "aws" | "c4" | "uml" | "informal"
    complexity:       str    # "low" | "medium" | "high"
    arch_type:        str
    components:       List[ComponentSchema]
    connections:      List[ConnectionSchema]
    response_time_ms: int
    # Legacy field kept for frontend compatibility
    confidence_score: Optional[float] = None
    # Hallucination filter results (gemini pipeline only) — OCR cross-validation
    hallucinated_components: Optional[List[str]] = None
    hallucination_rate:      Optional[float]     = None

    @field_validator("components")
    @classmethod
    def must_have_components(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("Architecture must have at least one component")
        return v

# Backward-compat aliases used by chat.py and cases-related code
Component  = ComponentSchema
Connection = ConnectionSchema


# ── API Request Schemas ───────────────────────────────────
class ChatRequest(BaseModel):
    session_id:     str
    message:        str
    interview_mode: bool = False

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

class ComponentExplainRequest(BaseModel):
    session_id:   str
    component_id: str

class BenchmarkRequest(BaseModel):
    session_id: str
    diagram_id: str   # links to evaluation/ground_truth/{diagram_id}.json


# ── API Response Schemas ──────────────────────────────────

# Three-pipeline response from POST /api/analyze
class DualAnalyzeResponse(BaseModel):
    session_id: str
    classical:  ArchitectureSchema
    hybrid:     Optional[ArchitectureSchema] = None   # SAM+CLIP+TrOCR pipeline
    gemini:     ArchitectureSchema
    image_url:  str

# Legacy single-pipeline response (kept for chat.py compatibility)
class AnalyzeResponse(BaseModel):
    session_id:   str
    architecture: ArchitectureSchema
    image_url:    str
    cached:       bool = False
    llm_provider: str

class ChatResponse(BaseModel):
    message:        str
    session_id:     str
    interview_mode: bool

class ComponentExplainResponse(BaseModel):
    component_id:   str
    component_name: str
    role:           str
    responsibilities: List[str]
    bottleneck_risk:  str
    scalability:      str
    security:         str
    suggestions:      List[str]

class SessionListItem(BaseModel):
    session_id:        str
    original_filename: Optional[str]
    created_at:        str

class SessionResponse(BaseModel):
    session_id: str
    classical:  Optional[ArchitectureSchema] = None
    gemini:     Optional[ArchitectureSchema] = None
    image_url:  str
    created_at: str
