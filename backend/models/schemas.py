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
    LOW     = "low"
    MEDIUM  = "medium"
    HIGH    = "high"

class PromptVariant(str, Enum):
    ZERO_SHOT   = "zero_shot"
    FEW_SHOT    = "few_shot"
    COT         = "chain_of_thought"  # chain of thought


# ── Component Schemas ─────────────────────────────────────
# Represents one node in the diagram
class ComponentPosition(BaseModel):
    x: float = 0.0
    y: float = 0.0

class ComponentMetadata(BaseModel):
    role:               Optional[str]       = None
    bottleneck_risk:    Optional[RiskLevel] = None
    scalability:        Optional[str]       = None   # horizontal | vertical | both
    security_surface:   Optional[RiskLevel] = None
    responsibilities:   Optional[List[str]] = []
    suggestions:        Optional[List[str]] = []

class Component(BaseModel):
    id:          str
    name:        str
    type:        ComponentType = ComponentType.OTHER
    technology:  Optional[str] = None               # e.g. "NGINX", "PostgreSQL 15"
    position:    ComponentPosition = ComponentPosition()
    metadata:    ComponentMetadata = ComponentMetadata()

    @field_validator("id")
    @classmethod
    def id_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Component id cannot be empty")
        return v


# ── Connection Schemas ────────────────────────────────────
# Represents one arrow/edge between two components
class Connection(BaseModel):
    id:         str
    source:     str                     # component id
    target:     str                     # component id
    label:      Optional[str] = None    # e.g. "REST", "gRPC", "SQL"
    direction:  Optional[str] = "unidirectional"
    protocol:   Optional[str] = None
    data_type:  Optional[str] = None    # JSON | binary | stream


# ── Full Architecture Schema ──────────────────────────────
# This is what Gemini must return — strictly validated
class ArchitectureSchema(BaseModel):
    components:       List[Component]
    connections:      List[Connection]
    arch_type:        Optional[str]   = None   # microservices | monolith | serverless
    confidence_score: Optional[float] = None   # 0.0 - 1.0

    @field_validator("components")
    @classmethod
    def must_have_components(cls, v):
        if len(v) == 0:
            raise ValueError("Architecture must have at least one component")
        return v


# ── API Request Schemas ───────────────────────────────────
# What the frontend sends to the backend

class ChatRequest(BaseModel):
    session_id:     str
    message:        str
    interview_mode: bool = False

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

class ComponentExplainRequest(BaseModel):
    session_id:     str
    component_id:   str

class BenchmarkRequest(BaseModel):
    session_id:     str
    providers:      List[str] = ["gemini"]      # which LLMs to compare
    prompt_variants: List[PromptVariant] = [PromptVariant.ZERO_SHOT]
    ground_truth:   Optional[Any] = None        # manually labeled JSON


# ── API Response Schemas ──────────────────────────────────
# What the backend sends back to the frontend

class AnalyzeResponse(BaseModel):
    session_id:     str
    architecture:   ArchitectureSchema
    image_url:      str
    cached:         bool = False                # was this from Redis cache?
    llm_provider:   str

class ChatResponse(BaseModel):
    message:        str
    session_id:     str
    interview_mode: bool

class ComponentExplainResponse(BaseModel):
    component_id:   str
    component_name: str
    role:           str
    responsibilities: List[str]
    bottleneck_risk: str
    scalability:    str
    security:       str
    suggestions:    List[str]

class SessionResponse(BaseModel):
    session_id:     str
    architecture:   ArchitectureSchema
    image_url:      str
    created_at:     str

class CaseStudyListItem(BaseModel):
    id:             str
    title:          str
    company:        str
    description:    Optional[str]
    difficulty:     str
    tags:           List[str]

class CaseStudyDetail(BaseModel):
    id:             str
    title:          str
    company:        str
    description:    Optional[str]
    difficulty:     str
    tags:           List[str]
    architecture:   ArchitectureSchema
    hld_content:    Optional[str]
    lld_content:    Optional[str]
    flashcards:     List[Any]
