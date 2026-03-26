from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.connection import get_db
from models.database import CaseStudy
from models.database import Architecture
from models.schemas import (
    CaseStudyListItem,
    CaseStudyDetail,
    ArchitectureSchema,
    ComponentExplainRequest,
    ComponentExplainResponse,
)
from services.llm.factory import get_llm_provider

router = APIRouter()


# ── List All Case Studies ─────────────────────────────────
@router.get("/cases", response_model=list[CaseStudyListItem])
async def list_cases(db: AsyncSession = Depends(get_db)):
    """Returns all active industry case studies"""
    result = await db.execute(
        select(CaseStudy).where(CaseStudy.is_active == True)
    )
    cases = result.scalars().all()
    return [
        CaseStudyListItem(
            id=c.id,
            title=c.title,
            company=c.company,
            description=c.description,
            difficulty=c.difficulty,
            tags=c.tags or [],
        )
        for c in cases
    ]


# ── Get Single Case Study ─────────────────────────────────
@router.get("/cases/{case_id}", response_model=CaseStudyDetail)
async def get_case(case_id: str, db: AsyncSession = Depends(get_db)):
    """Returns full case study with HLD, LLD, and flashcards"""
    result = await db.execute(
        select(CaseStudy).where(CaseStudy.id == case_id)
    )
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail=f"Case study '{case_id}' not found")

    # Validate architecture JSON
    arch = ArchitectureSchema(**case.architecture_json)

    return CaseStudyDetail(
        id=case.id,
        title=case.title,
        company=case.company,
        description=case.description,
        difficulty=case.difficulty,
        tags=case.tags or [],
        architecture=arch,
        hld_content=case.hld_content,
        lld_content=case.lld_content,
        flashcards=case.flashcards or [],
    )


# ── Explain Component ─────────────────────────────────────
# Called when user clicks any node on the React Flow canvas
@router.post("/component/explain", response_model=ComponentExplainResponse)
async def explain_component(
    request: ComponentExplainRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Deep analysis of a single component.
    Called when user clicks a node on the diagram canvas.

    Flow:
    1. Load full architecture from PostgreSQL
    2. Find the specific component by ID
    3. Send component + full architecture to Gemini
    4. Return structured analysis
    """

    # Load architecture
    arch_result = await db.execute(
        select(Architecture).where(Architecture.session_id == request.session_id)
    )
    architecture = arch_result.scalar_one_or_none()

    if not architecture:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find the specific component
    components = architecture.raw_json.get("components", [])
    component = next(
        (c for c in components if c["id"] == request.component_id),
        None
    )

    if not component:
        raise HTTPException(
            status_code=404,
            detail=f"Component {request.component_id} not found"
        )

    # Call Gemini for deep component analysis
    provider = get_llm_provider()
    try:
        analysis = await provider.explain_component(
            component=component,
            full_architecture=architecture.raw_json,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    if not analysis:
        raise HTTPException(status_code=500, detail="Failed to generate analysis")

    return ComponentExplainResponse(
        component_id=analysis.get("component_id", request.component_id),
        component_name=analysis.get("component_name", component.get("name", "")),
        role=analysis.get("role", ""),
        responsibilities=analysis.get("responsibilities", []),
        bottleneck_risk=analysis.get("bottleneck_risk", "unknown"),
        scalability=analysis.get("scalability", "unknown"),
        security=analysis.get("security", "unknown"),
        suggestions=analysis.get("suggestions", []),
    )
