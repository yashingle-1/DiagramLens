from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.connection import get_db
from models.database import Session as SessionModel, Architecture
from models.schemas import SessionResponse, ArchitectureSchema

router = APIRouter()


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Loads a saved session by ID.
    Used when:
    - User reloads the page
    - User opens a shared diagram link
    - Frontend needs to restore previous state
    """

    # Load session
    session_result = await db.execute(
        select(SessionModel).where(SessionModel.id == session_id)
    )
    session = session_result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )

    # Load architecture
    arch_result = await db.execute(
        select(Architecture).where(Architecture.session_id == session_id)
    )
    architecture = arch_result.scalar_one_or_none()

    if not architecture:
        raise HTTPException(
            status_code=404,
            detail=f"Architecture for session {session_id} not found"
        )

    # Validate architecture JSON through Pydantic
    arch_schema = ArchitectureSchema(**architecture.raw_json)

    return SessionResponse(
        session_id=session_id,
        architecture=arch_schema,
        image_url=session.image_url,
        created_at=str(session.created_at),
    )


@router.get("/sessions")
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    limit: int = 10,
):
    """Lists recent sessions — useful for history panel"""
    result = await db.execute(
        select(SessionModel)
        .order_by(SessionModel.created_at.desc())
        .limit(limit)
    )
    sessions = result.scalars().all()

    return [
        {
            "session_id": s.id,
            "original_filename": s.original_filename,
            "image_url": s.image_url,
            "status": s.status,
            "created_at": str(s.created_at),
        }
        for s in sessions
    ]
