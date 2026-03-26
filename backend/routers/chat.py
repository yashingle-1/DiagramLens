from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from db.connection import get_db
from models.database import Session as SessionModel, Architecture, ChatMessage
from models.schemas import ChatRequest, ChatResponse
from services.cache import cache_service
from services.llm.factory import get_llm_provider

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    AI chat endpoint — sends message with full architecture context.

    Flow:
    1. Load architecture JSON from PostgreSQL using session_id
    2. Load conversation history from Redis
    3. Send message + architecture + history to Gemini
    4. Save message and response to PostgreSQL
    5. Update conversation history in Redis
    6. Return AI response
    """

    # ── Step 1: Load architecture from PostgreSQL ─────────
    arch_result = await db.execute(
        select(Architecture).where(Architecture.session_id == request.session_id)
    )
    architecture = arch_result.scalar_one_or_none()

    if not architecture:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found"
        )

    # ── Step 2: Load conversation history from Redis ──────
    history = await cache_service.get_chat_history(request.session_id)

    # ── Step 3: Call Gemini with full context ─────────────
    provider = get_llm_provider()
    try:
        response_text = await provider.chat(
            message=request.message,
            architecture_context=architecture.raw_json,
            conversation_history=history,
            interview_mode=request.interview_mode,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI chat failed: {str(e)}")

    # ── Step 4: Save messages to PostgreSQL ───────────────
    user_msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=request.session_id,
        role="user",
        content=request.message,
        interview_mode=request.interview_mode,
    )
    assistant_msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=request.session_id,
        role="assistant",
        content=response_text,
        interview_mode=request.interview_mode,
    )
    db.add(user_msg)
    db.add(assistant_msg)
    await db.commit()

    # ── Step 5: Update Redis conversation history ─────────
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": response_text})
    await cache_service.set_chat_history(request.session_id, history)

    # ── Step 6: Return response ───────────────────────────
    return ChatResponse(
        message=response_text,
        session_id=request.session_id,
        interview_mode=request.interview_mode,
    )
