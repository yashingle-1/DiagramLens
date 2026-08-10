from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from db.connection import get_db
from models.database import Session as SessionModel, Architecture, ChatMessage
from models.schemas import ChatRequest, ChatResponse
from services.cache import cache_service
from services.llm.factory import get_llm_provider
import traceback

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    print(f"DEBUG chat: session_id={request.session_id}, message={request.message[:50]}")

    # Load the architecture the user is actually looking at. Gemini is the
    # default because it produces the richest descriptions, but when the canvas
    # is showing another pipeline the chat must discuss that one's components,
    # or it will reference names the user cannot see.
    wanted = request.pipeline or "gemini"
    arch_result = await db.execute(
        select(Architecture)
        .where(Architecture.session_id == request.session_id)
        .where(Architecture.pipeline == wanted)
    )
    architecture = arch_result.scalars().first()

    # Fall back to any architecture if no Gemini result saved yet
    if not architecture:
        arch_result = await db.execute(
            select(Architecture).where(Architecture.session_id == request.session_id)
        )
        architecture = arch_result.scalars().first()

    if not architecture:
        # Try loading from sessions table directly
        print(f"DEBUG: Architecture not found for session {request.session_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found. Please upload a new diagram."
        )

    print(f"DEBUG: Found architecture with {len(architecture.raw_json.get('components', []))} components")

    # Load conversation history from Redis
    history = await cache_service.get_chat_history(request.session_id)

    # Resolve the selected component so "this component" has a referent.
    # Matched by id, then by name, because ids differ between pipelines.
    focus = None
    if request.component_id:
        for component in architecture.raw_json.get("components", []):
            if (component.get("id") == request.component_id
                    or component.get("name") == request.component_id):
                focus = component
                break

    # Call Gemini
    provider = get_llm_provider()
    try:
        response_text = await provider.chat(
            message=request.message,
            architecture_context=architecture.raw_json,
            conversation_history=history,
            interview_mode=request.interview_mode,
            focus_component=focus,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI chat failed: {str(e)}")

    # Save messages to PostgreSQL
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

    # Update Redis history
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": response_text})
    await cache_service.set_chat_history(request.session_id, history)

    return ChatResponse(
        message=response_text,
        session_id=request.session_id,
        interview_mode=request.interview_mode,
    )