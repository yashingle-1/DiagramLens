from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.ext.asyncio import AsyncSession
from db.connection import get_db
from services.storage import storage_service
from services.extraction import extraction_service
from services.llm.factory import get_llm_provider
from models.schemas import AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_diagram(
    file: UploadFile = File(...),
    prompt_variant: str = Form(default="zero_shot"),
    db: AsyncSession = Depends(get_db),
):
    """
    Main endpoint — receives image, returns extracted architecture JSON.

    Flow:
    1. Validate and save image to local filesystem
    2. Check Redis cache (same image uploaded before?)
    3. If cache miss → call Gemini Vision API
    4. Validate JSON with Pydantic
    5. Save session to PostgreSQL
    6. Cache result in Redis
    7. Return architecture JSON + session_id to frontend
    """

    # ── Step 1: Save uploaded image ───────────────────────
    try:
        image_path, image_url, image_bytes = await storage_service.save_upload(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")

    # ── Step 2-6: Run full extraction pipeline ────────────
    import traceback
    print(f"DEBUG: File received: {file.filename}, size: {file.size}")
    try:
        session_id, architecture, cached = await extraction_service.extract(
            image_bytes=image_bytes,
            image_path=image_path,
            image_url=image_url,
            original_filename=file.filename or "diagram.png",
            db=db,
            prompt_variant=prompt_variant,
        )
    except ValueError as e:
        # Pydantic validation failed — LLM returned bad JSON
        raise HTTPException(status_code=422, detail=str(e))
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    # except Exception as e:
    #     # Gemini API failed or other error
    #     raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    # ── Step 7: Return response ───────────────────────────
    provider = get_llm_provider()
    return AnalyzeResponse(
        session_id=session_id,
        architecture=architecture,
        image_url=image_url,
        cached=cached,
        llm_provider=provider.get_provider_name(),
    )
