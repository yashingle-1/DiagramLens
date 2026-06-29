from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.ext.asyncio import AsyncSession
from db.connection import get_db
from services.storage import storage_service
from services.extraction_orchestrator import run_extraction
from models.schemas import DualAnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=DualAnalyzeResponse)
async def analyze_diagram(
    file: UploadFile = File(...),
    prompt_variant: str = Form(default="chain_of_thought"),
    db: AsyncSession = Depends(get_db),
):
    """
    Receives an architecture diagram image and runs three pipelines in parallel:
    - Classical CV (OpenCV + Tesseract, zero AI)
    - Hybrid ML (SAM + CLIP + TrOCR, specialized models, no LLM)
    - Gemini 2.5 Flash (VLM)

    Returns all extracted architectures for side-by-side comparison.
    """

    # ── Save uploaded image ───────────────────────────────
    try:
        image_path, image_url, image_bytes = await storage_service.save_upload(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")

    # ── Run both pipelines ────────────────────────────────
    try:
        result = await run_extraction(
            image_bytes=image_bytes,
            image_path=image_path,
            image_url=image_url,
            original_filename=file.filename or "diagram.png",
            db=db,
            prompt_variant=prompt_variant,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return DualAnalyzeResponse(
        session_id=result["session_id"],
        classical=result["classical"],
        hybrid=result["hybrid"],
        gemini=result["gemini"],
        image_url=image_url,
    )
