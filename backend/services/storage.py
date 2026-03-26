import uuid
import os
from pathlib import Path
from fastapi import UploadFile, HTTPException
from PIL import Image
import io
from config import settings


ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_SIZE_BYTES = settings.max_upload_size_mb * 1024 * 1024


class StorageService:

    def __init__(self):
        # Ensure uploads directory exists
        os.makedirs(settings.upload_dir, exist_ok=True)

    # ── Save Uploaded File ────────────────────────────────
    # Returns (file_path, file_url, image_bytes)
    async def save_upload(self, file: UploadFile) -> tuple[str, str, bytes]:

        # Read file into memory
        image_bytes = await file.read()

        # Validate file size
        if len(image_bytes) > MAX_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB"
            )

        # Validate it's actually an image using Pillow
        # This prevents non-image files from being saved
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Please upload PNG, JPG, or WebP."
            )

        # Validate MIME type
        content_type = file.content_type or ""
        if content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type {content_type} not allowed. Use PNG, JPG, or WebP."
            )

        # Generate unique filename using UUID
        ext = Path(file.filename or "image.png").suffix or ".png"
        filename = f"{uuid.uuid4()}{ext}"

        # Save to uploads directory
        file_path = os.path.join(settings.upload_dir, filename)
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        # URL that the frontend can use to display the image
        file_url = f"/uploads/{filename}"

        return file_path, file_url, image_bytes

    # ── Delete File ───────────────────────────────────────
    def delete_file(self, file_path: str) -> None:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not delete file {file_path}: {e}")


# Singleton instance
storage_service = StorageService()
