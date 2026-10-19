from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.embeddings.visual_lens_service import VisualLensService

app = FastAPI(
    title="Recosys Visual Lens API",
    description="Google Lens-like visual & keyword search over Cloudflare R2 images and Supabase items",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_lens_service: Optional[VisualLensService] = None


def get_lens_service() -> VisualLensService:
    global _lens_service
    if _lens_service is None:
        _lens_service = VisualLensService()
    return _lens_service


class LensUrlSearchRequest(BaseModel):
    image_url_or_base64: str
    top_k: int = 10
    active_only: bool = True
    crop_detected_object: bool = True


@app.post("/api/lens/search")
async def search_by_uploaded_image(
    file: Optional[UploadFile] = File(default=None),
    image_url: Optional[str] = Form(default=None),
    top_k: int = Form(default=10),
    active_only: bool = Form(default=True),
    crop_detected_object: bool = Form(default=True),
):
    """
    Upload a photo (`file`) or pass an `image_url` to receive:
    - `query_analysis`: Detected object bounding box + extracted English & Arabic keywords
    - `top_image_matches`: Top 5-10 items by R2 image embedding similarity
    - `top_keyword_matches`: Top 5-10 items by keyword-to-name/description similarity
    """
    service = get_lens_service()
    if file is not None:
        image_bytes = await file.read()
        return service.search_by_image(
            image_input=image_bytes,
            top_k=top_k,
            active_only=active_only,
            crop_detected_object=crop_detected_object,
        )
    elif image_url:
        return service.search_by_image(
            image_input=image_url,
            top_k=top_k,
            active_only=active_only,
            crop_detected_object=crop_detected_object,
        )
    else:
        raise HTTPException(status_code=400, detail="Provide either 'file' (UploadFile) or 'image_url'.")


@app.post("/api/lens/search-json")
def search_by_json_payload(req: LensUrlSearchRequest):
    """JSON endpoint accepting `image_url_or_base64` and `top_k` (default 10)."""
    service = get_lens_service()
    return service.search_by_image(
        image_input=req.image_url_or_base64,
        top_k=req.top_k,
        active_only=req.active_only,
        crop_detected_object=req.crop_detected_object,
    )


@app.post("/api/lens/sync")
def sync_visual_index(limit: Optional[int] = None, only_with_images: bool = False):
    """Incremental sync of Supabase `items` + `item_images` (R2 links) into `recosys_visual_index`."""
    service = get_lens_service()
    return service.build_and_sync_visual_index(limit=limit, only_with_images=only_with_images)


@app.get("/api/lens/history")
def get_recent_lens_searches(limit: int = 20):
    """Fetches recent Lens search comparisons saved in Supabase `recosys_lens_searches`."""
    service = get_lens_service()
    res = (
        service.supabase.table("recosys_lens_searches")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {"searches": res.data or []}
