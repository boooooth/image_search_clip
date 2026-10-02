"""FastAPI backend serving the editorial Multimodal Retrieval Web Application."""

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from image_search_clip.config import IMAGES_DIR, PROJECT_ROOT
from image_search_clip.milvus_manager import get_or_create_collection
from image_search_clip.model import CLIPEncoder
from image_search_clip.search import search

# Directory paths
WEB_DIR = PROJECT_ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
INDEX_HTML = WEB_DIR / "index.html"

# Global model and collection instances for instant response
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload model and collection into memory on startup."""
    print("Pre-loading Milvus collection and CLIP encoder on startup...")
    state["collection"] = get_or_create_collection(recreate=False)
    state["encoder"] = CLIPEncoder()
    print("Ready to serve multimodal queries!")
    yield
    state.clear()


app = FastAPI(title="Fashion Multimodal Retrieval API", lifespan=lifespan)

# Allow CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local dataset images so frontend can render thumbnails
if IMAGES_DIR.is_dir():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# Mount web static assets (CSS, JS)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serves the main editorial search page."""
    if not INDEX_HTML.is_file():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(INDEX_HTML)


@app.get("/api/search")
async def api_text_search(q: str, top_k: int = 10):
    """Text query search endpoint."""
    clean_query = q.strip()
    if not clean_query:
        return {"query": "", "latency_ms": 0.0, "results": []}

    start_time = time.perf_counter()
    try:
        results = search(
            query_input=clean_query,
            query_type="text",
            top_k=top_k,
            collection=state.get("collection"),
            encoder=state.get("encoder"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        "query": clean_query,
        "type": "text",
        "latency_ms": round(latency_ms, 2),
        "results": results,
    }


@app.post("/api/search-image")
async def api_image_search(
    file: UploadFile = File(...),
    top_k: int = Form(10),
):
    """Image upload query search endpoint."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    start_time = time.perf_counter()
    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = search(
            query_input=pil_image,
            query_type="image",
            top_k=top_k,
            collection=state.get("collection"),
            encoder=state.get("encoder"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        "query": file.filename,
        "type": "image",
        "latency_ms": round(latency_ms, 2),
        "results": results,
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start uvicorn server directly."""
    import uvicorn
    uvicorn.run("image_search_clip.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start_server()
