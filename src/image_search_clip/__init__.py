"""Multimodal retrieval pipeline using OpenAI CLIP and Milvus."""

from image_search_clip.config import (
    COLLECTION_NAME,
    DEVICE,
    IMAGE_QUERY_WEIGHTS,
    MODEL_NAME,
    TEXT_QUERY_WEIGHTS,
)
from image_search_clip.ingest import index_dataset, load_dataset_from_csv
from image_search_clip.milvus_manager import get_or_create_collection
from image_search_clip.model import CLIPEncoder
from image_search_clip.search import search


def main() -> None:
    """CLI entrypoint when running `image-search-clip`."""
    from image_search_clip.demo import run_demo
    run_demo(sample_size=25)


def start_app():
    """Start the editorial Web UI server."""
    from image_search_clip.server import start_server
    start_server()


__all__ = [
    "CLIPEncoder",
    "get_or_create_collection",
    "index_dataset",
    "load_dataset_from_csv",
    "search",
    "main",
    "MODEL_NAME",
    "DEVICE",
    "COLLECTION_NAME",
    "TEXT_QUERY_WEIGHTS",
    "IMAGE_QUERY_WEIGHTS",
]
