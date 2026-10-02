"""Configuration settings for the Multimodal CLIP + Milvus Retrieval Pipeline."""

from pathlib import Path

# Paths
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

DATA_CSV_PATH = PROJECT_ROOT / "styles.csv"
IMAGES_DIR = PROJECT_ROOT / "images"

# Milvus Lite Storage Path & Collection
# Uses local SQLite-backed Milvus Lite file
MILVUS_URI = str(PROJECT_ROOT / "milvus_multimodal.db")
COLLECTION_NAME = "fashion_multimodal"

import torch

# Model Configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512

# Auto-detect GPU (e.g. RTX 3060) or fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Milvus Index Configuration
# AUTOINDEX or HNSW with COSINE metric for normalized vectors
INDEX_TYPE = "AUTOINDEX"
METRIC_TYPE = "COSINE"

# Hybrid Search Weights (query-adaptive)
TEXT_QUERY_WEIGHTS = {
    "image_weight": 0.3,
    "text_weight": 0.7,
}

IMAGE_QUERY_WEIGHTS = {
    "image_weight": 0.8,
    "text_weight": 0.2,
}

# Batch size: 64 on RTX 3060 for maximum throughput, 16 on CPU
INGEST_BATCH_SIZE = 64 if torch.cuda.is_available() else 16
