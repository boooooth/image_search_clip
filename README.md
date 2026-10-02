# Fashion Multimodal Retrieval (CLIP + Milvus Lite)

A high-performance multimodal search engine for fashion products powered by **OpenAI CLIP** and **Milvus Lite**. Supports text-to-image, image-to-image, and hybrid multimodal semantic retrieval with an editorial web UI.

---

## Features

- **Multimodal Search**: Query fashion items using natural language descriptions or reference images.
- **CLIP Embeddings**: Uses `openai/clip-vit-base-patch32` to project text and images into a shared 512-dimensional vector space.
- **Milvus Lite Vector Store**: Fast, embedded vector search with Cosine metric and query-adaptive hybrid fusion weights.
- **FastAPI Backend**: Asynchronous REST API serving query endpoints with pre-loaded models in memory.
- **Interactive Web Interface**: Clean UI for searching by query text or dragging & dropping image files with live similarity scores.

---

## Project Structure

```text
image_search_clip/
├── src/
│   └── image_search_clip/
│       ├── __init__.py
│       ├── config.py           # Configuration, weights, and paths
│       ├── model.py            # CLIP encoder for text and images
│       ├── milvus_manager.py   # Milvus Lite schema and collection manager
│       ├── ingest.py           # Ingestion pipeline for encoding dataset
│       ├── search.py           # Hybrid vector search logic
│       ├── server.py           # FastAPI application
│       └── demo.py             # CLI demonstration script
├── web/
│   ├── index.html              # Frontend search UI
│   └── static/
│       ├── css/style.css       # Editorial styling
│       └── js/app.js           # Query interaction and image preview
├── styles.csv                  # Dataset product metadata
├── pyproject.toml              # Dependencies and packaging
├── .gitignore                  # Git ignore rules for dataset, weights, and caches
└── README.md
```

---

## Quickstart

### 1. Prerequisites

Ensure you have Python 3.12+ and [uv](https://docs.astral.sh/uv/) installed (or standard `pip`).

```bash
# Clone the repository
git clone https://github.com/<your-username>/image_search_clip.git
cd image_search_clip
```

### 2. Install Dependencies

Using `uv`:
```bash
uv sync
```

Or using standard `pip`:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e .
```

### 3. Prepare Dataset

The dataset is based on the [Kaggle Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).

1. Place `styles.csv` in the project root.
2. Extract product images into an `images/` directory in the project root:
   ```text
   images/
   ├── 10000.jpg
   ├── 10001.jpg
   └── ...
   ```
*(Note: Large dataset folders and archives like `images/` and `images.zip` are excluded by `.gitignore` to keep repository clones fast and lightweight).*

### 4. Run Data Ingestion

Encode the dataset and index vectors into Milvus Lite:

```bash
uv run python -m image_search_clip.ingest
```

### 5. Launch the Web Application

Start the FastAPI server:

```bash
uv run uvicorn image_search_clip.server:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Search Strategies

- **Text Search**: Encodes the search query via CLIP text transformer and retrieves closest image & caption vectors in Milvus with cosine similarity.
- **Image Search**: Encodes query images via CLIP ViT vision model to find visually similar fashion items.
- **Adaptive Weighting**: Configurable fusion weights (`image_weight` and `text_weight`) dynamically balance visual similarity vs. categorical text relevance.

---

## License

MIT
