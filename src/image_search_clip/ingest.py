"""Offline ingestion pipeline for encoding images & captions and upserting into Milvus."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from image_search_clip.config import (
    DATA_CSV_PATH,
    IMAGES_DIR,
    INGEST_BATCH_SIZE,
)
from image_search_clip.milvus_manager import get_or_create_collection
from image_search_clip.model import CLIPEncoder


def load_dataset_from_csv(
    csv_path: Path = DATA_CSV_PATH,
    images_dir: Path = IMAGES_DIR,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Reads styles.csv and creates ingestion items for existing image files.

    Args:
        csv_path: Path to styles.csv
        images_dir: Path to directory containing {id}.jpg files
        limit: Optional maximum number of items to load

    Returns:
        List of item dictionaries: [{'id': int, 'image_path': str, 'caption': str, 'image_url': str}]
    """
    print(f"Loading styles metadata from: {csv_path}")
    # Use on_bad_lines='skip' to guard against any malformed lines in CSV
    df = pd.read_csv(csv_path, on_bad_lines="skip")

    items = []
    for _, row in df.iterrows():
        try:
            item_id = int(row["id"])
            img_filename = f"{item_id}.jpg"
            img_path = images_dir / img_filename

            # Verify image file exists and caption is present
            if not img_path.is_file():
                continue

            caption = str(row.get("productDisplayName", "")).strip()
            if not caption or caption.lower() == "nan":
                # Fallback to articleType and baseColour if display name is blank
                article = str(row.get("articleType", "")).strip()
                color = str(row.get("baseColour", "")).strip()
                caption = f"{color} {article}".strip()

            items.append(
                {
                    "id": item_id,
                    "image_path": str(img_path),
                    "caption": caption[:512],
                    "image_url": str(img_path),
                }
            )

            if limit and len(items) >= limit:
                break
        except (ValueError, TypeError):
            continue

    print(f"Loaded {len(items)} valid items with matching image files.")
    return items


def index_dataset(
    items: List[Dict[str, Any]],
    batch_size: int = INGEST_BATCH_SIZE,
    collection=None,
    encoder: Optional[CLIPEncoder] = None,
) -> int:
    """Offline Ingestion Pipeline:
    Embeds each item's image via CLIP Vision Encoder -> image_vector (512-D)
    Embeds each item's caption via CLIP Text Encoder -> text_vector (512-D)
    Upserts all records into Milvus.

    Args:
        items: List of dicts, each with 'id', 'image_path', 'caption', 'image_url'
        batch_size: Batch size for CPU embedding extraction
        collection: Optional pymilvus Collection instance
        encoder: Optional CLIPEncoder instance

    Returns:
        Total number of indexed items.
    """
    if not items:
        print("No items to index.")
        return 0

    if collection is None:
        collection = get_or_create_collection()

    if encoder is None:
        encoder = CLIPEncoder()

    total_items = len(items)
    print(f"Starting ingestion of {total_items} items (batch_size={batch_size})...")

    indexed_count = 0
    for i in tqdm(range(0, total_items, batch_size), desc="Ingesting Batches"):
        batch = items[i : i + batch_size]

        # Extract captions and image paths
        captions = [item["caption"] for item in batch]
        image_paths = [item["image_path"] for item in batch]

        # Generate L2-normalized 512-D embeddings
        text_vectors = encoder.encode_text(captions)
        image_vectors = encoder.encode_image(image_paths)

        # Prepare records for Milvus upsert
        records = [
            {
                "id": item["id"],
                "image_vector": img_vec.tolist(),
                "text_vector": txt_vec.tolist(),
                "caption": item["caption"],
                "image_url": item["image_url"],
            }
            for item, img_vec, txt_vec in zip(batch, image_vectors, text_vectors)
        ]

        collection.upsert(records)
        indexed_count += len(records)

    collection.flush()
    print(f"Successfully indexed and flushed {indexed_count} items into Milvus.")
    return indexed_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest and embed dataset into Milvus.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of items to index (default: all items in styles.csv).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=INGEST_BATCH_SIZE,
        help=f"Batch size for embedding (default: {INGEST_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before indexing.",
    )
    args = parser.parse_args()

    col = get_or_create_collection(recreate=args.recreate)
    dataset_items = load_dataset_from_csv(limit=args.limit)
    index_dataset(dataset_items, batch_size=args.batch_size, collection=col)
