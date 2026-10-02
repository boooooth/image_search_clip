"""End-to-end demonstration of the multimodal CLIP + Milvus retrieval pipeline."""

import sys
from pathlib import Path
from PIL import Image

from image_search_clip.config import DATA_CSV_PATH, IMAGES_DIR
from image_search_clip.ingest import index_dataset, load_dataset_from_csv
from image_search_clip.milvus_manager import get_or_create_collection
from image_search_clip.model import CLIPEncoder
from image_search_clip.search import search


def print_results(query_title: str, query_type: str, results: list) -> None:
    """Pretty prints search results."""
    print("\n" + "=" * 80)
    print(f"RESULTS FOR: '{query_title}' [Type: {query_type}]")
    print("=" * 80)
    if not results:
        print("No matches found.")
        return

    print(f"{'Rank':<6}{'ID':<10}{'Score':<10}{'Caption':<40}{'Image Path'}")
    print("-" * 80)
    for rank, res in enumerate(results, start=1):
        item_id = res.get("id", "N/A")
        score = res.get("score", 0.0)
        caption = str(res.get("caption", ""))[:38]
        image_url = str(res.get("image_url", ""))
        print(f"{rank:<6}{item_id:<10}{score:<10.4f}{caption:<40}{image_url}")
    print("=" * 80)


def create_mock_items(count: int = 5) -> list:
    """Creates synthetic fallback items with PIL images if dataset files are unavailable."""
    print("Generating synthetic mock items for testing...")
    mock_items = []
    colors = [
        ("red", (255, 0, 0), "Red sports running shoes"),
        ("blue", (0, 0, 255), "Blue casual denim jeans"),
        ("green", (0, 255, 0), "Green casual cotton t-shirt"),
        ("black", (0, 0, 0), "Black luxury stainless watch"),
        ("white", (255, 255, 255), "White formal office shirt"),
    ]

    mock_dir = Path("mock_images")
    mock_dir.mkdir(exist_ok=True)

    for i, (name, rgb, caption) in enumerate(colors[:count], start=1):
        img_path = mock_dir / f"mock_{i}.jpg"
        img = Image.new("RGB", (224, 224), color=rgb)
        img.save(img_path)
        mock_items.append({
            "id": i,
            "image_path": str(img_path),
            "caption": caption,
            "image_url": str(img_path),
        })
    return mock_items


def run_demo(sample_size: int = 25) -> None:
    """Runs end-to-end pipeline demonstration:
    1. Loads / Recreates Milvus Collection
    2. Initializes shared CLIPEncoder on CPU
    3. Indexes sample dataset items
    4. Performs Text Query hybrid search
    5. Performs Image Query hybrid search
    """
    print("\n--- Starting Multimodal Pipeline Demo ---\n")

    # 1. Initialize Collection & Model
    collection = get_or_create_collection(recreate=True)
    encoder = CLIPEncoder()

    # 2. Load dataset items
    items = []
    if DATA_CSV_PATH.is_file() and IMAGES_DIR.is_dir():
        items = load_dataset_from_csv(DATA_CSV_PATH, IMAGES_DIR, limit=sample_size)

    if not items:
        print(f"Warning: Could not load items from {DATA_CSV_PATH}. Using mock items.")
        items = create_mock_items(count=5)

    # 3. Index items into Milvus
    index_dataset(items, batch_size=16, collection=collection, encoder=encoder)

    # 4. Online Search: Text Query
    text_query = "navy blue shirt"
    print(f"\n[1] Executing Text Search for: '{text_query}' (Weights: img=0.3, txt=0.7)...")
    text_results = search(
        query_input=text_query,
        query_type="text",
        top_k=5,
        collection=collection,
        encoder=encoder,
    )
    print_results(text_query, "text", text_results)

    # 5. Online Search: Image Query
    # Use the first indexed item's image as the query
    sample_image_path = items[0]["image_path"]
    sample_item_caption = items[0]["caption"]
    print(f"\n[2] Executing Image Search using image from item ID {items[0]['id']} ('{sample_item_caption}')...")
    print(f"    (Weights: img=0.8, txt=0.2)")

    image_results = search(
        query_input=sample_image_path,
        query_type="image",
        top_k=5,
        collection=collection,
        encoder=encoder,
    )
    print_results(f"Image ({items[0]['id']}: {sample_item_caption})", "image", image_results)

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    count = 25
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            pass
    run_demo(sample_size=count)
