"""Milvus collection management, schema definitions, and indexing."""

from typing import Optional
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from image_search_clip.config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    INDEX_TYPE,
    METRIC_TYPE,
    MILVUS_URI,
)


def connect_milvus(uri: str = MILVUS_URI) -> None:
    """Connect to Milvus / Milvus Lite instance."""
    # Check if default connection already exists
    if not connections.has_connection("default"):
        print(f"Connecting to Milvus at: {uri}")
        connections.connect(alias="default", uri=uri)
    else:
        print("Using existing Milvus connection ('default').")


def get_collection_schema() -> CollectionSchema:
    """Defines the multi-vector schema for multimodal retrieval:

    - id: Primary Key (INT64, auto_id=False)
    - image_vector: 512-D float vector from CLIP vision encoder
    - text_vector: 512-D float vector from CLIP text encoder
    - caption: Product description or title (VARCHAR, max 512)
    - image_url: Local or remote image URI (VARCHAR, max 512)
    """
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
            description="Unique identifier",
        ),
        FieldSchema(
            name="image_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
            description="CLIP 512-D Vision Embedding",
        ),
        FieldSchema(
            name="text_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
            description="CLIP 512-D Text Embedding",
        ),
        FieldSchema(
            name="caption",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Item caption / product display name",
        ),
        FieldSchema(
            name="image_url",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Item image path or URL",
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description="Multimodal Collection with CLIP dual vectors",
        enable_dynamic_field=False,
    )


def create_indices(collection: Collection) -> None:
    """Build indices on both image_vector and text_vector fields."""
    index_params = {
        "metric_type": METRIC_TYPE,
        "index_type": INDEX_TYPE,
        "params": {} if INDEX_TYPE == "AUTOINDEX" else {"M": 16, "efConstruction": 200},
    }

    print(f"Building index for 'image_vector' ({INDEX_TYPE}, {METRIC_TYPE})...")
    collection.create_index(field_name="image_vector", index_params=index_params)

    print(f"Building index for 'text_vector' ({INDEX_TYPE}, {METRIC_TYPE})...")
    collection.create_index(field_name="text_vector", index_params=index_params)


def get_or_create_collection(
    collection_name: str = COLLECTION_NAME, recreate: bool = False
) -> Collection:
    """Retrieves existing collection or creates a new one with indices loaded into memory.

    Args:
        collection_name: Name of the collection.
        recreate: If True, drops any existing collection with this name.

    Returns:
        Loaded pymilvus.Collection instance ready for search.
    """
    connect_milvus()

    if utility.has_collection(collection_name):
        if recreate:
            print(f"Dropping existing collection '{collection_name}'...")
            utility.drop_collection(collection_name)
        else:
            print(f"Found existing collection '{collection_name}'. Loading...")
            collection = Collection(collection_name)
            collection.load()
            return collection

    print(f"Creating new collection '{collection_name}'...")
    schema = get_collection_schema()
    collection = Collection(name=collection_name, schema=schema)

    create_indices(collection)
    collection.load()
    print(f"Collection '{collection_name}' created, indexed, and loaded successfully.")
    return collection
