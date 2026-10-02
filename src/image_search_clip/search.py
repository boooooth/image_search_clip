"""Online search and multi-vector hybrid re-ranking using Milvus WeightedRanker."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from pymilvus import AnnSearchRequest, Collection, WeightedRanker

from image_search_clip.config import (
    IMAGE_QUERY_WEIGHTS,
    INDEX_TYPE,
    METRIC_TYPE,
    TEXT_QUERY_WEIGHTS,
)
from image_search_clip.milvus_manager import get_or_create_collection
from image_search_clip.model import CLIPEncoder


def search(
    query_input: Union[str, Path, Image.Image],
    query_type: str = "text",
    top_k: int = 5,
    collection: Optional[Collection] = None,
    encoder: Optional[CLIPEncoder] = None,
    image_weight: Optional[float] = None,
    text_weight: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Executes a multimodal hybrid dual-vector search using Milvus native hybrid_search()

    and WeightedRanker.

    Args:
        query_input: Text string (if query_type="text") or Image / file path (if query_type="image")
        query_type: "text" or "image"
        top_k: Number of ranked results to return
        collection: Optional Collection instance (defaults to loaded collection)
        encoder: Optional CLIPEncoder instance
        image_weight: Custom weight for image_vector similarity (optional)
        text_weight: Custom weight for text_vector similarity (optional)

    Returns:
        List of dicts: [{'id': int, 'caption': str, 'image_url': str, 'score': float}]
    """
    if collection is None:
        collection = get_or_create_collection()

    if encoder is None:
        encoder = CLIPEncoder()

    query_type_clean = query_type.strip().lower()

    # 1. Embed query input and assign adaptive weights
    if query_type_clean == "text":
        if not isinstance(query_input, str):
            query_input = str(query_input)
        query_vector = encoder.encode_text(query_input)
        w_img = image_weight if image_weight is not None else TEXT_QUERY_WEIGHTS["image_weight"]
        w_txt = text_weight if text_weight is not None else TEXT_QUERY_WEIGHTS["text_weight"]

    elif query_type_clean == "image":
        query_vector = encoder.encode_image(query_input)
        w_img = image_weight if image_weight is not None else IMAGE_QUERY_WEIGHTS["image_weight"]
        w_txt = text_weight if text_weight is not None else IMAGE_QUERY_WEIGHTS["text_weight"]

    else:
        raise ValueError(f"Unsupported query_type '{query_type}'. Expected 'text' or 'image'.")

    # query_vector is of shape (1, 512). Convert to list of floats for AnnSearchRequest.
    query_vector_list = query_vector.tolist()

    # 2. Configure search parameters for ANN
    search_param = {
        "metric_type": METRIC_TYPE,
        "params": {} if INDEX_TYPE == "AUTOINDEX" else {"ef": 64},
    }

    # 3. Create two AnnSearchRequest objects:
    # Request 1: Query against image_vector
    req_image = AnnSearchRequest(
        data=query_vector_list,
        anns_field="image_vector",
        param=search_param,
        limit=top_k,
    )

    # Request 2: Query against text_vector
    req_text = AnnSearchRequest(
        data=query_vector_list,
        anns_field="text_vector",
        param=search_param,
        limit=top_k,
    )

    # 4. WeightedRanker fuses scores:
    # Final Score = (w_img * score_img) + (w_txt * score_txt)
    # The order of weights corresponds to [req_image, req_text]
    ranker = WeightedRanker(w_img, w_txt)

    # 5. Execute Milvus Native hybrid_search
    results = collection.hybrid_search(
        reqs=[req_image, req_text],
        rerank=ranker,
        limit=top_k,
        output_fields=["id", "caption", "image_url"],
    )

    # 6. Parse and format results
    formatted_results = []
    if results and len(results) > 0:
        hits = results[0]
        for hit in hits:
            # Extract fields safely across pymilvus versions
            entity_dict = {}
            if hasattr(hit, "entity"):
                entity = hit.entity
                for field in ["id", "caption", "image_url"]:
                    if hasattr(entity, "get"):
                        entity_dict[field] = entity.get(field)
                    elif hasattr(entity, field):
                        entity_dict[field] = getattr(entity, field)
            elif hasattr(hit, "fields"):
                entity_dict = hit.fields

            # Fallback for primary key id
            item_id = entity_dict.get("id") if entity_dict.get("id") is not None else hit.id
            score = getattr(hit, "score", getattr(hit, "distance", 0.0))

            formatted_results.append(
                {
                    "id": item_id,
                    "caption": entity_dict.get("caption", ""),
                    "image_url": entity_dict.get("image_url", ""),
                    "score": round(float(score), 4),
                }
            )

    return formatted_results
