"""CLIP model wrapper for generating L2-normalized text and image embeddings."""

from pathlib import Path
from typing import Any, List, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from image_search_clip.config import DEVICE, MODEL_NAME


def _extract_embedding(outputs: Union[torch.Tensor, Any]) -> torch.Tensor:
    """Extracts the embedding tensor across transformers 4.x and 5.x return types."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        return outputs.text_embeds
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state[:, 0, :]
    return outputs[0]


class CLIPEncoder:
    """Encodes text and images into 512-D L2-normalized vectors using OpenAI CLIP."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.device = device
        device_desc = f"{self.device} ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else self.device
        print(f"Loading CLIP model '{model_name}' on device '{device_desc}'...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text or a list of texts into L2-normalized 512-D vectors.

        Args:
            texts: Single text prompt or list of strings.

        Returns:
            np.ndarray of shape (N, 512) with dtype float32 and unit norm.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            text_features = _extract_embedding(outputs)
            # L2 normalize across embedding dimension (dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features.cpu().numpy().astype(np.float32)

    def encode_image(
        self, images: Union[Image.Image, str, Path, List[Union[Image.Image, str, Path]]]
    ) -> np.ndarray:
        """Encode an image or list of images into L2-normalized 512-D vectors.

        Args:
            images: Single PIL Image/path or list of PIL Images/paths.

        Returns:
            np.ndarray of shape (N, 512) with dtype float32 and unit norm.
        """
        if not isinstance(images, list):
            images = [images]

        loaded_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                loaded_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                loaded_images.append(img.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

        inputs = self.processor(
            images=loaded_images,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            image_features = _extract_embedding(outputs)
            # L2 normalize across embedding dimension (dim=-1)
            image_features = F.normalize(image_features, p=2, dim=-1)

        return image_features.cpu().numpy().astype(np.float32)
