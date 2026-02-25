"""
MobileCLIP Solver — Extreme Speed for CPU
Model: MobileCLIP-S2 (Apple)
Features:
  - Super lightweight MobileNet backbone
  - Batch processing for 9 grids
  - Cache for prompt features
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from functools import lru_cache
from typing import List

import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# CPU Speed Tuning
_cpu_count = os.cpu_count() or 4
try:
    torch.set_num_threads(_cpu_count)
except RuntimeError:
    pass

# Model Config
# Apple's MobileCLIP is very fast on CPU
_MODEL_NAME = "MobileCLIP-S2"
_PRETRAINED = "datacompdr"

_model = None
_preprocess = None
_tokenizer = None
_device = "cpu"

def _load_model():
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return
    logger.info(f"Loading {_MODEL_NAME} for CPU speed ...")
    _model, _, _preprocess = open_clip.create_model_and_transforms(_MODEL_NAME, pretrained=_PRETRAINED)
    _model = _model.to(_device).eval()
    _tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
    logger.info("MobileCLIP ready ✓")

def _decode_image(b64: str) -> Image.Image:
    b64 = re.sub(r"^data:image/[^;]+;base64,", "", b64)
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")

@lru_cache(maxsize=128)
def _get_text_features(text: str):
    tokens = _tokenizer([text]).to(_device)
    with torch.inference_mode():
        feats = _model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats

class MobileCLIPSolver:
    def __init__(self):
        _load_model()

    def solve_grid(self, image_chunks: List[str], question: str) -> dict:
        # 1. Prepare Text Query (Positive + Negative)
        clean_q = re.sub(r"^(select|click on|find|identify|all images with|all|the)\s+", "", question.lower()).strip()
        pos_text = f"a photo of {clean_q}"
        neg_text = f"a photo of something else"

        pos_feat = _get_text_features(pos_text)
        neg_feat = _get_text_features(neg_text)
        txt_feats = torch.cat([pos_feat, neg_feat], dim=0) # (2, D)

        # 2. Prepare 9 grid images
        images = [_decode_image(chunk) for chunk in image_chunks]
        img_tensors = torch.stack([_preprocess(img) for img in images]).to(_device)

        # 3. Predict BATCH
        with torch.inference_mode():
            img_feats = _model.encode_image(img_tensors)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            
            # (9, D) @ (D, 2) -> (9, 2)
            logits = (img_feats @ txt_feats.T)
            probs = logits.softmax(dim=-1)[:, 0].cpu().tolist() # Positive probabilities

        solution = []
        cell_scores = {}
        for i, p in enumerate(probs):
            cell_scores[str(i + 1)] = round(p * 100, 1)
            if p >= 0.50: # Threshold
                solution.append(i + 1)

        # Fallback
        if not solution:
            indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            solution = sorted([i + 1 for i, _ in indexed[:3]])

        return {
            "solution": solution,
            "cell_scores": cell_scores,
            "mean_confidence": round(sum(probs)/9 * 100, 1)
        }

    def solve_single(self, image_b64: str, question: str) -> dict:
        clean_q = re.sub(r"^(select|click on|find|identify|all images with|all|the)\s+", "", question.lower()).strip()
        pos_feat = _get_text_features(f"a photo of {clean_q}")
        neg_feat = _get_text_features("a photo of something else")
        txt_feats = torch.cat([pos_feat, neg_feat], dim=0)

        image = _decode_image(image_b64)
        img_tensor = _preprocess(image).unsqueeze(0).to(_device)

        with torch.inference_mode():
            img_feat = _model.encode_image(img_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            prob = (img_feat @ txt_feats.T).softmax(dim=-1)[0, 0].item()

        matched = prob >= 0.5
        return {
            "solution": [1] if matched else [],
            "confidence": round(prob * 100, 1),
            "matched": matched
        }

    def solve(self, image_data: str | List[str], question: str, question_type: str) -> dict:
        images = image_data if isinstance(image_data, list) else [image_data]
        if len(images) == 9:
            return self.solve_grid(images, question)
        return self.solve_single(images[0], question)

    def decode_image_b64(self, b64: str) -> Image.Image:
        return _decode_image(b64)

    # For compatibility with classify router
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        tokens = _tokenizer(texts).to(_device)
        with torch.inference_mode():
            feats = _model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([_preprocess(img) for img in images]).to(_device)
        with torch.inference_mode():
            feats = _model.encode_image(tensors)
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats
