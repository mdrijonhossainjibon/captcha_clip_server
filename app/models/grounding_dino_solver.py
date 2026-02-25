"""
Grounding DINO — Optimized for CPU (Batch Processing)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from functools import lru_cache
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

logger = logging.getLogger(__name__)

# CPU thread optimization
_cpu_count = os.cpu_count() or 4
try:
    torch.set_num_threads(_cpu_count)
except RuntimeError:
    pass

_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_THRESHOLD = 0.30

_processor = None
_model = None
_device = "cpu"

def _load_model():
    global _processor, _model
    if _model is not None:
        return
    logger.info(f"Loading Grounding DINO '{_MODEL_ID}' on CPU ...")
    _processor = AutoProcessor.from_pretrained(_MODEL_ID)
    _model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID)
    _model = _model.to(_device).eval()
    logger.info("Grounding DINO loaded ✓")

def _decode_image(b64: str) -> Image.Image:
    b64 = re.sub(r"^data:image/[^;]+;base64,", "", b64)
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")

def _build_text_query(question: str) -> str:
    q = question.strip().lower().rstrip("?.")
    # Clean question to extract object
    q = re.sub(r"^(select|click on|find|identify|all images with|all|the)\s+", "", q)
    q = re.sub(r"^(a |an |the )", "", q).strip()
    return q + "."

class GroundingDINOSolver:
    def __init__(self):
        _load_model()

    def solve_grid(self, image_chunks: List[str], question: str) -> dict:
        text_query = _build_text_query(question)
        images = [_decode_image(chunk) for chunk in image_chunks]
        
        # 🚀 BATCH PROCESSING: Process all 9 images at once for speed
        inputs = _processor(images=images, text=[text_query]*len(images), return_tensors="pt").to(_device)
        
        with torch.inference_mode():
            outputs = _model(**inputs)

        # Handle threshold parameters safely for different transformer versions
        process_args = {
            "outputs": outputs,
            "input_ids": inputs["input_ids"],
            "target_sizes": [img.size[::-1] for img in images]
        }
        
        # Determine if we should use 'box_threshold' or just 'threshold' based on error feedback
        # Falling back to simpler post-processing if keywords fail
        try:
            results = _processor.post_process_grounded_object_detection(**process_args, box_threshold=_THRESHOLD)
        except TypeError:
            results = _processor.post_process_grounded_object_detection(**process_args, threshold=_THRESHOLD)

        solution = []
        cell_scores = {}
        for i, res in enumerate(results):
            score = float(res["scores"].max().item()) if len(res["scores"]) > 0 else 0.0
            cell_scores[str(i + 1)] = round(score * 100, 1)
            if score >= _THRESHOLD:
                solution.append(i + 1)

        # Fallback if nothing detected
        if not solution:
            sorted_cells = sorted(cell_scores.items(), key=lambda x: x[1], reverse=True)
            solution = sorted([int(k) for k, _ in sorted_cells[:3]])

        return {
            "solution": solution,
            "detected_object": text_query.rstrip("."),
            "cell_scores": cell_scores,
            "mean_confidence": round(sum(cell_scores.values())/9, 1)
        }

    def solve_single(self, image_b64: str, question: str) -> dict:
        text_query = _build_text_query(question)
        image = _decode_image(image_b64)
        inputs = _processor(images=image, text=text_query, return_tensors="pt").to(_device)
        
        with torch.inference_mode():
            outputs = _model(**inputs)

        try:
            results = _processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], box_threshold=_THRESHOLD, target_sizes=[image.size[::-1]]
            )[0]
        except TypeError:
            results = _processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=_THRESHOLD, target_sizes=[image.size[::-1]]
            )[0]

        score = float(results["scores"].max().item()) if len(results["scores"]) > 0 else 0.0
        matched = score >= _THRESHOLD
        return {
            "solution": [1] if matched else [],
            "confidence": round(score * 100, 1),
            "matched": matched
        }

    def solve(self, image_data: str | List[str], question: str, question_type: str) -> dict:
        images = image_data if isinstance(image_data, list) else [image_data]
        if question_type == "gridcaptcha" or len(images) == 9:
            return self.solve_grid(images, question)
        return self.solve_single(images[0], question)

    def decode_image_b64(self, b64: str) -> Image.Image:
        return _decode_image(b64)
