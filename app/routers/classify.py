"""
/classify — object detection only endpoint.
Ported from Flask app.py (Color detection removed).
"""
from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Any

import torch
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.models.clip_solver import CLIPSolver
from app.dependencies import get_solver

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Constants & Configuration ─────────────────────────────────────────────────

OBJECT_CLASSES = [
    # --- ANIMALS (Mammals) ---
    "cow", "lion", "tiger", "zebra", "elephant", "giraffe", "monkey", "panda", "bear",
    "koala", "kangaroo", "llama", "camel", "deer", "fox",
    "dog", "cat", "rabbit", "mouse", "pig", "sheep", "goat", "horse", "donkey", "hedgehog",
    "hippopotamus", "squirrel",

    # --- ANIMALS (Birds/Reptiles/Insects) ---
    "owl", "duck", "chicken", "penguin",
    "snake", "turtle", "crocodile",
    "fish", "shark", "whale", "dolphin", "robot",

    # --- TRANSPORT & VEHICLES ---
    "car", "boat", "ship",

    # --- TOYS & HOUSEHOLD ---
    "balloon", "piano", "guitar", "book",
    "phone", "computer", "camera",

    # --- NATURE & FOOD ---
    "tree", "flower", "star",
    "house", "turkey",
]

OBJECT_PROMPTS = [
    "a cartoon {obj}",
    "a cute cartoon {obj}",
    "a {obj} icon"
]


# ── Global Cache for Object Embeddings ────────────────────────────────────────

_obj_text_features: torch.Tensor | None = None


def get_descriptive_label(c: str) -> str:
    if c == "donkey": return "donkey with long ears"
    if c == "llama":  return "llama with long neck"
    if c == "camel":  return "camel with humps"
    if c == "horse":  return "horse pony"
    return c


# ── API Endpoint ──────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    image: str  # Base64 string

@router.post("/classify")
async def classify(payload: ClassifyRequest, solver: CLIPSolver = Depends(get_solver)):
    try:
        # 1. Pre-compute text features if needed
        global _obj_text_features
        if _obj_text_features is None:
            logger.info("Computing text features for classification...")
            all_obj_features = []
            for prompt_template in OBJECT_PROMPTS:
                prompts = [prompt_template.format(obj=get_descriptive_label(c)) for c in OBJECT_CLASSES]
                features = solver.embed_texts(prompts)  # (N_classes, D)
                all_obj_features.append(features)
            
            # Average features from different templates
            # stack -> (3, N, D) -> mean(0) -> (N, D)
            stacked = torch.stack(all_obj_features)
            mean_feats = stacked.mean(dim=0)
            _obj_text_features = mean_feats / mean_feats.norm(dim=-1, keepdim=True)
            logger.info("Classification text features ready.")

        # 2. Process Image
        try:
            full_image = solver.decode_image_b64(payload.image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        w, h = full_image.size
        cell_w, cell_h = w // 3, h // 3
        padding = 8
        
        cells_images = []
        
        # 3. Split 3x3 Grid
        for idx in range(9):
            r, c = divmod(idx, 3)
            left  = c * cell_w + padding
            top   = r * cell_h + padding
            right = (c + 1) * cell_w - padding
            bottom= (r + 1) * cell_h - padding
            
            # Ensure valid crop box
            if right > left and bottom > top:
                cell = full_image.crop((left, top, right, bottom))
            else:
                # Fallback if padding is too aggressive for small images
                cell = full_image.crop((c*cell_w, r*cell_h, (c+1)*cell_w, (r+1)*cell_h))
            
            cells_images.append(cell)

        # 4. CLIP Embedding for all 9 cells
        img_feats = solver.embed_images(cells_images)  # (9, D)

        # 5. Determine Top Objects
        temperature = 0.01
        
        # Ensure descriptors are on same device as images
        text_feats = _obj_text_features.to(img_feats.device)
        
        obj_logits = (img_feats @ text_feats.T) / temperature
        obj_probs = obj_logits.softmax(dim=-1)
        
        # Top-3 per cell
        top_conf, top_idx = torch.topk(obj_probs, k=3, dim=-1)
        
        top_conf = top_conf.cpu().tolist()
        top_idx = top_idx.cpu().tolist()

        results = []
        for i in range(9):
            r, c = divmod(i, 3)
            
            # Filter matches
            cell_confs = top_conf[i]
            cell_indices = top_idx[i]
            
            pass_indices = [cell_indices[0]]
            # Add secondary matches if confident enough
            for j in range(1, 3):
                if cell_confs[j] >= 0.3 and cell_confs[j] >= (cell_confs[0] * 0.5):
                    pass_indices.append(cell_indices[j])
            
            pass_names = [OBJECT_CLASSES[idx] for idx in pass_indices]
            
            obj_name_str = " ".join(pass_names)
            
            result_item = {
                "grid_position": i,
                "object_name":   obj_name_str,  # No color prefix anymore
                "detected_objects": pass_names,
                "confidence":    round(cell_confs[0], 3),
                "bbox": [c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h]
            }
            results.append(result_item)
            
        return {"grid_size": "3x3", "detected_objects": results}

    except Exception as e:
        logger.exception("Classification error")
        raise HTTPException(status_code=500, detail=str(e))
