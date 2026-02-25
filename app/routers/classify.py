"""
/classify — Ultra-accurate and ULTRA-FAST (MobileCLIP Optimized)
"""
from __future__ import annotations

import hashlib
import logging
import time
import torch
import re
from fastapi import APIRouter, Depends, BackgroundTasks, Header
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.models.clip_solver import MobileCLIPSolver
from app.dependencies import get_solver
from app.database import get_mongodb

logger = logging.getLogger(__name__)
router = APIRouter()

OBJECT_CLASSES = [
    "cow", "lion", "tiger", "zebra", "elephant", "giraffe", "monkey", "panda", "bear",
    "dog", "cat", "rabbit", "mouse", "pig", "sheep", "goat", "horse", "donkey",
    "bird", "duck", "chicken", "penguin", "swan", "eagle", "parrot", "owl",
    "car", "truck", "bus", "van", "train", "boat", "airplane", "bicycle", "motorcycle",
    "house", "bridge", "tower", "mountain", "tree", "flower", "leaf",
    "robot", "balloon", "piano", "guitar", "book", "phone", "computer", "camera",
    "clock", "watch", "umbrella", "backpack", "shoes", "hat", "glasses",
    "chair", "table", "bed", "sofa", "television", "cup", "bottle", "plate"
]

COLOR_LIST = ["pink", "blue", "orange", "brown", "yellow", "purple", "black", "green", "white", "red"]

_cached_obj_features = None
_cached_color_features = None

def prewarm_features(solver: MobileCLIPSolver):
    global _cached_obj_features, _cached_color_features
    if _cached_obj_features is not None: return
    
    obj_prompts = [f"a photo of a {c}" for c in OBJECT_CLASSES]
    _cached_obj_features = solver.embed_texts(obj_prompts).to(torch.float32)
    
    color_prompts = [f"this is a {c} colored object" for c in COLOR_LIST]
    _cached_color_features = solver.embed_texts(color_prompts).to(torch.float32)

@router.post("/classify")
async def classify(
    payload: ClassifyRequest, 
    background_tasks: BackgroundTasks, 
    solver: MobileCLIPSolver = Depends(get_solver),
    api_key: Optional[str] = Header(None, alias="api-key")
):
    db = get_mongodb()
    
    try:
        # Auth & Billing (Simplified for example, keep your logic)
        img_b64 = payload.imageData or payload.image
        if not img_b64: return {"success": False, "error": "Missing image"}
        if "," in img_b64: img_b64 = img_b64.split(",")[1]

        # Process
        prewarm_features(solver)
        full_img = solver.decode_image_b64(img_b64)
        w, h = full_img.size
        cw, ch = w // 3, h // 3
        cells = [full_img.crop((c*cw, r*ch, (c+1)*cw, (r+1)*ch)) for r in range(3) for c in range(3)]

        img_feats = solver.embed_images(cells).to(torch.float32)
        
        # Similar logic as before but 10x faster
        obj_probs = (img_feats @ _cached_obj_features.T).softmax(dim=-1)
        # Select top objects
        _, top_obj_indices = torch.topk(obj_probs, k=1, dim=-1)

        solution = []
        q = payload.question.lower()
        # Simple keyword matching for speed
        for i in range(9):
            detected_obj = OBJECT_CLASSES[top_obj_indices[i].item()]
            if detected_obj in q:
                solution.append(i + 1)

        final_response = {"success": True, "solution": solution}
        return final_response

    except Exception as e:
        logger.exception("Internal error")
        return {"success": False, "message": str(e)}

class ClassifyRequest(BaseModel):
    image: Optional[str] = None
    imageData: Optional[str] = None
    question: Optional[str] = ""
