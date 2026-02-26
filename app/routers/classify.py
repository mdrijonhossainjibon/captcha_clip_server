"""
/classify — Ultra-accurate and ULTRA-FAST (MobileCLIP Optimized)
"""
from __future__ import annotations

import logging
import time
import torch
import re
import hashlib
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

# ── Auth Cache ─────────────────────────────────────────────────────────────────
_auth_cache: Dict[str, tuple] = {}
_AUTH_TTL = 60

def _get_auth(api_key: str, db):
    now = time.monotonic()
    if api_key in _auth_cache:
        val, ts = _auth_cache[api_key]
        if now - ts < _AUTH_TTL:
            return val
        else:
            del _auth_cache[api_key]
            
    pipeline = [
        {"$match": {"key": api_key, "status": "active"}},
        {"$lookup": {"from": "users", "localField": "userId", "foreignField": "_id", "as": "user"}},
        {"$unwind": "$user"},
        {"$lookup": {
            "from": "packages",
            "let": {"uid": "$userId"},
            "pipeline": [
                {"$match": {"$expr": {"$and": [
                    {"$eq": ["$userId", "$$uid"]},
                    {"$eq": ["$status", "active"]},
                    {"$gt": ["$endDate", datetime.utcnow()]}
                ]}}}
            ],
            "as": "pkg"
        }},
        {"$unwind": "$pkg"}
    ]
    res = list(db.apikeys.aggregate(pipeline))
    if not res: return None
    
    _auth_cache[api_key] = (res[0], now)
    return res[0]

def _bill_credit(pkg_id, db):
    db.packages.update_one({"_id": pkg_id}, {"$inc": {"creditsUsed": 1}})

def prewarm_features(solver: MobileCLIPSolver):
    global _cached_obj_features, _cached_color_features
    if _cached_obj_features is not None: return
    
    obj_prompts = [f"a photo of a {c}" for c in OBJECT_CLASSES]
    _cached_obj_features = solver.embed_texts(obj_prompts).to(torch.float32)
    
    color_prompts = [f"this is a {c} colored object" for c in COLOR_LIST]
    _cached_color_features = solver.embed_texts(color_prompts).to(torch.float32)

class ClassifyRequest(BaseModel):
    image: Optional[str] = None
    imageData: Optional[str] = None
    question: Optional[str] = ""

@router.post("/classify")
async def classify(
    payload: ClassifyRequest, 
    background_tasks: BackgroundTasks, 
    solver: MobileCLIPSolver = Depends(get_solver),
    api_key: Optional[str] = Header(None, alias="api-key")
):
    db = get_mongodb()
    
    if not api_key:
        return {"success": False, "error": {"code": 1001, "message": "No API Key"}}

    try:
        auth_data = _get_auth(api_key, db)
        if not auth_data:
            return {"success": False, "error": {"code": 1001, "message": "Invalid Key/Package"}}

        active_pkg = auth_data["pkg"]

        # Credits Check
        credits_limit = active_pkg.get("credits", 0)
        credits_used  = active_pkg.get("creditsUsed", 0)
        if credits_used >= credits_limit:
            return {"success": False, "error": {"code": 4029, "message": "Credits exhausted"}}

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
        
        # Object Detection
        obj_probs = (img_feats @ _cached_obj_features.T).softmax(dim=-1)
        _, top_obj_indices = torch.topk(obj_probs, k=1, dim=-1)

        # Color Detection
        color_probs = (img_feats @ _cached_color_features.T).softmax(dim=-1)
        _, top_color_indices = torch.topk(color_probs, k=1, dim=-1)

        solution = []
        q = payload.question.lower()
        
        # Extract target object and color from question
        target_objs = [obj for obj in OBJECT_CLASSES if obj in q]
        target_colors = [color for color in COLOR_LIST if color in q]

        for i in range(9):
            detected_obj = OBJECT_CLASSES[top_obj_indices[i].item()]
            detected_color = COLOR_LIST[top_color_indices[i].item()]
            
            # Logic: 
            # 1. If both color and object are in question, match both (e.g., "red car")
            # 2. If only color is in question, match color (e.g., "red objects")
            # 3. If only object is in question, match object (e.g., "all cars")
            
            match_obj = any(target_obj == detected_obj for target_obj in target_objs) if target_objs else True
            match_color = any(target_color == detected_color for target_color in target_colors) if target_colors else True
            
            # Special case: if nothing specifically matched but we have targets, it's a fail for this cell
            # If we have targets but they don't match, match_x will be false.
            if target_objs or target_colors:
                if (not target_objs or match_obj) and (not target_colors or match_color):
                    if target_objs or target_colors: # Ensure at least one constraint was active
                         solution.append(i + 1)
            else:
                # If no keywords found in question, fallback to keyword in q (old logic)
                if detected_obj in q or detected_color in q:
                    solution.append(i + 1)

        final_response = {"success": True, "solution": solution}

        # ── Log & Billing ───────────────────────────────────────────────────
        background_tasks.add_task(
            db.solutions.insert_one,
            {
                "hash": hashlib.sha256(f"{payload.question}:{img_b64}".encode()).hexdigest(),
                "solution": solution,
                "question": payload.question,
                "imageData": [img_b64],
                "type": "classify",
                "service": "classify",
                "createdAt": datetime.utcnow()
            }
        )
        background_tasks.add_task(_bill_credit, active_pkg["_id"], db)

        return final_response

    except Exception as e:
        logger.exception("Internal error in /classify")
        return {"success": False, "message": str(e)}
