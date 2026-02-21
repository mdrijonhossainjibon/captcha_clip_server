"""
/classify — High-speed object classification with A-Z color names and hex codes.
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict, Any

import torch
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.models.clip_solver import CLIPSolver
from app.dependencies import get_solver

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Constants ────────────────────────────────────────────────────────────────

OBJECT_CLASSES = [
    "cow", "lion", "tiger", "zebra", "elephant", "giraffe", "monkey", "panda", "bear",
    "koala", "kangaroo", "llama", "camel", "deer", "fox",
    "dog", "cat", "rabbit", "mouse", "pig", "sheep", "goat", "horse", "donkey", "hedgehog",
    "hippopotamus", "squirrel", "frog",
    "owl", "duck", "chicken", "penguin", "goose", "parrot", "bird",
    "snake", "turtle", "crocodile",
    "fish", "shark",   "dolphin", "robot",
    "car", "boat", "ship",
    "balloon", "piano", "guitar", "book",
    "phone", "computer", "camera",
    "tree", "flower", "star", "house", "turkey",
]

# Comprehensive A-Z Color mapping to Hex Codes
COLOR_MAP = {
    "amber": "#FFBF00", "aqua": "#00FFFF", "aquamarine": "#7FFFD4", "azure": "#F0FFFF",
    "beige": "#F5F5DC", "black": "#000000", "blue": "#0000FF", "bronze": "#CD7F32",
    "brown": "#A52A2A", "burgundy": "#800020", "cherry": "#DE3163",  
    "copper": "#B87333", "coral": "#FF7F50", "crimson": "#DC143C", "cyan": "#00FFFF",
    "emerald": "#50C878", "gold": "#FFD700", "gray": "#808080", "green": "#008000",
    "indigo": "#4B0082", "ivory": "#FFFFF0", "jade": "#00A86B", "khaki": "#C3B091",
    "lavender": "#E6E6FA", "lemon": "#FFF700", "lilac": "#C8A2C8", "lime": "#00FF00",
    "magenta": "#FF00FF", "maroon": "#800000", "mauve": "#E0B0FF", "mint": "#98FF98",
    "navy": "#000080", "olive": "#808000", "orange": "#FFA500", "peach": "#FFDAB9",
    "pear": "#D1E231", "pink": "#FFC0CB", "plum": "#8E4585", "purple": "#800080",
    "red": "#FF0000", "rose": "#FF007F", "ruby": "#E0115F", "salmon": "#FA8072",
    "scarlet": "#FF2400", "silver": "#C0C0C0", "sky": "#87CEEB", "tan": "#D2B48C",
    "teal": "#008080", "turquoise": "#40E0D0", "violet": "#EE82EE", "white": "#FFFFFF",
    "yellow": "#FFFF00"
}

COLOR_LIST = list(COLOR_MAP.keys())

def get_descriptive_label(c: str) -> str:
    """Provides semantic hints to CLIP for better recognition."""
    if c == "parrot":   return "a parrot bird or colorful macaw with a curved beak and vibrant feathers"
    if c == "owl":      return "an owl bird with big round eyes and feathers"
    if c == "goose":    return "a goose or water bird with a long neck and beak"
    if c == "camel":    return "a camel with a hump on its back"
    if c == "sheep":    return "a puffy woolly sheep or lamb with thick curly white fleece"
    if c == "goat":     return "a goat with horns and a small beard"
    if c == "rabbit":   return "a rabbit or bunny with long ears and fluffy fur"
    if c == "phone":    return "a smartphone or mobile phone with a touchscreen and a camera lens"
    if c == "mouse":    return "a mouse or rodent with large rounded ears, whiskers, and a thin tail"
    if c == "cat":      return "a cat or kitten with pointy ears, whiskers, and a long tail"
    if c == "elephant": return "a large elephant with a long trunk and big floppy ears"
    if c == "frog":     return "a frog or toad with bulging eyes, a squat body, and damp skin"
    return c

# ── Global Cache ─────────────────────────────────────────────────────────────
_cached_obj_features: torch.Tensor | None = None
_cached_color_features: torch.Tensor | None = None


class ClassifyRequest(BaseModel):
    image: str|None = None
    imageData: str|None = None
    question: str|None = None
    questionType: str|None = None

@router.post("/classify")
async def classify(payload: ClassifyRequest, solver: CLIPSolver = Depends(get_solver)):
    try:
        img_b64 = payload.imageData or payload.image
        if not img_b64: raise HTTPException(status_code=400, detail="Missing image")

        global _cached_obj_features, _cached_color_features
        if _cached_obj_features is None:
            obj_feats_list = []
            for pt in ["a photo of {}", "a cartoon {} character"]:
                prompts = [pt.format(get_descriptive_label(c)) for c in OBJECT_CLASSES]
                obj_feats_list.append(solver.embed_texts(prompts))
            _cached_obj_features = torch.stack(obj_feats_list).mean(dim=0).to(torch.float32)
            _cached_obj_features /= _cached_obj_features.norm(dim=-1, keepdim=True)
            
            color_prompts = [f"a {c} colored object" for c in COLOR_LIST]
            _cached_color_features = solver.embed_texts(color_prompts).to(torch.float32)
            _cached_color_features /= _cached_color_features.norm(dim=-1, keepdim=True)

        full_img = solver.decode_image_b64(img_b64)
        w, h = full_img.size
        cells = [full_img.crop((c*(w//3), r*(h//3), (c+1)*(w//3), (r+1)*(h//3))) for r in range(3) for c in range(3)]

        img_feats = solver.embed_images(cells).to(torch.float32)
        device = img_feats.device
        
        # 1. Base Object Detection
        obj_text_feats = _cached_obj_features.to(device)
        obj_probs = ((img_feats @ obj_text_feats.T) / 0.02).softmax(dim=-1)
        top_obj_conf, top_obj_idx = torch.topk(obj_probs, k=3, dim=-1)

        # 2. Advanced Color Detection
        color_text_feats = _cached_color_features.to(device)
        color_probs = ((img_feats @ color_text_feats.T) / 0.02).softmax(dim=-1)
        top_color_conf, top_color_idx = torch.topk(color_probs, k=1, dim=-1)

        # 3. Dynamic Question Matching
        answer = []
        if payload.question:
            q = payload.question.lower().strip()
            q_refined = q
            if "parrot" in q: q_refined = q.replace("parrot", "colorful parrot or macaw with curved beak")
            
            target_prompt = f"a photo of {q_refined}"
            neg_prompt = "background scene without the target object"
            
            q_feats = solver.embed_texts([target_prompt, neg_prompt]).to(device)
            q_probs = ((img_feats @ q_feats.T) / 0.02).softmax(dim=-1)
            
            bird_set = {"bird", "parrot", "duck", "owl", "penguin", "goose", "chicken"}
            keywords = [kw for kw in re.findall(r'\w+', q) if kw not in COLOR_LIST and kw not in ['a', 'the', 'select', 'all', 'of', 'and']]
            
            for i in range(9):
                # High threshold for precision (updated from 0.86 to 0.90)
                is_q_match = q_probs[i, 0] > 0.90 
                cell_objs = {OBJECT_CLASSES[top_obj_idx[i, 0].item()], OBJECT_CLASSES[top_obj_idx[i, 1].item()]}
                is_obj_related = False
                for kw in keywords:
                    if any(kw in ro or ro in kw for ro in cell_objs):
                        is_obj_related = True; break
                    if kw in bird_set and any(b in cell_objs for b in bird_set):
                        is_obj_related = True; break

                if is_q_match and is_obj_related:
                    answer.append(i)

        results = []
        for i in range(9):
            color_name = COLOR_LIST[top_color_idx[i, 0].item()]
            results.append({
                "grid_position": i,
                "object": OBJECT_CLASSES[top_obj_idx[i, 0].item()],
                "confidence": round(top_obj_conf[i, 0].item(), 3),
                "color": color_name,
                "color_code": COLOR_MAP[color_name]
            })

        return {"answer": answer, "detected_objects": results, "question": payload.question}

    except Exception as e:
        logger.exception("Classification error")
        raise HTTPException(status_code=500, detail=str(e))
