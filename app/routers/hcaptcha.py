"""
/hcaptcha  — hCaptcha Solver (CaptchaMaster Extension Compatible)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Exact payload sent by CaptchaMaster extension (background.ts):
  POST /hcaptcha
  Headers:
    X-CSRF-Token: <token>
    API-KEY: <apiKey>
  Body:
  {
    "imageData"    : ["b64img1", "b64img2", ...],   ← array of base64 tile images
    "question"     : "Please click each image containing a bird",
    "questionType" : "objectClassify" | "objectClick" | "objectDrag" | "objectTag",
    "tileCount"    : 9
  }

  Response (success):
  {
    "success": true,
    "solution": <depends on questionType>
  }

  solution format per questionType:
  ┌──────────────────┬──────────────────────────────────────────────────────┐
  │ objectClassify   │ boolean[]  — length 9, true = click that cell        │
  │                  │ e.g. [true, false, true, false, false, true, ...]    │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │ objectClick      │ { x: number, y: number }[]                           │
  │                  │ e.g. [{ x: 120, y: 240 }]                            │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │ objectDrag       │ { start: [x,y], end: [x,y] }[]                       │
  │                  │ e.g. [{ start: [412, 82], end: [150, 200] }]         │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │ objectTag        │ string[]  — e.g. ["bus", "truck"]                    │
  └──────────────────┴──────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import APIRouter, BackgroundTasks, Depends, Header
from pydantic import BaseModel

from app.database import get_mongodb
from app.dependencies import get_solver
from app.models.clip_solver import MobileCLIPSolver, _get_text_features

logger = logging.getLogger(__name__)
router = APIRouter()

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
        {"$lookup": {"from": "users", "localField": "userId",
                     "foreignField": "_id", "as": "user"}},
        {"$unwind": "$user"},
        {"$lookup": {
            "from": "packages",
            "let": {"uid": "$userId"},
            "pipeline": [{"$match": {"$expr": {"$and": [
                {"$eq": ["$userId", "$$uid"]},
                {"$eq": ["$status", "active"]},
                {"$gt": ["$endDate", datetime.utcnow()]}
            ]}}}],
            "as": "pkg"
        }},
        {"$unwind": "$pkg"}
    ]
    res = list(db.apikeys.aggregate(pipeline))
    if not res:
        return None
    _auth_cache[api_key] = (res[0], now)
    return res[0]


def _bill_credit(pkg_id, db):
    """Increment creditsUsed by 1 for the given package."""
    db.packages.update_one({"_id": pkg_id}, {"$inc": {"creditsUsed": 1}})


# ── Request Model (exact match to CaptchaMaster extension) ────────────────────
class HCaptchaRequest(BaseModel):
    imageData:    List[str]          # array of base64 tile images (no data: prefix needed)
    question:     str = ""           # challenge question text
    questionType: str = "objectClassify"  # objectClassify | objectClick | objectDrag | objectTag
    tileCount:    Optional[int] = 9  # number of tiles (usually 9)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _clean_question(q: str) -> str:
    """Strip hCaptcha boilerplate from question to get core subject."""
    q = q.lower().strip()
    q = re.sub(
        r"^(please\s+)?(select|click\s+(each|all|on)|find|identify"
        r"|all\s+images?\s+(with|containing?)|all|the)\s+",
        "", q
    ).strip()
    return q or q


def _b64_strip(uri: str) -> str:
    """Remove data:image/...;base64, prefix if present."""
    if "," in uri:
        return uri.split(",", 1)[1]
    return uri


# ══════════════════════════════════════════════════════════════════════════════
#  TYPE 1 — objectClassify  (3×3 Grid Click)
#  Returns: boolean[]  — true = click this tile
# ══════════════════════════════════════════════════════════════════════════════
# ── Character Recognition ───────────────────────────────────────────────────
_CHAR_LIST = "abcdefghijklmnopqrstuvwxyz0123456789"
_cached_char_feats = None

def _get_char_features(solver: MobileCLIPSolver):
    global _cached_char_feats
    if _cached_char_feats is not None: return _cached_char_feats
    prompts = [f"a photo of the letter {c}" if c.isalpha() else f"a photo of the number {c}" for c in _CHAR_LIST]
    _cached_char_feats = solver.embed_texts(prompts).to(torch.float32)
    return _cached_char_feats

def _solve_duplicate_characters(
    solver: MobileCLIPSolver,
    images: List[str],
) -> List[bool]:
    """Identify characters appearing more than once in the grid."""
    cells = [solver.decode_image_b64(_b64_strip(img)) for img in images]
    img_feats = solver.embed_images(cells).to(torch.float32)
    char_feats = _get_char_features(solver)

    # (9, D) @ (D, 36) -> (9, 36)
    probs = (img_feats @ char_feats.T).softmax(dim=-1)
    best_char_indices = probs.argmax(dim=-1).cpu().tolist()
    
    detected_chars = [_CHAR_LIST[idx] for idx in best_char_indices]
    counts = {}
    for c in detected_chars:
        counts[c] = counts.get(c, 0) + 1
    
    # Identify which cells contain characters that appear > 1 time
    results = [counts[c] > 1 for c in detected_chars]
    
    logger.info("duplicateCharacters — detected: %s | counts: %s | selected: %d cells", 
                detected_chars, counts, sum(results))
    return results

def _solve_objectClassify(
    solver: MobileCLIPSolver,
    images: List[str],
    question: str,
) -> List[bool]:
    from PIL import Image

    if "more than once" in question.lower() or "repeated" in question.lower():
        return _solve_duplicate_characters(solver, images)

    cells = [solver.decode_image_b64(_b64_strip(img)) for img in images]
    clean_q = _clean_question(question)

    pos_feat = _get_text_features(f"a photo of {clean_q}")
    neg_feat = _get_text_features("a photo of something else, not the target")
    txt_feats = torch.cat([pos_feat, neg_feat], dim=0).to(torch.float32)

    img_feats = solver.embed_images(cells).to(torch.float32)
    probs = (img_feats @ txt_feats.T).softmax(dim=-1)[:, 0].cpu().tolist()

    THRESHOLD = 0.50
    results = [p >= THRESHOLD for p in probs]

    # Fallback: at least 1 must be True
    if not any(results):
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        results[best_idx] = True

    logger.info(
        "objectClassify — q: %r | probs: %s | selected: %d cells",
        clean_q,
        [round(p * 100, 1) for p in probs],
        sum(results),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  TYPE 2 — objectClick  (Bounding Box Click)
#  Returns: { x: number, y: number }[]
#  Content uses: dispatchMouseEvent(canvas, "click", cx, cy)
#  → coordinates are pixel offsets from top-left of canvas
# ══════════════════════════════════════════════════════════════════════════════
def _solve_duplicate_icons(
    solver: MobileCLIPSolver,
    images: List[str],
) -> List[Dict[str, float]]:
    """Find visually identical icons in a single canvas image."""
    img = solver.decode_image_b64(_b64_strip(images[0]))
    W, H = img.size
    
    # 1. Detect all candidate objects using a generic prompt
    pos_feat = _get_text_features("a small icon or object")
    neg_feat = _get_text_features("background, empty area")
    txt_feats = torch.cat([pos_feat, neg_feat], dim=0).to(torch.float32)

    candidates = []
    # Use multiple window sizes to capture different icon scales
    for wf in [0.15, 0.22, 0.30]:
        ww = int(W * wf)
        wh = ww # Icons are usually square
        sx = sy = int(ww * 0.5) # 50% overlap for better coverage
        
        crops, coords = [], []
        for x in range(0, W - ww + 1, sx):
            for y in range(0, H - wh + 1, sy):
                crops.append(img.crop((x, y, x + ww, y + wh)))
                coords.append((x + ww // 2, y + wh // 2))
        
        if crops:
            feats = solver.embed_images(crops).to(torch.float32)
            # p is probability of being an 'object'
            p_list = (feats @ txt_feats.T).softmax(dim=-1)[:, 0].cpu().tolist()
            for i, p in enumerate(p_list):
                if p > 0.65: # High threshold for candidates
                    candidates.append({
                        "feat": feats[i], 
                        "x": float(coords[i][0]), 
                        "y": float(coords[i][1])
                    })

    if not candidates: return []

    # 2. Group candidates by distance to find unique objects
    unique_objects = []
    MIN_DIST = W * 0.12
    for cand in candidates:
        is_new = True
        for obj in unique_objects:
            dist = ((cand["x"] - obj["x"])**2 + (cand["y"] - obj["y"])**2)**0.5
            if dist < MIN_DIST:
                # Merge with existing: keep the one with higher confidence (if we had it)
                is_new = False; break
        if is_new: unique_objects.append(cand)

    # 3. Find pairs with high visual similarity
    final_points = []
    if len(unique_objects) < 2: return []

    # Compare every object with every other object
    for i in range(len(unique_objects)):
        for j in range(i + 1, len(unique_objects)):
            feat1 = unique_objects[i]["feat"]
            feat2 = unique_objects[j]["feat"]
            # Cosine similarity
            sim = (feat1 @ feat2.T).item()
            
            if sim > 0.88: # Very high similarity threshold for "identical" icons
                logger.info("Found duplicate icons! sim: %0.3f", sim)
                final_points.append({"x": unique_objects[i]["x"], "y": unique_objects[i]["y"]})
                final_points.append({"x": unique_objects[j]["x"], "y": unique_objects[j]["y"]})

    # Remove duplicate coordinates in our final list
    seen = set()
    deduped = []
    for p in final_points:
        key = (round(p["x"]), round(p["y"]))
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    logger.info("duplicateIcons — detected %d repeating points", len(deduped))
    return deduped

def _solve_objectClick(
    solver: MobileCLIPSolver,
    images: List[str],
    question: str,
) -> List[Dict[str, float]]:
    """CLIP-based sliding window for coordinate detection (returns multiple hits)."""
    if "more than once" in question.lower() or "repeated" in question.lower() or "multiple times" in question.lower():
        return _solve_duplicate_icons(solver, images)

    img = solver.decode_image_b64(_b64_strip(images[0]))
    W, H = img.size
    clean_q = _clean_question(question)
    
    pos_feat = _get_text_features(f"a photo of {clean_q}")
    neg_feat = _get_text_features("background, empty area")
    txt_feats = torch.cat([pos_feat, neg_feat], dim=0).to(torch.float32)

    win_fracs = [(0.15, 0.15), (0.25, 0.25), (0.35, 0.35)]
    stride_f = 0.12
    
    all_results = []

    for wf, hf in win_fracs:
        ww, wh = int(W * wf), int(H * hf)
        sx, sy = max(1, int(W * stride_f)), max(1, int(H * stride_f))
        crops, coords = [], []
        for x in range(0, W - ww + 1, sx):
            for y in range(0, H - wh + 1, sy):
                crops.append(img.crop((x, y, x + ww, y + wh)))
                coords.append((x + ww // 2, y + wh // 2))

        if crops:
            feats = solver.embed_images(crops).to(torch.float32)
            probs = (feats @ txt_feats.T).softmax(dim=-1)[:, 0].cpu().tolist()
            for p, (cx, cy) in zip(probs, coords):
                if p > 0.45:  # Threshold for detection
                    all_results.append({"p": p, "x": float(cx), "y": float(cy)})

    # ── Non-Maximum Suppression (Distance based) ──────────────────────────
    # Sort by probability
    all_results.sort(key=lambda x: x["p"], reverse=True)
    
    final_points = []
    MIN_DIST = W * 0.15 # 15% of width as minimum distance between clicks

    for res in all_results:
        is_too_close = False
        for kept in final_points:
            dist = ((res["x"] - kept["x"])**2 + (res["y"] - kept["y"])**2)**0.5
            if dist < MIN_DIST:
                is_too_close = True
                break
        if not is_too_close:
            final_points.append(res)
            if len(final_points) >= 5: break # Cap at 5 points to prevent spam

    if not final_points:
        return [{"x": float(W // 2), "y": float(H // 2)}]

    logger.info("objectClick (CLIP) — target: %r | detected: %d distinct points",
                clean_q, len(final_points))
    
    # Return just the coordinates
    return [{"x": float(r["x"]), "y": float(r["y"])} for r in final_points]


# ══════════════════════════════════════════════════════════════════════════════
#  TYPE 3 — objectDrag  (Bounding Box Drag-Drop)
#  Returns: { start: [x, y], end: [x, y] }[]
#  Content uses: mousedown at start → smooth mousemove → mouseup at end
# ══════════════════════════════════════════════════════════════════════════════
def _solve_objectDrag(
    solver: MobileCLIPSolver,
    images: List[str],
    question: str,
) -> List[Dict[str, Any]]:
    """CLIP-based drag detection."""
    scene_img = solver.decode_image_b64(_b64_strip(images[0]))
    W, H = scene_img.size
    clean_q = _clean_question(question)

    # 1. Find target area (end point)
    target_res = _solve_objectClick(solver, [images[0]], question)
    if target_res:
        end_x, end_y = target_res[0]["x"], target_res[0]["y"]
    else:
        end_x, end_y = W // 2, H // 2

    # 2. Start point (usually a handle on the left or tray)
    # Generic heuristic: search for 'icon' or 'handle' or fallback to left
    start_x, start_y = W // 6, H // 2
    
    logger.info("objectDrag (CLIP) — from:(%d,%d) to:(%d,%d)", start_x, start_y, end_x, end_y)
    return [{"start": [float(start_x), float(start_y)], "end": [float(end_x), float(end_y)]}]


# ══════════════════════════════════════════════════════════════════════════════
#  TYPE 4 — objectTag  (Multi-choice text answer)
#  Returns: string[]  — selected answer text(s)
# ══════════════════════════════════════════════════════════════════════════════
# hCaptcha objectTag shows a single image + a list of answer options.
# We pick the option most semantically similar to what we see in the image.

_COMMON_TAGS = [
    "airplane", "animal", "bag", "ball", "banana", "bear", "bed", "bench",
    "bicycle", "bird", "boat", "book", "bottle", "bowl", "bridge", "bus",
    "butterfly", "cake", "car", "cat", "chair", "clock", "computer", "cow",
    "cup", "dog", "dolphin", "door", "duck", "elephant", "fish", "flower",
    "food", "fork", "giraffe", "glasses", "guitar", "gun", "hat", "horse",
    "house", "keyboard", "knife", "lamp", "leaf", "lion", "monitor", "monkey",
    "motorcycle", "mountain", "mouse", "mushroom", "orange", "panda", "penguin",
    "phone", "piano", "pig", "plant", "rabbit", "raccoon", "ship", "shoe",
    "sofa", "spider", "spoon", "squirrel", "star", "table", "tiger", "train",
    "tree", "truck", "turtle", "umbrella", "van", "watch", "wave", "wheel",
    "window", "wolf", "zebra",
]


def _solve_objectTag(
    solver: MobileCLIPSolver,
    images: List[str],
    question: str,
) -> List[str]:
    """
    Match the image against a label set.
    If the question contains a specific object name, use that.
    Otherwise rank _COMMON_TAGS by CLIP similarity and return top match.
    """
    if not images:
        return ["unknown"]

    img = solver.decode_image_b64(_b64_strip(images[0]))
    img_feat = solver.embed_images([img]).to(torch.float32)

    # Rank all common tags
    tag_feats = _get_text_batch(solver, [f"a photo of {t}" for t in _COMMON_TAGS])
    scores = (img_feat @ tag_feats.T).squeeze(0).cpu().tolist()

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_tag = _COMMON_TAGS[best_idx]

    logger.info("objectTag — best: %r (%.1f%%)", best_tag, scores[best_idx] * 100)
    return [best_tag]


def _get_text_batch(solver: MobileCLIPSolver, texts: List[str]) -> torch.Tensor:
    """Encode a list of texts in one batch."""
    return solver.embed_texts(texts).to(torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTE
# ══════════════════════════════════════════════════════════════════════════════
@router.post("/hcaptcha")
async def solve_hcaptcha(
    payload: HCaptchaRequest,
    background_tasks: BackgroundTasks,
    solver: MobileCLIPSolver = Depends(get_solver),
    # CaptchaMaster sends "API-KEY" header (uppercase with hyphen)
    api_key: Optional[str] = Header(None, alias="API-KEY"),
    x_csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
):
    """
    CaptchaMaster-compatible hCaptcha solver endpoint.

    Payload:
      imageData    : list of base64 tile images
      question     : challenge question string
      questionType : objectClassify | objectClick | objectDrag | objectTag
      tileCount    : number of tiles

    Returns:
      { success: true, solution: <type-specific answer> }
    """
    db = get_mongodb()

    # ── Auth ──────────────────────────────────────────────────────────────────
    if not api_key:
        return {"success": False, "error": {"code": 1001, "message": "No API-KEY header"}}

    try:
        auth_data = _get_auth(api_key, db)
        if not auth_data:
            return {"success": False, "error": {"code": 1001, "message": "Invalid API Key or expired package"}}

        active_pkg    = auth_data["pkg"]
        
        # ── Credits Check ───────────────────────────────────────────────────
        credits_limit = active_pkg.get("credits", 0)
        credits_used  = active_pkg.get("creditsUsed", 0)
        if credits_used >= credits_limit:
            return {"success": False, "error": {"code": 4029, "message": "Credits exhausted"}}

        q_type        = (payload.questionType or "objectClassify").strip()
        question      = payload.question or ""
        images        = payload.imageData or []

        if not images:
            return {"success": False, "error": {"code": 4001, "message": "imageData is empty"}}

        logger.info("hCaptcha request — type: %r | question: %r | tiles: %d",
                    q_type, question, len(images))

        # ── Caching ───────────────────────────────────────────────────────────
        # Create a unique hash for this challenge (Question + Images)
        # Note: We sort image data or join them to ensure consistency
        payload_str = f"{question}:{'|'.join(images)}"
        challenge_hash = hashlib.sha256(payload_str.encode()).hexdigest()

        # Check for cached solution
        cached_solution = db.solutions.find_one({"hash": challenge_hash})
        if cached_solution:
            logger.info("hCaptcha — Cache HIT for %s", challenge_hash[:10])
            return {
                "success": True,
                "solution": cached_solution["solution"],
                "ai_processed": False,
                "from_cache": True
            }

        # ── Solve by type ─────────────────────────────────────────────────────
        if q_type == "objectClassify":
            # 3×3 grid — boolean array of length 9
            solution = _solve_objectClassify(solver, images, question)

        elif q_type == "objectClick":
            # Bounding box click — [{x, y}] -> [[{x, y}]]
            raw_solution = _solve_objectClick(solver, images, question)
            solution = [raw_solution]

        elif q_type == "objectDrag":
            # Drag and drop — [{start:[x,y], end:[x,y]}] -> [[{start:[x,y], end:[x,y]}]]
            raw_solution = _solve_objectDrag(solver, images, question)
            solution = [raw_solution]

        elif q_type == "objectTag":
            # Multi-choice text answer — [string]
            solution = _solve_objectTag(solver, images, question)

        else:
            # Unknown type — fallback to objectClassify (grid)
            logger.warning("Unknown questionType %r — falling back to objectClassify", q_type)
            solution = _solve_objectClassify(solver, images, question)

        # ── Save to Cache & Billing ───────────────────────────────────────────
        background_tasks.add_task(
            db.solutions.insert_one, 
            {
                "hash": challenge_hash,
                "solution": solution,
                "question": question,
                "imageData": images,           # Save full image data for admin panel
                "type": q_type,
                "service": "hcaptcha",
                "createdAt": datetime.utcnow()
            }
        )
        
        background_tasks.add_task(_bill_credit, active_pkg["_id"], db)

        return {
            "success": True,
            "solution": solution,
            "ai_processed": True,
            "from_cache": False
        }

    except Exception as e:
        logger.exception("hCaptcha error [type=%s]", payload.questionType)
        return {"success": False, "error": {"code": 5000, "message": str(e)}}
