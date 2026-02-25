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
def _solve_objectClassify(
    solver: MobileCLIPSolver,
    images: List[str],
    question: str,
) -> List[bool]:
    from PIL import Image

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
def _solve_objectClick(
    solver: MobileCLIPSolver,
    images: List[str],
    question: str,
) -> List[Dict[str, float]]:
    """
    Sliding-window CLIP search on the canvas image.
    Returns list of { x, y } pixel coords (centre of best-matching region).
    We assume canvas is ~600×600 pixels (hCaptcha default bbox canvas size).
    """
    from PIL import Image

    # Use first image (bbox canvas is usually 1 image)
    img = solver.decode_image_b64(_b64_strip(images[0]))
    W, H = img.size

    clean_q = _clean_question(question)
    pos_feat = _get_text_features(f"a photo of {clean_q}")
    neg_feat = _get_text_features("background, empty area, nothing relevant")
    txt_feats = torch.cat([pos_feat, neg_feat], dim=0).to(torch.float32)

    win_fracs  = [(0.25, 0.25), (0.35, 0.35), (0.45, 0.45)]
    stride_f   = 0.10
    best_score = -1.0
    best_cx, best_cy = W // 2, H // 2

    for wf, hf in win_fracs:
        ww = int(W * wf)
        wh = int(H * hf)
        sx = max(1, int(W * stride_f))
        sy = max(1, int(H * stride_f))
        crops, coords = [], []
        x = 0
        while x + ww <= W:
            y = 0
            while y + wh <= H:
                crops.append(img.crop((x, y, x + ww, y + wh)))
                coords.append((x, y, ww, wh))
                y += sy
            x += sx

        BATCH = 48
        for start in range(0, len(crops), BATCH):
            batch = crops[start: start + BATCH]
            feats = solver.embed_images(batch).to(torch.float32)
            p_list = (feats @ txt_feats.T).softmax(dim=-1)[:, 0].cpu().tolist()
            for prob, (bx, by, bw, bh) in zip(p_list, coords[start: start + BATCH]):
                if prob > best_score:
                    best_score = prob
                    best_cx = bx + bw // 2
                    best_cy = by + bh // 2

    logger.info("objectClick — target: %r | best: (%d, %d) conf: %.1f%%",
                clean_q, best_cx, best_cy, best_score * 100)
    return [{"x": float(best_cx), "y": float(best_cy)}]


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
    """
    hCaptcha drag-drop: one image has the entity (small object),
    another is the target scene. We find WHERE in the scene the entity fits.

    images[0] = background/scene canvas
    images[1] = entity to drag (if 2+ images provided)
    """
    from PIL import Image

    scene_img = solver.decode_image_b64(_b64_strip(images[0]))
    W, H = scene_img.size

    if len(images) >= 2:
        # Entity image available — match entity to scene via similarity
        entity_img  = solver.decode_image_b64(_b64_strip(images[1]))
        ew, eh      = entity_img.size
        entity_feat = solver.embed_images([entity_img]).to(torch.float32)

        sx = max(1, ew // 3)
        sy = max(1, eh // 3)
        crops, coords = [], []
        x = 0
        while x + ew <= W:
            y = 0
            while y + eh <= H:
                crops.append(scene_img.crop((x, y, x + ew, y + eh)))
                coords.append((x, y))
                y += sy
            x += sx

        best_score = -1.0
        best_cx, best_cy = W // 2, H // 2

        BATCH = 48
        for start in range(0, len(crops), BATCH):
            batch = crops[start: start + BATCH]
            feats = solver.embed_images(batch).to(torch.float32)
            sims  = (feats @ entity_feat.T).squeeze(-1).cpu().tolist()
            for sim, (bx, by) in zip(sims, coords[start: start + BATCH]):
                if sim > best_score:
                    best_score = sim
                    best_cx = bx + ew // 2
                    best_cy = by + eh // 2

        # start = centre of scene image (entity initial position placeholder)
        start_x, start_y = W * 3 // 4, H // 2   # right-side "entity zone" guess
        end_x,   end_y   = best_cx, best_cy
    else:
        # Only 1 image — do text-guided sliding window to find target area
        clean_q = _clean_question(question)
        pos_feat = _get_text_features(f"a photo of {clean_q}")
        neg_feat = _get_text_features("background, empty space")
        txt_feats = torch.cat([pos_feat, neg_feat], dim=0).to(torch.float32)

        win_fracs = [(0.25, 0.25), (0.35, 0.35)]
        stride_f  = 0.12
        best_score = -1.0
        best_cx, best_cy = W // 2, H // 2

        for wf, hf in win_fracs:
            ww = int(W * wf)
            wh = int(H * hf)
            sx = max(1, int(W * stride_f))
            sy = max(1, int(H * stride_f))
            crops, coords = [], []
            x = 0
            while x + ww <= W:
                y = 0
                while y + wh <= H:
                    crops.append(scene_img.crop((x, y, x + ww, y + wh)))
                    coords.append((x, y, ww, wh))
                    y += sy
                x += sx

            BATCH = 48
            for start_i in range(0, len(crops), BATCH):
                batch = crops[start_i: start_i + BATCH]
                feats = solver.embed_images(batch).to(torch.float32)
                p_list = (feats @ txt_feats.T).softmax(dim=-1)[:, 0].cpu().tolist()
                for prob, (bx, by, bw, bh) in zip(p_list, coords[start_i: start_i + BATCH]):
                    if prob > best_score:
                        best_score = prob
                        best_cx = bx + bw // 2
                        best_cy = by + bh // 2

        start_x, start_y = W * 3 // 4, H // 2
        end_x, end_y = best_cx, best_cy

    logger.info("objectDrag — start: (%d,%d) → end: (%d,%d)", start_x, start_y, end_x, end_y)
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
        q_type        = (payload.questionType or "objectClassify").strip()
        question      = payload.question or ""
        images        = payload.imageData or []

        if not images:
            return {"success": False, "error": {"code": 4001, "message": "imageData is empty"}}

        logger.info("hCaptcha request — type: %r | question: %r | tiles: %d",
                    q_type, question, len(images))

        # ── Solve by type ─────────────────────────────────────────────────────
        if q_type == "objectClassify":
            # 3×3 grid — boolean array of length 9
            solution = _solve_objectClassify(solver, images, question)

        elif q_type == "objectClick":
            # Bounding box click — [{x, y}]
            solution = _solve_objectClick(solver, images, question)

        elif q_type == "objectDrag":
            # Drag and drop — [{start:[x,y], end:[x,y]}]
            solution = _solve_objectDrag(solver, images, question)

        elif q_type == "objectTag":
            # Multi-choice text answer — [string]
            solution = _solve_objectTag(solver, images, question)

        else:
            # Unknown type — fallback to objectClassify (grid)
            logger.warning("Unknown questionType %r — falling back to objectClassify", q_type)
            solution = _solve_objectClassify(solver, images, question)

        # ── Billing ───────────────────────────────────────────────────────────
        background_tasks.add_task(
            db.packages.update_one,
            {"_id": active_pkg["_id"]},
            {"$inc": {"creditsUsed": 1}}
        )

        return {"success": True, "solution": solution}

    except Exception as e:
        logger.exception("hCaptcha error [type=%s]", payload.questionType)
        return {"success": False, "error": {"code": 5000, "message": str(e)}}
