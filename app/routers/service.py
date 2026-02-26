"""
/service  — main CAPTCHA solving endpoint (MobileCLIP Optimized).
"""
from __future__ import annotations

import logging
import time
import hashlib
from datetime import datetime
from typing import List, Union, Optional, Dict

from fastapi import APIRouter, Depends, Header, BackgroundTasks
from pydantic import BaseModel

from app.models.clip_solver import MobileCLIPSolver
from app.dependencies import get_solver
from app.database import get_mongodb

logger = logging.getLogger(__name__)
router = APIRouter()

# ── In-memory Auth Cache ──────────────────────────────────────────────────────
_auth_cache: Dict[str, tuple] = {}
_AUTH_TTL = 60  # seconds


def _get_auth(api_key: str, db):
    now = time.monotonic()
    if api_key in _auth_cache:
        val, ts = _auth_cache[api_key]
        if now - ts < _AUTH_TTL:
            return val
        else:
            # TTL expired — remove stale cache entry
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
    if not res:
        return None

    _auth_cache[api_key] = (res[0], now)
    return res[0]


def _bill_credit(pkg_id, db):
    """Increment creditsUsed by 1 for the given package."""
    db.packages.update_one({"_id": pkg_id}, {"$inc": {"creditsUsed": 1}})


class ServiceRequest(BaseModel):
    imageData: Union[str, List[str]]
    question: str = ""
    questionType: str = ""


@router.post("/service")
async def solve_captcha(
    payload: ServiceRequest,
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

        # ── Credits Check ───────────────────────────────────────────────────
        credits_limit = active_pkg.get("credits", 0)
        credits_used  = active_pkg.get("creditsUsed",  0)
        if credits_used >= credits_limit:
            return {"success": False, "error": {"code": 4029, "message": "Credits exhausted"}}

        # ── Solve ───────────────────────────────────────────────────────────
        result = solver.solve(payload.imageData, payload.question, payload.questionType)
        final_res = {"success": True, "solution": result["solution"]}

        # ── Log & Billing ───────────────────────────────────────────────────
        background_tasks.add_task(
            db.solutions.insert_one,
            {
                "hash": hashlib.sha256(f"{payload.question}:{payload.imageData}".encode()).hexdigest(),
                "solution": result["solution"],
                "question": payload.question,
                "imageData": payload.imageData if isinstance(payload.imageData, list) else [payload.imageData],
                "type": payload.questionType or "grid",
                "service": "service",
                "createdAt": datetime.utcnow()
            }
        )
        background_tasks.add_task(_bill_credit, active_pkg["_id"], db)

        return final_res

    except Exception as e:
        logger.exception("Solver error")
        return {"success": False, "error": {"code": 5000, "message": str(e)}}
