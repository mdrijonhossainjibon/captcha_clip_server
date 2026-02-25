"""
/service  — main CAPTCHA solving endpoint (MobileCLIP Optimized).
"""
from __future__ import annotations

import logging
import hashlib
import time
from datetime import datetime
from typing import Any, List, Union, Optional, Dict

from fastapi import APIRouter, Depends, Header, BackgroundTasks
from pydantic import BaseModel, field_validator

from app.models.clip_solver import MobileCLIPSolver
from app.dependencies import get_solver
from app.database import get_mongodb

logger = logging.getLogger(__name__)
router = APIRouter()

# ── In-memory Auth Cache ──────────────────────────────────────────────────────
_auth_cache: Dict[str, tuple] = {}
_AUTH_TTL = 60 # seconds

def _get_auth(api_key: str, db):
    now = time.monotonic()
    if api_key in _auth_cache:
        val, ts = _auth_cache[api_key]
        if now - ts < _AUTH_TTL: return val

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
    if not api_key: return {"success": False, "error": {"code": 1001, "message": "No API Key"}}

    try:
        auth_data = _get_auth(api_key, db)
        if not auth_data: return {"success": False, "error": {"code": 1001, "message": "Invalid Key/Package"}}
        
        user_id = auth_data["userId"]
        active_pkg = auth_data["pkg"]

        # Cache Hash
        hasher = hashlib.sha256()
        img_input = payload.imageData if isinstance(payload.imageData, str) else "".join(payload.imageData)
        hasher.update(img_input.encode())
        hasher.update(payload.question.encode())
        img_hash = hasher.hexdigest()

        # Solve
        result = solver.solve(payload.imageData, payload.question, payload.questionType)
        final_res = {"success": True, "solution": result["solution"]}

        # Async Billing
        background_tasks.add_task(db.packages.update_one, {"_id": active_pkg["_id"]}, {"$inc": {"creditsUsed": 1}})
        
        return final_res

    except Exception as e:
        logger.exception("Solver error")
        return {"success": False, "error": {"code": 5000, "message": str(e)}}
