"""
/service  — main CAPTCHA solving endpoint with MongoDB and Billing support (Numeric Error Codes).
"""
from __future__ import annotations

import logging
import hashlib
from datetime import datetime
from typing import Any, List, Union, Optional, Dict

from fastapi import APIRouter, Depends, Header, HTTPException, BackgroundTasks
from pydantic import BaseModel, field_validator

from app.models.clip_solver import CLIPSolver
from app.dependencies import get_solver
from app.database import get_mongodb

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request schema ────────────────────────────────────────────────────────────

class ServiceRequest(BaseModel):
    imageData: Union[str, List[str]]
    question: str = ""
    questionType: str = ""

    @field_validator("imageData")
    @classmethod
    def validate_image_data(cls, v: Any) -> Any:
        if isinstance(v, str) and not v:
            raise ValueError("imageData must not be empty")
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("imageData list must not be empty")
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_to_cache(imageHash: str, imageData: Union[str, List[str]], question: str, response: Dict, userId: Any):
    """Saves solving results to DataCaches collection. Only saves if there's a solution."""
    if not response.get("solution"):
        return
        
    try:
        db = get_mongodb()
        # Representative image data for storage
        img_rep = imageData if isinstance(imageData, str) else imageData[0]
        
        save_data = {
            "imageData": img_rep,
            "question": question,
            "answer": response.get("solution", []),
            "rawResponse": response,
            "userId": userId,
            "createdAt": datetime.utcnow()
        }
        db.DataCaches.update_one({"imageHash": imageHash}, {"$set": save_data}, upsert=True)
    except Exception as e:
        logger.error(f"DataCaches save failed: {e}")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/service")
async def solve_captcha(
    payload: ServiceRequest,
    background_tasks: BackgroundTasks,
    solver: CLIPSolver = Depends(get_solver),
    api_key: Optional[str] = Header(None, alias="api-key")
):
    db = get_mongodb()
    
    # 1. API Key Validation (Error Code: 1001)
    if not api_key:
        return {"success": False, "error": {"code": 1001, "message": "Missing api-key in headers"}}
    
    try:
        api_key_doc = db.apikeys.find_one({"key": api_key, "status": "active"})
        if not api_key_doc:
            return {"success": False, "error": {"code": 1001, "message": "Invalid or inactive API key"}}
        
        user_id = api_key_doc["userId"]
        
        # 2. User & Status Check (Error Code: 1002)
        user_doc = db.users.find_one({"_id": user_id})
        if not user_doc:
            return {"success": False, "error": {"code": 1002, "message": "User not found"}}
            
        user_status = user_doc.get("status")
        if not user_status:
            db.users.update_one({"_id": user_id}, {"$set": {"status": "active"}})
            user_status = "active"

        if user_status == "suespend":
            return {"success": False, "error": {"code": 1002, "message": "User account is suspended"}}
        
        if user_status != "active":
            return {"success": False, "error": {"code": 1002, "message": "User account is inactive"}}
            
        # 3. Package & Credit Validation (Error Code: 1003)
        now = datetime.utcnow()
        active_package = db.packages.find_one({
            "userId": user_id, 
            "status": "active",
            "endDate": {"$gt": now}
        })
        
        if not active_package:
            return {"success": False, "error": {"code": 1003, "message": "Package expired or not found"}}
            
        if active_package.get("creditsUsed", 0) >= active_package.get("credits", 0):
            return {"success": False, "error": {"code": 1003, "message": "Insufficient credits"}}

        # 4. Cache Lookup
        # Create hash based on imageData and question to ensure unique caching
        hasher = hashlib.sha256()
        if isinstance(payload.imageData, list):
            for part in payload.imageData: hasher.update(part.encode())
        else:
            hasher.update(payload.imageData.encode())
        hasher.update(payload.question.encode())
        img_hash = hasher.hexdigest()
        
        cached_entry = db.DataCaches.find_one({"imageHash": img_hash})
        if cached_entry:
            # Deduct credit for cache hit
            db.packages.update_one({"_id": active_package["_id"]}, {"$inc": {"creditsUsed": 1}})
            db.apikeys.update_one({"_id": api_key_doc["_id"]}, {"$set": {"lastUsedAt": now}})
            return cached_entry["rawResponse"]

        # 5. Core Solve
        result = solver.solve(
            image_data=payload.imageData,
            question=payload.question,
            question_type=payload.questionType,
        )
        
        final_response = {
            "success": True,
            "solution": result["solution"]
        }

        # 6. Billing & Billing Metadata
        db.packages.update_one({"_id": active_package["_id"]}, {"$inc": {"creditsUsed": 1}})
        db.apikeys.update_one({"_id": api_key_doc["_id"]}, {"$set": {"lastUsedAt": now}})
        
        # Save to DB asynchronously
        background_tasks.add_task(save_to_cache, img_hash, payload.imageData, payload.question, final_response, user_id)

        return final_response

    except Exception as exc:
        logger.exception("Solver error")
        return {
            "success": False,
            "error": {"code": 5000, "message": str(exc)},
        }
