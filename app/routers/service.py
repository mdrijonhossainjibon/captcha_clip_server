"""
/service  — main CAPTCHA solving endpoint.

Expected request body (JSON):
{
    "imageData"    : "<base64>" | ["<base64>", ...],   // single image or 9 grid chunks
    "question"     : "Select all images with a bicycle",
    "questionType" : "gridcaptcha" | "toycarcity" | ""
}

Success response:
{
    "success"  : true,
    "solution" : [1, 3, 5]   // 1-indexed cell indices (empty list = no match / skip)
}

Error response:
{
    "success" : false,
    "error"   : { "message": "..." }
}
"""

from __future__ import annotations

import logging
from typing import Any, List, Union

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator

from app.models.clip_solver import CLIPSolver
from app.dependencies import get_solver

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


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/service")
async def solve_captcha(
    payload: ServiceRequest,
    solver: CLIPSolver = Depends(get_solver),
):
    logger.info(
        f"POST /service | type='{payload.questionType}' | "
        f"images={len(payload.imageData) if isinstance(payload.imageData, list) else 1} | "
        f"question='{payload.question[:60]}'"
    )

    try:
        result = solver.solve(
            image_data=payload.imageData,
            question=payload.question,
            question_type=payload.questionType,
        )
        return {
            "success": True,
            "solution": result["solution"],
            "detected_object": result.get("detected_object"),
            # grid captcha extras
            "cell_scores":    result.get("cell_scores"),    # {cell: match %}
            "cell_contents":  result.get("cell_contents"),  # {cell: "label (conf%)"}
            "mean_confidence": result.get("mean_confidence"),
            # single-image extras
            "confidence": result.get("confidence"),
            "matched":    result.get("matched"),
        }

    except Exception as exc:
        logger.exception("Solver error")
        return {
            "success": False,
            "error": {"message": str(exc)},
        }
