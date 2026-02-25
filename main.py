"""
Captcha CLIP Solver – FastAPI entry point
"""

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import HOST, PORT
from app.routers.service import router as service_router
from app.routers.classify import router as classify_router
from app.routers.hcaptcha import router as hcaptcha_router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Captcha CLIP Solver",
    description="Solves AWS WAF grid / toycarcity CAPTCHAs using OpenAI CLIP",
    version="1.0.0",
)

# Allow requests from the Chrome extension (runtime URL is chrome-extension://...)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten to your extension ID in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(service_router)
app.include_router(classify_router)
app.include_router(hcaptcha_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Pre-load MobileCLIP at startup so first request is fast ───────────────
@app.on_event("startup")
async def preload_model():
    logger.info("Pre-loading MobileCLIP model …")
    from app.models.clip_solver import _load_model
    _load_model()
    logger.info("MobileCLIP ready ✓ — server is ready for requests")


# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True, log_level="info")
