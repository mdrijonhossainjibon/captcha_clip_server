import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN: str = os.getenv("API_TOKEN", "")
CLIP_MODEL: str = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED: str = os.getenv("CLIP_PRETRAINED", "openai")
GRID_THRESHOLD: float = float(os.getenv("GRID_THRESHOLD", "0.55"))
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
