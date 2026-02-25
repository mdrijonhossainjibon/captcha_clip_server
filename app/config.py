import os
from dotenv import load_dotenv

load_dotenv()

 
CLIP_MODEL: str = os.getenv("CLIP_MODEL", "ViT-L-14-336")
CLIP_PRETRAINED: str = os.getenv("CLIP_PRETRAINED", "openai")
GRID_THRESHOLD: float = float(os.getenv("GRID_THRESHOLD", "0.55"))
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://admin:Roman6432@206.189.157.8:27017/AdminHub?authSource=admin")
MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "AdminHub")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
JWT_SECRET: str = os.getenv("JWT_SECRET", "your-secret-key-change-this-in-production")
