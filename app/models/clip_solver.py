"""
CLIP-based captcha solver.

Handles three question types sent by the Chrome extension:
  - gridcaptcha  : 9 base64 image chunks; returns 1-indexed list of matching cells
  - toycarcity   : 1 base64 image; returns [1] if it matches the object in question, [] otherwise
  - (default)    : 1 base64 image; same as toycarcity logic
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import List

import open_clip
import torch
from PIL import Image

from app.config import CLIP_MODEL, CLIP_PRETRAINED, GRID_THRESHOLD

logger = logging.getLogger(__name__)

# ── Singleton model loader ────────────────────────────────────────────────────

_model = None
_preprocess = None
_tokenizer = None
_device = None


def _load_model():
    global _model, _preprocess, _tokenizer, _device

    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading CLIP model '{CLIP_MODEL}' (pretrained={CLIP_PRETRAINED}) on {_device} …")

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    _model = _model.to(_device)
    _model.eval()

    _tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    logger.info("CLIP model loaded ✓")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_image(b64: str) -> Image.Image:
    """Decode a raw base64 string (no data-URL prefix) into a PIL image."""
    # Strip data-URL prefix if still present
    b64 = re.sub(r"^data:image/[^;]+;base64,", "", b64)
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _embed_images(images: List[Image.Image]) -> torch.Tensor:
    """Return normalised image feature tensor of shape (N, D)."""
    tensors = torch.stack([_preprocess(img) for img in images]).to(_device)  # type: ignore[arg-type]
    with torch.no_grad():
        feats = _model.encode_image(tensors)  # type: ignore[union-attr]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def _embed_texts(texts: List[str]) -> torch.Tensor:
    """Return normalised text feature tensor of shape (N, D)."""
    tokens = _tokenizer(texts).to(_device)  # type: ignore[operator]
    with torch.no_grad():
        feats = _model.encode_text(tokens)  # type: ignore[union-attr]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def _extract_object(question: str) -> str:
    """
    Pull the subject out of a typical CAPTCHA question.

    Examples
    --------
    "Select all images with a bicycle"  →  "a bicycle"
    "Click on all toy cars"             →  "toy cars"
    "Find the traffic lights"           →  "the traffic lights"
    """
    question = question.strip().rstrip("?.")
    patterns = [
        r"(?:select|click(?: on)?|find|identify)\s+all\s+(?:images?\s+(?:with|of|showing|containing)\s+)?(.+)",
        r"(?:select|click(?: on)?|find|identify)\s+(?:the\s+)?(.+)",
        r"(?:images?\s+(?:with|of|showing|containing)\s+)(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return question  # fallback: use entire question




# Irregular plurals known to appear in CAPTCHAs
_IRREGULAR_SINGULAR = {
    "bicycles": "bicycle", "motorcycles": "motorcycle", "vehicles": "vehicle",
    "buses": "bus", "chimneys": "chimney", "bridges": "bridge",
    "benches": "bench", "couches": "couch", "coaches": "coach",
    "boxes": "box", "watches": "watch", "churches": "church",
    "traffic lights": "traffic light", "fire hydrants": "fire hydrant",
    "crosswalks": "crosswalk", "staircases": "staircase",
    "mountains": "mountain", "rivers": "river", "trees": "tree",
    "cars": "car", "trucks": "truck", "boats": "boat",
    "planes": "plane", "trains": "train", "taxis": "taxi",
    "signs": "sign", "lights": "light", "doors": "door",
    "windows": "window", "chairs": "chair", "tables": "table",
    "beds": "bed", "sofas": "sofa", "lamps": "lamp",
    "bags": "bag", "hats": "hat", "clocks": "clock",
    "buckets": "bucket", "curtains": "curtain",
    "cats": "cat", "dogs": "dog", "birds": "bird", "horses": "horse",
    "elephants": "elephant", "zebras": "zebra", "giraffes": "giraffe",
    "people": "person", "persons": "person", "humans": "human",
    "crossroads": "crossroad", "pavements": "pavement",
    "storefronts": "storefront", "stores": "store", "shops": "shop",
    "mattresses": "mattress", "blankets": "blanket", "pillows": "pillow",
    "drapes": "drape",
    "purses": "purse", "satchels": "satchel", "briefcases": "briefcase",
    "cases": "case", "suitcases": "suitcase", "luggage": "luggage",
    "backpacks": "backpack", "rucksacks": "rucksack", "handbags": "handbag",
    "totes": "totebag", "plastic bags": "plastic bag", "paper bags": "paper bag",
    "armchairs": "armchair", "stools": "stool", "recliners": "recliner",
    "wheelchairs": "wheelchair", "office chairs": "office chair",
    "caps": "cap", "helmets": "helmet", "fedoras": "fedora", "beanies": "beanie",
    "clutches": "clutch", "duffels": "duffel bag",
    "bedrooms": "bedroom",
}


def _singularize(word: str) -> str:
    """Best-effort plural → singular for CAPTCHA objects."""
    low = word.lower().strip()
    if low in _IRREGULAR_SINGULAR:
        return _IRREGULAR_SINGULAR[low]
    # Generic English rules (covers most regular CAPTCHA nouns)
    if low.endswith("ies") and len(low) > 4:
        return low[:-3] + "y"          # butterflies → butterfly
    if low.endswith("ves") and len(low) > 4:
        return low[:-3] + "f"          # knives → knife
    if low.endswith("ses") or low.endswith("xes") or low.endswith("zes"):
        return low[:-2]                 # buses → bus, boxes → box
    if low.endswith("s") and not low.endswith("ss") and len(low) > 3:
        return low[:-1]                 # cars → car, beds → bed
    return low


# Semantic groups to prevent misses (e.g. "watch" should count for "clock")
_RELATED_TERMS = {
    "clock":  {"clock", "watch"},
    "watch":  {"clock", "watch"},
    "bag":    {"bag", "handbag", "backpack", "suitcase", "purse", "briefcase", "satchel", "luggage", "totebag", "case", "rucksack", "plastic bag", "paper bag", "clutch", "duffel bag"},
    "hat":    {"hat", "cap", "helmet", "fedora", "beanie"},
    "bed":    {"bed", "couch", "sofa", "mattress", "blanket", "pillow", "bedroom", "furniture"},
    "chair":  {"chair", "seat", "bench", "armchair", "stool", "recliner", "office chair", "wheelchair"},
    "bus":    {"bus", "truck", "vehicle"},
    "truck":  {"truck", "bus", "vehicle", "car"},
    "car":    {"car", "taxi", "vehicle", "toy car"},
    "boat":   {"boat", "ship"},
    "bicycle": {"bicycle", "motorcycle"},
    "motorcycle": {"motorcycle", "bicycle"},
    "curtain": {"curtain", "window", "blind", "door", "drape"},
    "window":  {"window", "curtain"},
    "bridge":  {"bridge", "road"},
    "lamp":    {"lamp", "light"},
}


def _build_prompts(obj: str):
    """
    Return (pos_texts, neg_texts) — rich multi-prompt lists for CLIP.
    Works on both singular and plural; always generates singular variants.
    """
    # Strip leading articles so singularize works cleanly
    clean = re.sub(r"^(a |an |the )", "", obj, flags=re.IGNORECASE).strip()
    singular = _singularize(clean)
    # Use both forms so CLIP sees plural context too
    forms = list(dict.fromkeys([singular, clean]))  # dedup, singular first

    pos_texts = []
    neg_texts = []
    
    for f in forms:
        pos_texts += [
            f"a photo of a {f}",
            f"an image containing a {f}",
            f"a {f}",
            f"a picture of {f}",
        ]
        
        chunk_neg = [
            f"a photo with no {f}",
            f"an image without a {f}",
            f"something other than a {f}",
        ]

        # Specific negatives to fix common confusions
        if singular in ("bed", "beds"):
            chunk_neg += ["a chair", "an armchair", "office chair", "wooden chair"]
        elif singular in ("chair", "chairs"):
            chunk_neg += ["a mattress", "a bed", "sleeping bed"]
        elif singular in ("bus", "buses"):
            chunk_neg += ["a train", "a tram"]
        elif singular in ("train", "trains"):
            chunk_neg += ["a bus", "a truck"]
        elif singular in ("hat", "hats"):
            chunk_neg += ["a bag", "a backpack", "a purse", "a handbag", "duffel bag", "totebag"]
        elif singular in ("bag", "bags"):
            chunk_neg += ["a hat", "a cap", "a helmet"]

        neg_texts += chunk_neg

    return pos_texts, neg_texts


# Pre-computed label embeddings (cached after first call)
_label_feats: torch.Tensor | None = None
_label_texts: list[str] = []


def _get_label_feats() -> tuple[torch.Tensor, list[str]]:
    """Return (label_feats, label_texts) — computed once and cached.
    Labels are derived from the unique singular values in _IRREGULAR_SINGULAR.
    """
    global _label_feats, _label_texts
    if _label_feats is not None:
        return _label_feats, _label_texts
    # Unique singular forms — preserve insertion order, deduplicate
    labels = list(dict.fromkeys(_IRREGULAR_SINGULAR.values()))
    prompts = [f"a photo of a {lb}" for lb in labels]
    feats = _embed_texts(prompts)                        # (L, D)
    _label_feats = feats
    _label_texts = labels
    logger.info(f"Label embeddings cached for {len(labels)} classes: {labels}")
    return _label_feats, _label_texts


def _identify_cells(img_feats: torch.Tensor) -> list[dict]:
    """
    For each image in img_feats (N, D), find the top-1 CAPTCHA label.
    Returns a list of {label, confidence} dicts, one per cell.
    """
    label_feats, label_texts = _get_label_feats()
    sims = (img_feats @ label_feats.T).float()           # (N, L)
    probs = sims.softmax(dim=-1)                         # (N, L)
    top_idx = probs.argmax(dim=-1).cpu().tolist()
    top_conf = probs.max(dim=-1).values.cpu().tolist()
    return [
        {"label": label_texts[i], "confidence": round(c * 100, 1)}
        for i, c in zip(top_idx, top_conf)
    ]


# ── Public API ────────────────────────────────────────────────────────────────

class CLIPSolver:
    """Stateless solver; the model is loaded once at startup."""

    def __init__(self):
        _load_model()

    # ------------------------------------------------------------------
    # Grid captcha  (9 chunks → return dict with solution + details)
    # ------------------------------------------------------------------
    def solve_grid(self, image_chunks: List[str], question: str) -> dict:
        obj = _extract_object(question)
        pos_texts, neg_texts = _build_prompts(obj)

        # Extract singular form for logging/response
        clean = re.sub(r"^(a |an |the )", "", obj, flags=re.IGNORECASE).strip()
        singular = _singularize(clean)

        images = [_decode_image(chunk) for chunk in image_chunks]
        img_feats = _embed_images(images)                          # (N, D)

        # Average positive and negative embeddings separately
        pos_feats = _embed_texts(pos_texts).mean(dim=0, keepdim=True)  # (1, D)
        neg_feats = _embed_texts(neg_texts).mean(dim=0, keepdim=True)  # (1, D)

        # Normalise averaged embeddings
        pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)
        neg_feats = neg_feats / neg_feats.norm(dim=-1, keepdim=True)

        txt_feats = torch.cat([pos_feats, neg_feats], dim=0)       # (2, D)

        # Cosine similarity matrix (N, 2)
        sims = (img_feats @ txt_feats.T).float()

        # Softmax over the two classes
        probs = sims.softmax(dim=-1)                               # (N, 2)
        pos_probs = probs[:, 0].cpu().tolist()                     # probability of matching

        # --- Robust Selection Logic --------------------------------------
        # 1. Gather all per-cell labels first
        cell_labels = _identify_cells(img_feats)   # [{label, confidence}, ...]

        candidates = []
        for i, p in enumerate(pos_probs):
            cell_idx = i + 1
            lbl = cell_labels[i]["label"]
            
            # Condition A: Strong binary match
            cond_a = p >= 0.53

            # Condition B: Weak binary match BUT correct label identified
            # (e.g. score 0.35 but identified as 'watch'/clock when asking for 'clock')
            allowed_labels = _RELATED_TERMS.get(singular, {singular})
            # Also check if the detected label is functionally same (e.g. singular 'bed' == label 'bed')
            is_label_match = (lbl == singular) or (lbl in allowed_labels)
            
            cond_b = (p >= 0.32) and is_label_match

            if cond_a or cond_b:
                candidates.append(cell_idx)

        # Fallback: If nothing selected (or very few), use relative ranking
        if not candidates:
            # Smart relative ranking: find natural score gap → cutoff
            import statistics
            mean_p = statistics.mean(pos_probs)
            indexed = sorted(enumerate(pos_probs), key=lambda x: x[1], reverse=True)
            sorted_scores = [p for _, p in indexed]
            
            # Find the biggest DROP between consecutive scores
            if len(sorted_scores) > 1:
                gaps = [sorted_scores[i] - sorted_scores[i + 1] for i in range(len(sorted_scores) - 1)]
                max_gap_idx = gaps.index(max(gaps))
                n_pick = max(1, min(max_gap_idx + 1, 6))
            else:
                n_pick = 1

            # Safety: if top score < 0.42, force top-3
            if sorted_scores[0] < 0.42:
                n_pick = 3

            top_cells = {orig_i + 1 for orig_i, _ in indexed[:n_pick]}
            candidates = sorted(top_cells)

        solution = sorted(list(set(candidates)))

        # Per-cell match confidence  {cell (1-based): match %}
        cell_scores = {str(i + 1): round(p * 100, 1) for i, p in enumerate(pos_probs)}

        # Per-cell top-label details
        cell_contents = {
            str(i + 1): f"{c['label']} ({c['confidence']}%)"
            for i, c in enumerate(cell_labels)
        }
        
        # Calculate mean for logging/response
        import statistics
        mean_p = statistics.mean(pos_probs)

      
        logger.info(f"Grid | obj='{singular}' | mean={mean_p:.2f} | solution={solution}")

        return {
            "solution": solution,
            "detected_object": singular,
            "cell_scores": cell_scores,
            "cell_contents": cell_contents,
            "mean_confidence": round(mean_p * 100, 1),
        }

    # ------------------------------------------------------------------
    # Single-image captcha (toycarcity / default)
    # ------------------------------------------------------------------
    def solve_single(self, image_b64: str, question: str) -> dict:
        """
        Returns dict with solution=[1] if image matches, solution=[] if not.
        The upstream caller (background.ts) uses solution list only.
        """
        obj = _extract_object(question) if question else "the object"
        pos_texts, neg_texts = _build_prompts(obj)

        clean = re.sub(r"^(a |an |the )", "", obj, flags=re.IGNORECASE).strip()
        singular = _singularize(clean)

        image = _decode_image(image_b64)
        img_feats = _embed_images([image])                         # (1, D)

        pos_feats = _embed_texts(pos_texts).mean(dim=0, keepdim=True)
        neg_feats = _embed_texts(neg_texts).mean(dim=0, keepdim=True)
        pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)
        neg_feats = neg_feats / neg_feats.norm(dim=-1, keepdim=True)
        txt_feats = torch.cat([pos_feats, neg_feats], dim=0)       # (2, D)

        sims = (img_feats @ txt_feats.T).float()
        probs = sims.softmax(dim=-1)
        pos_prob = probs[0, 0].item()

        matched = pos_prob >= GRID_THRESHOLD
 
        logger.info(f"Single | obj='{singular}' | confidence={pos_prob:.1%} | match={matched}")

        return {
            "solution": [1] if matched else [],
            "detected_object": singular,
            "confidence": round(pos_prob * 100, 1),
            "matched": matched,
        }

    # ------------------------------------------------------------------
    # Top-level dispatcher
    # ------------------------------------------------------------------
    def solve(
        self,
        image_data: str | List[str],
        question: str,
        question_type: str,
    ) -> dict:
        images = image_data if isinstance(image_data, list) else [image_data]

        if question_type == "gridcaptcha":
            return self.solve_grid(images, question)


        # toycarcity or any single-image type
        return self.solve_single(images[0], question)

    # ------------------------------------------------------------------
    # Public methods for external usage (e.g. classifier router)
    # ------------------------------------------------------------------
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        return _embed_texts(texts)

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        return _embed_images(images)
    
    def decode_image_b64(self, b64: str) -> Image.Image:
        return _decode_image(b64)
