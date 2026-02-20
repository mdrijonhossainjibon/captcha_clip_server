# Captcha CLIP Solver — FastAPI Server

A local Python server that uses **OpenAI CLIP** to automatically solve AWS WAF grid and Toycarcity CAPTCHAs sent by the CaptchaMaster Chrome extension.

---

## 📁 Structure

```
captcha_clip_server/
├── main.py                  ← FastAPI app entry point
├── requirements.txt
├── .env.example             ← Copy to .env and configure
├── start.bat                ← One-click Windows launcher
└── app/
    ├── config.py            ← Reads .env settings
    ├── models/
    │   └── clip_solver.py   ← CLIP model + solving logic
    └── routers/
        └── service.py       ← POST /service endpoint
```

---

## 🚀 Quick Start

### 1. Install & Run (Windows)
Double-click **`start.bat`** — it will:
- Create a Python virtual environment
- Install all dependencies
- Copy `.env.example` → `.env`
- Start the server on `http://localhost:8000`

### 2. Configure `.env`
```env
API_TOKEN=your_secret_token_here   # Must match extension's stored token
CLIP_MODEL=ViT-B-32                # or ViT-L-14 for higher accuracy
CLIP_PRETRAINED=openai
GRID_THRESHOLD=0.55                # Lower = more cells selected
PORT=8000
```

### 3. Manual Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## 🔌 API

### `POST /service`

**Headers:**
```
X-CSRF-Token: <your API_TOKEN>
Content-Type: application/json
```

**Body:**
```json
{
  "imageData"    : ["<base64>", "<base64>", ...],  // 9 chunks for grid, 1 for single
  "question"     : "Select all images with a bicycle",
  "questionType" : "gridcaptcha"  // or "toycarcity"
}
```

**Success Response:**
```json
{ "success": true, "solution": [1, 3, 7] }
```
> `solution` is a **1-indexed** list of matching cell positions.
> Empty array `[]` means no match found → extension will skip/reload.

**Error Response:**
```json
{ "success": false, "error": { "message": "..." } }
```

### `GET /health`
Returns `{ "status": "ok" }`.

### `GET /docs`
Interactive Swagger UI.

---

## 🧠 How CLIP Solving Works

1. **Question parsing** — extracts the object label from the question text  
   e.g. `"Select all images with a bicycle"` → `"a bicycle"`

2. **Text embeddings** — encodes `"a photo of a bicycle"` vs `"a photo of something else"`

3. **Image embeddings** — encodes each base64 image chunk

4. **Cosine similarity + softmax** — scores each image against both text labels

5. **Threshold filter** — returns 1-indexed cells where positive probability ≥ `GRID_THRESHOLD`

---

## ⚙️ Model Options

| Model | Accuracy | Speed | VRAM |
|---|---|---|---|
| `ViT-B-32` | Good | Fast | ~1 GB |
| `ViT-B-16` | Better | Medium | ~1.5 GB |
| `ViT-L-14` | Best | Slow | ~4 GB |

Set `CLIP_MODEL` and `CLIP_PRETRAINED=openai` in `.env`.

---

## 🔗 Extension Integration

The Chrome extension (`background.ts`) sends `AWS_WAF_DETECTED` messages to this server via the `/service` endpoint. Make sure:
- `API_TOKEN` in `.env` matches the token stored in `chrome.storage.local` under key `token`
- `API_BASE_URL` in `lib/api.ts` points to `http://localhost:8000` (or your deployed URL)
