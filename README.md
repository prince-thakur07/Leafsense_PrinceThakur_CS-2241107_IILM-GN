# Plant Disease Detection (LeafSense)

Plant disease detection app: **Flask backend** (PyTorch model) + **React front end** (LeafSense UI).

## Project layout (mapping)

| Part | Location | Role |
|------|----------|------|
| **Backend (API)** | `app.py` (project root) | Flask app: loads EfficientNet model, exposes `POST /predict`, serves built front end from `leaf-doctor-frontend-main/dist` |
| **Front end (UI)** | `leaf-doctor-frontend-main/` | React + Vite + TypeScript + shadcn/ui. Build output: `leaf-doctor-frontend-main/dist/` |
| **Old UI (deprecated)** | `templates/index.html`, `templates/result.html` | Legacy HTML forms; only used if the React app is not built |

## API contract (front end ↔ backend)

- **Endpoint:** `POST /predict`
- **Request:** `multipart/form-data` with field **`image`** (file). Max size **10 MB** (configurable via `MAX_CONTENT_MB`). Allowed types: JPG, PNG, GIF, WEBP.
- **Response (JSON):**
  - Success: `{ "class": "Healthy" | "Diseased", "confidence": number, "message", "recommendation", "confidence_tier", "nutrient_score" }`
  - Error: `{ "error": "message" }` with HTTP 4xx/5xx.
- **Health check:** `GET /health` returns `{ "status": "ok", "model_loaded": true }` with 200 when the app and model are ready (for load balancers / deployments).
- **Rate limit:** 30 requests per 60 seconds per IP (configurable via env). Exceeding returns **429** and `{ "error": "Too many requests. Please try again later." }`.

Front end source: `leaf-doctor-frontend-main/src/lib/api.ts` (`predictImage()`).

## How to run

### Option A – Single server (Flask serves UI + API)

1. Build the front end:
   ```bash
   cd leaf-doctor-frontend-main
   npm install
   npm run build
   cd ..
   ```
2. Run the backend (serves the built React app at `/` and API at `/predict`):
   ```bash
   python app.py
   ```
3. Open **http://localhost:5000**

### Option B – Dev: React dev server + Flask API

1. Start Flask (API only; old template at `/` if dist not present):
   ```bash
   python app.py
   ```
2. In another terminal, start the React dev server:
   ```bash
   cd leaf-doctor-frontend-main
   npm run dev
   ```
3. Open **http://localhost:8080** (React). The UI calls the API at `http://localhost:5000` (set in `leaf-doctor-frontend-main/src/lib/api.ts` via `VITE_API_BASE_URL` or default).

## Backend requirements

- Python 3 with: `flask`, `torch`, `torchvision`, `PIL` (Pillow), `timm` (see `requirements.txt`)
- Model file: `efficientnet_plantdoc.pth` in the project root (same folder as `app.py`)

## Dataset (for training)

LeafSense uses a **binary leaf-disease dataset** (Healthy vs Diseased) built for this project.

1. **Source images:** Obtain a set of labelled leaf images in a source folder with one subfolder per class (e.g. `Tomato___healthy`, `Tomato_early_blight`, …). Place them under `plantvillage dataset/color` (or another folder; see step 2).
2. **Build the LeafSense dataset:** From the project root, run:
   ```bash
   python prepare_leafsense_dataset.py
   ```
   This creates `leafsense_binary_dataset/Healthy/` and `leafsense_binary_dataset/Diseased/` by applying the binary labeling rule (folder name contains "healthy" → Healthy, else → Diseased). Optional: `--max-per-class 500` to cap images per class; `--source` and `--output` to change paths.
3. **Train:** Run `python train.py` (default data path is `leafsense_binary_dataset`).

The original leaf images come from a publicly available collection; our contribution is the binary labeling, subset, and split used for training.

## Environment variables (backend)

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_DEBUG` | `false` | Set to `true`/`1` for debug mode (do not use in production). |
| `CORS_ORIGIN` | `*` | Allowed CORS origin; set to your front-end URL in production (e.g. `https://yourdomain.com`). |
| `MAX_CONTENT_MB` | `10` | Max upload size in MB for `/predict`. |
| `MIN_PLANT_CONFIDENCE` | `0.5` | Minimum confidence (0–1) to accept a prediction; below this returns a low-confidence error. |
| `RATE_LIMIT_REQUESTS` | `30` | Max requests per IP per window. |
| `RATE_LIMIT_WINDOW_SEC` | `60` | Rate limit window in seconds. |
| `RATE_LIMIT_MAX_IPS` | `10000` | Max number of IPs to track (older entries evicted). |
| `DISABLE_PLANT_CHECKER` | (unset) | Set to `1` or `true` to skip loading the ImageNet plant-checker model (saves ~400 MB RAM; use on 512 MB free tier). |
| `LAZY_LOAD_MODEL` | (unset) | Set to `1` or `true` to load the disease model on first `/predict` instead of at startup (reduces startup RAM; use with 512 MB if still OOM). |

## Deploying on Render

LeafSense is a **Flask backend + React frontend** app. On Render you must use a **Web Service**, not a Static Site.

- **Service type:** **Web Service** (not Static Site). Static Site does not run a server, so `/predict` and the model would never run.
- **Root Directory:** leave default (repo root).
- **Build Command:**
  ```bash
  pip install -r requirements.txt && cd leaf-doctor-frontend-main && npm install && npm run build && cd ..
  ```
- **Start Command (required for Render to route traffic):**
  ```bash
  gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 app:app
  ```
  The server must bind to `0.0.0.0` and use the `PORT` environment variable that Render sets.
- **Publish Directory:** leave empty (Web Services ignore this).
- **Python version:** The repo includes `runtime.txt` (e.g. `python-3.10.12`). Render uses it automatically.
- **Model file:** The app expects `efficientnet_plantdoc.pth` in the project root at startup. Either commit this file to the repo (or use Git LFS), or add a build step that downloads it from your own URL. Without it, the service will fail to start.
- **Environment variables (optional):**
  - `FLASK_DEBUG=false` — recommended for production.
  - `CORS_ORIGIN=https://your-frontend-domain.com` — if the frontend is on another domain.
  - **`DISABLE_PLANT_CHECKER=1`** — **set this on Render free tier (512 MB RAM).** It skips loading the second EfficientNet (ImageNet) model used to reject non-leaf images, saving ~400 MB RAM.
  - **`LAZY_LOAD_MODEL=1`** — **set this on 512 MB if you still get Out of memory at startup.** The disease model then loads on the first `/predict` request instead of at startup, so the app can bind to the port and serve `/health` and the UI. The first prediction may be slow or may still hit OOM; if it does, use a plan with **at least 1 GB RAM** (e.g. Render Starter) for reliable inference.

You can also use **Blueprint**: connect the repo and use the included `render.yaml` so Render picks up build and start commands automatically.

The build installs Python deps (including torch) and builds the React app. At runtime, gunicorn runs Flask, which serves both the API and the built React files from `leaf-doctor-frontend-main/dist`.

## Netlify (frontend only)

Netlify is for **static sites** and **serverless functions**. It does **not** run a long-lived Flask server with PyTorch. To use Netlify:

- Deploy **only the frontend** (build `leaf-doctor-frontend-main` with `npm run build`, publish the `dist/` folder).
- Set **Environment variable** `VITE_API_BASE_URL` to your backend URL (e.g. your Render Web Service URL like `https://leafsense.onrender.com`).
- The backend must run elsewhere (e.g. Render, Railway, or Fly.io).

---

## Summary of changes made

- **Backend:** Uses uploaded file (field `image`), saves to a temp file, runs prediction, returns JSON, then deletes the temp file. CORS enabled for dev. Serves React build from `leaf-doctor-frontend-main/dist` at `/` when present; fallback to `templates/index.html` if not built.
- **Front end:** Hero section no longer depends on a missing `hero-bg.jpg` (uses CSS gradient). Production build uses relative `/predict` when served from Flask (`.env.production`).
- **Old templates:** Kept for fallback only; primary UI is the React app in `leaf-doctor-frontend-main`.
