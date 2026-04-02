import os
import time
import logging
import threading
import torch
from flask import Flask, request, jsonify, send_from_directory
from torchvision import transforms
from PIL import Image
from timm import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Config from env (optional overrides for production)
def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v is not None and v.isdigit() else default

def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default

# Limit upload size (default 10 MB) to prevent DoS
_max_content_mb = _env_int("MAX_CONTENT_MB", 10)
app.config["MAX_CONTENT_LENGTH"] = _max_content_mb * 1024 * 1024

# Allowed image extensions for /predict
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif"}

def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS

# Magic bytes for allowed image types (first few bytes of file)
_IMAGE_SIGNATURES = [
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"RIFF", "WEBP"),  # WebP: RIFF....WEBP
]

# Rate limit: max requests per window per IP (in-memory; per process). Env: RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SEC.
_RATE_LIMIT_REQUESTS = _env_int("RATE_LIMIT_REQUESTS", 30)
_RATE_LIMIT_WINDOW_SEC = _env_int("RATE_LIMIT_WINDOW_SEC", 60)
_RATE_LIMIT_MAX_IPS = _env_int("RATE_LIMIT_MAX_IPS", 10000)
_rate_limit_store = {}  # ip -> list of request timestamps

def _rate_limit_exceeded(ip: str) -> bool:
    now = time.time()
    cutoff = now - _RATE_LIMIT_WINDOW_SEC
    if ip not in _rate_limit_store:
        _rate_limit_store[ip] = []
    times = [t for t in _rate_limit_store[ip] if t > cutoff]
    times.append(now)
    _rate_limit_store[ip] = times
    # Prune IPs with no recent requests and cap total IPs
    to_del = [k for k, v in _rate_limit_store.items() if not v or (k != ip and max(v) < cutoff)]
    for k in to_del:
        del _rate_limit_store[k]
    while len(_rate_limit_store) > _RATE_LIMIT_MAX_IPS:
        oldest_key = min(_rate_limit_store, key=lambda k: max(_rate_limit_store[k]) if _rate_limit_store[k] else 0)
        del _rate_limit_store[oldest_key]
    return len(_rate_limit_store[ip]) > _RATE_LIMIT_REQUESTS

def _is_valid_image_file(path: str) -> bool:
    """Check file content by magic bytes; avoids accepting renamed non-image files."""
    try:
        with open(path, "rb") as f:
            header = f.read(12)
    except OSError:
        return False
    if len(header) < 6:
        return False
    for sig, _ in _IMAGE_SIGNATURES:
        if sig == b"RIFF":
            if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
                return True
        elif header.startswith(sig):
            return True
    return False

# CORS: set CORS_ORIGIN env to your front-end origin in production (e.g. https://yourdomain.com); default * for dev
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "*")

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = CORS_ORIGIN
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# Paths and Parameters
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "efficientnet_plantdoc.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Minimum confidence from disease model (env: MIN_PLANT_CONFIDENCE, default 0.50)
MIN_PLANT_CONFIDENCE = _env_float("MIN_PLANT_CONFIDENCE", 0.50)

# WHITELIST: ImageNet top‑5 must contain at least one of these to be treated as plant/leaf (otherwise we reject)
# Using top‑5 so real leaves that get top‑1 as "leaf beetle" or "cabbage butterfly" still pass
PLANT_ACCEPT_KEYWORDS = [
    "cabbage", "broccoli", "cauliflower", "zucchini", "squash", "cucumber", "artichoke",
    "pepper", "cardoon", "mushroom", "strawberry", "orange", "lemon", "fig", "pineapple",
    "banana", "jackfruit", "apple", "pomegranate", "hay", "daisy", "corn",
    "acorn", "hip", "buckeye", "fungus", "agaric", "gyromitra", "stinkhorn", "earthstar",
    "hen-of-the-woods", "bolete", "ear", "rapeseed", "lady's slipper", "granny",
    "greenhouse", "leaf", "vegetable", "fruit", "flower", "plant", "potato", "tomato",
]
# Number of ImageNet top predictions to check for plant keywords (any match = allow)
PLANT_CHECK_TOP_K = 5

# Disease model: load at startup or lazily on first /predict (set LAZY_LOAD_MODEL=1 to save RAM on 512MB)
NUM_CLASSES = 2  # Adjust based on your dataset (e.g., 2 for Diseased and Healthy)
LAZY_LOAD_MODEL = os.environ.get("LAZY_LOAD_MODEL", "").strip().lower() in ("1", "true", "yes")
_disease_model = None
_disease_model_lock = threading.Lock()
_disease_model_load_failed = False


def _load_disease_model():
    """Load disease model once; used for lazy loading or at startup."""
    global _disease_model, _disease_model_load_failed
    with _disease_model_lock:
        if _disease_model is not None:
            return _disease_model
        if _disease_model_load_failed:
            return None
        try:
            m = create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
            m = m.to(DEVICE)
            try:
                state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            except TypeError:
                state = torch.load(MODEL_PATH, map_location=DEVICE)
            m.load_state_dict(state)
            m.eval()
            _disease_model = m
            print("Model loaded successfully.")
            return _disease_model
        except FileNotFoundError:
            logger.error(f"Model file '{MODEL_PATH}' not found.")
            _disease_model_load_failed = True
            return None
        except Exception as e:
            logger.exception(f"Error loading the model: {e}")
            _disease_model_load_failed = True
            return None


if not LAZY_LOAD_MODEL:
    model = _load_disease_model()
    if model is None:
        exit(1)
else:
    model = None  # will be set on first get_model() call
    print("LAZY_LOAD_MODEL=1: disease model will load on first /predict request.")


def get_model():
    """Return the disease model; load lazily if LAZY_LOAD_MODEL is set."""
    global model
    if model is not None:
        return model
    if LAZY_LOAD_MODEL:
        loaded = _load_disease_model()
        if loaded is not None:
            model = loaded
    return model

# Plant-vs-non-plant checker: pretrained ImageNet model + reject list (optional; set DISABLE_PLANT_CHECKER=1 to save RAM on small instances)
plant_checker_model = None
imagenet_class_names = []
DISABLE_PLANT_CHECKER = os.environ.get("DISABLE_PLANT_CHECKER", "").strip().lower() in ("1", "true", "yes")

def _load_imagenet_classes():
    """Load ImageNet class names (1000 classes). Tries local file first, then URL."""
    # 1) Local file next to app.py (optional)
    local_path = os.path.join(_BASE_DIR, "imagenet_classes.txt")
    if os.path.isfile(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        except Exception as e:
            print(f"Plant checker: could not load local imagenet_classes.txt: {e}")
    # 2) Fetch from PyTorch hub
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return [line.decode("utf-8").strip() for line in resp.readlines()]
    except Exception as e:
        print(f"Plant checker: could not fetch ImageNet classes from URL: {e}")
    return []

def _init_plant_checker():
    global plant_checker_model, imagenet_class_names
    if DISABLE_PLANT_CHECKER:
        print("Plant checker disabled (DISABLE_PLANT_CHECKER=1). Using confidence-only validation.")
        return
    try:
        plant_checker_model = create_model("efficientnet_b0", pretrained=True, num_classes=1000)
        plant_checker_model = plant_checker_model.to(DEVICE)
        plant_checker_model.eval()
        imagenet_class_names = _load_imagenet_classes()
        if not imagenet_class_names:
            print("Plant checker: ImageNet class names not loaded; using confidence-only validation.")
        else:
            print("Plant checker loaded (ImageNet). Non-plant images will be rejected.")
    except Exception as e:
        print(f"Plant checker not loaded ({e}). Using confidence-only validation.")

_init_plant_checker()

def _is_likely_non_plant(image_tensor):
    """Return True if the image should be REJECTED (not a plant/leaf). Uses whitelist on top‑K: allow if any of top‑K looks like plant."""
    global plant_checker_model, imagenet_class_names
    if plant_checker_model is None:
        return False  # skip check
    with torch.no_grad():
        logits = plant_checker_model(image_tensor)
        # Get top‑K predicted class indices (K = PLANT_CHECK_TOP_K)
        k = min(PLANT_CHECK_TOP_K, logits.shape[1])
        _, top_k = torch.topk(logits, k, dim=1)
        top_indices = top_k[0].tolist()
    if not imagenet_class_names:
        return False
    for idx in top_indices:
        if idx >= len(imagenet_class_names):
            continue
        label = imagenet_class_names[idx].lower()
        for kw in PLANT_ACCEPT_KEYWORDS:
            if kw in label:
                return False  # at least one of top‑K looks like plant, allow
    # None of top‑K is in plant whitelist -> reject (document, face, bench, etc.)
    return True

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Placeholder nutrient score until a real model is added (not used for decisions)
def get_nutrient_score(_image_tensor):
    return None  # Frontend can hide this; replace with real logic when available

# User-facing message when image is not a plant/leaf
NOT_PLANT_ERROR = (
    "This doesn't look like a plant or leaf image. "
    "Please upload a clear photo of a plant leaf for disease detection."
)
LOW_CONFIDENCE_ERROR = (
    "Unable to recognize a plant leaf in this image. "
    "Please upload a clear, close-up photo of a plant leaf."
)
GENERIC_PREDICTION_ERROR = "Prediction failed. Please try again or use a different image."

# Prediction function
def predict(image_path):
    m = get_model()
    if m is None:
        return {"error": "Model not available. Please try again in a moment or contact the administrator."}
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Reject obvious non-plant images (bench, house, person, etc.) using ImageNet
        if _is_likely_non_plant(image_tensor):
            return {"error": NOT_PLANT_ERROR}

        # Test-time augmentation (TTA): original + horizontal flip, then average softmax
        # Improves accuracy and confidence, especially for diseased leaves
        with torch.no_grad():
            logits_1 = m(image_tensor)
            image_flipped = torch.flip(image_tensor, dims=[-1])  # horizontal flip
            logits_2 = m(image_flipped)
            # Average logits then softmax (more stable than averaging softmax)
            avg_logits = (logits_1 + logits_2) / 2.0
            confidence_scores = torch.softmax(avg_logits, dim=1)
            _, predicted = torch.max(confidence_scores, 1)
        
        # Decode the results (order must match how the model was trained)
        # Many PlantDoc/training scripts use 0=Diseased, 1=Healthy (e.g. folder order)
        CLASS_NAMES = ['Diseased', 'Healthy']  # index 0 = Diseased, index 1 = Healthy
        pred_idx = predicted.item()
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = confidence_scores[0, pred_idx].item()

        # Reject low-confidence predictions (unclear or non-plant images)
        if confidence < MIN_PLANT_CONFIDENCE:
            return {"error": LOW_CONFIDENCE_ERROR}
        
        # Get nutrient score
        nutrient_score = get_nutrient_score(image_tensor)
        conf_pct = round(confidence * 100, 2)

        # Human-friendly message and confidence tier for the UI
        if predicted_class == "Healthy":
            message = "No disease detected. Your leaf appears healthy."
            recommendation = "Keep monitoring; ensure good light and water."
        else:
            message = "Disease indicators detected on the leaf."
            recommendation = "For a specific diagnosis and treatment, consult a plant expert or use a clearer, close-up photo of the affected area."

        if conf_pct >= 85:
            confidence_tier = "high"
        elif conf_pct >= 60:
            confidence_tier = "moderate"
        else:
            confidence_tier = "low"

        return {
            "class": predicted_class,
            "confidence": conf_pct,
            "message": message,
            "recommendation": recommendation,
            "confidence_tier": confidence_tier,
            "nutrient_score": nutrient_score,
        }
    except Exception as e:
        logger.exception("Prediction failed")
        return {"error": GENERIC_PREDICTION_ERROR}

# Front end: React app (leaf-doctor-frontend-main). Build with: cd leaf-doctor-frontend-main && npm run build
FRONTEND_DIST = os.path.join(_BASE_DIR, "leaf-doctor-frontend-main", "dist")

def _serve_frontend(path=""):
    if path and os.path.exists(os.path.join(FRONTEND_DIST, path)):
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, "index.html")

@app.route('/')
def index():
    if os.path.isdir(FRONTEND_DIST):
        return _serve_frontend("index.html")
    # Fallback: old template if React app not built yet
    from flask import render_template
    return render_template('index.html')

@app.errorhandler(413)
def request_entity_too_large(_e):
    return jsonify({"error": f"File too large. Maximum size is {_max_content_mb} MB."}), 413

@app.route("/health")
def health():
    """Health check for load balancers and deployments. Returns 200 when app is up; model_loaded true once model is ready."""
    return jsonify({"status": "ok", "model_loaded": get_model() is not None}), 200

@app.route('/predict', methods=['POST'])
def predict_route():
    client_ip = request.remote_addr or "unknown"
    if _rate_limit_exceeded(client_ip):
        return jsonify({"error": "Too many requests. Please try again later."}), 429
    # Front end sends the file with key "image" (see leaf-doctor-frontend-main/src/lib/api.ts)
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    if file.filename == '' or not file.filename:
        return jsonify({"error": "No image selected"}), 400
    if not _allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: JPG, PNG, WEBP, GIF."}), 400
    try:
        uploads_dir = os.path.join(_BASE_DIR, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        # Save with a unique name to avoid collisions
        import tempfile
        _, ext = os.path.splitext(file.filename)
        fd, image_path = tempfile.mkstemp(suffix=ext or ".jpg", dir=uploads_dir)
        try:
            file.save(image_path)
            if not _is_valid_image_file(image_path):
                return jsonify({"error": "File is not a valid image. Allowed: JPG, PNG, GIF, WEBP."}), 400
            result = predict(image_path)
            if "error" in result:
                if "not available" in result.get("error", ""):
                    return jsonify(result), 503
                return jsonify(result), 400
            return jsonify(result)
        finally:
            try:
                os.close(fd)
                if os.path.exists(image_path):
                    os.remove(image_path)
            except OSError:
                pass
    except Exception as e:
        logger.exception("Predict route failed")
        return jsonify({"error": GENERIC_PREDICTION_ERROR}), 500

# SPA: serve React app for any other path (e.g. /assets/..., or client-side routes)
@app.route('/<path:path>')
def serve_spa(path):
    if not os.path.isdir(FRONTEND_DIST):
        return jsonify({"error": "Front end not built. Run: cd leaf-doctor-frontend-main && npm run build"}), 404
    return _serve_frontend(path)

if __name__ == '__main__':
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    if debug:
        print("Warning: Running with debug=True. Set FLASK_DEBUG=false in production.")
    try:
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        print(f"An error occurred while starting the server: {e}")
