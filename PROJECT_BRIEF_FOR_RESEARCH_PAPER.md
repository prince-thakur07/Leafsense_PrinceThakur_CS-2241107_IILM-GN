# LeafSense — Full Project Brief for Research Paper

Use this document as the single source of truth when writing a research paper (e.g. IEEE conference) about the LeafSense project. It describes the system in detail so the paper can accurately reflect the implementation.

---

## Paper Abstract (for reference)

Plant diseases often first appear on leaves, making early detection important for preventing crop loss and controlling disease spread. Traditional diagnosis typically requires manual inspection by trained specialists or laboratory analysis, which may not always be accessible in remote agricultural regions. This paper presents **LeafSense**, a machine learning-powered web system designed for automated plant leaf disease detection from a single image. The system performs binary classification, identifying whether a leaf is **Healthy** or **Diseased**.

The proposed system employs an EfficientNet-B0 convolutional neural network trained on the **LeafSense binary dataset**, where labels are derived from dataset folder names containing the keyword "healthy". During inference, test-time augmentation is applied by performing horizontal flipping and averaging the model outputs to obtain more stable prediction confidence. To improve reliability in real-world usage, a secondary EfficientNet-B0 model pretrained on ImageNet is used to filter non-plant images before disease classification. Images that do not correspond to plant-related classes within the top-five predictions are rejected, preventing incorrect analysis of irrelevant inputs.

LeafSense is implemented as a deployable web application consisting of a Flask-based backend using PyTorch for model inference and a React-based frontend for user interaction. The backend exposes an API endpoint for image submission, performs input validation and rate limiting, and returns classification results along with prediction confidence. The entire system operates as a unified web service, enabling users to upload a leaf image and receive an automated health assessment in real time.

                                                                                                                                                                                                                                                                                                              - - -

## 1. Introduction

Every single year, plant infections wipe out crops, leading to bad harvests and huge income losses for farmers. You generally notice the problem on the leaves first through weird spots or sudden fading. If caught early, chemical treatments can stop the sickness dead in its tracks. However, getting a confirmed lab test or dragging an expert to the farm is expensive and slow. Without fast tools, many rural growers just end up guessing by eye. This usually causes mistakes since severe weather or poor soil can mimic a fungal infection perfectly. There is basically no straightforward way for farmers to get a reliable preliminary assessment without escalating the issue to a specialist.

Handing this early check-up step over to software is very doable today thanks to Convolutional Neural Networks (CNNs). Because researchers have access to massive folders of hand-labelled leaf photos, they can train smart models that recognise physical rot on a screen. These models look fantastic in a quiet lab, but pushing them onto the public internet creates massive headaches. Most researchers ignore what happens when real users upload totally random things. If you do not verify the incoming picture first, your disease scanner will happily look at a picture of a brick wall and instantly classify it as 'Healthy' or 'Diseased'. Pushing out nonsense answers ruins user trust entirely.

To solve this, we propose LeafSense, an end-to-end web system that provides real-time leaf health assessments. We built it around a strict binary classification—grouping inputs simply as Healthy or Diseased. We chose this binary setup because a farmer's first question is simply whether the plant is okay or not. The disease engine uses an EfficientNet-B0 CNN, trained on our curated dataset. We built this data split by automatically grouping any source image with the word "healthy" in its folder name into one pile, and throwing every other disease condition into the second pile. But before the actual app checks for disease, we run the upload through a strict non-plant gatekeeper. A secondary ImageNet-pretrained model looks at the image first. We check its top five predictions against a specific word whitelist. We look at five guesses instead of one because standard networks sometimes bizarrely classify a real leaf photograph as a 'leaf beetle' or 'cabbage butterfly'. If there are zero plant words in the top five, the system rejects the user's photo immediately.

When a valid leaf finally reaches the disease model, we use a trick known as Test-Time Augmentation (TTA). The system pushes the photo through the network, digitally flips it sideways to run it a second time, and then merges both scores to get a steady final guess. This stops things like weird shadows or oddly tilted phone cameras from throwing off the prediction. The entire setup is deployed as a cohesive web app built on Flask and React. When an image hits the API, the server quickly runs rate-limiting and file size checks before returning the JSON result. Because the interface and the backend share one single URL, anyone with a smartphone can get instant preliminary assessments without installing separate apps.

Therefore, the primary contributions of this paper are:
(i) developing a binary foliage disease detector running on EfficientNet-B0;
(ii) implementing a non-plant gatekeeper to ensure the system only runs on valid leaf images;
(iii) deploying these models within a lightweight, real-time web platform; and
(iv) applying test-time augmentation to keep final prediction scores from bouncing around.

The rest of this paper discusses older research, breaks down how we built the system, displays our testing scores, and outlines future work.

---

## 1. Project Overview

- **Name:** LeafSense  
- **Purpose:** Web-based system for **binary leaf disease detection**: given a single leaf image, the system classifies it as **Healthy** or **Diseased**.  
- **Scope:** The system analyzes **leaf images only**. Non-leaf images (e.g. documents, objects, faces) are rejected before classification.  
- **Repository:** https://github.com/abhishekhbihari007/leafSense  
- **Stack:** Flask (Python) backend + PyTorch/timm for ML; React (Vite, TypeScript, shadcn/ui) front end. The backend can serve the built React app so the whole system runs as **one web service** behind a single URL.

---

## 2. Problem and Motivation

- Plant diseases often appear on leaves first. Detecting affected leaves early helps limit spread and target treatment.
- Traditional diagnosis relies on specialists in the field or lab analysis of samples—both time-consuming and not always available in remote areas.
- An alternative is to let users photograph a leaf and get an automated assessment. Such a tool can support farmers, extension workers, and others who need a quick check without waiting for a lab or expert.
- Many research efforts focus on accuracy on fixed benchmarks; fewer address the full path from training to a usable service: handling arbitrary uploads, rejecting non-leaf images, and exposing the model through a simple web interface. LeafSense targets that gap.

---

## 3. Dataset

- **LeafSense binary dataset:** We use a custom binary leaf-disease dataset prepared for this project. It is built by the script `prepare_leafsense_dataset.py`, which reads from a **source folder** of labelled leaf images (one subfolder per original class) and writes two folders: **Healthy** and **Diseased**. The script applies our labeling rule: any source folder whose name contains **"healthy"** (case-insensitive) contributes to Healthy; all others to Diseased. The result is stored as `leafsense_binary_dataset/Healthy/` and `leafsense_binary_dataset/Diseased/` in the project. We do not claim to have created the original leaf images; they come from a publicly available collection of labelled leaf images. Our contribution is the binary labeling, the subset choice (color RGB images), and the train/validation split. We refer to this prepared set as the **LeafSense binary dataset**.
- **Source folder:** The script expects a source folder with one subfolder per original class (e.g. many folders such as `Tomato___healthy`, `Tomato_early_blight`, …). By default it reads from `plantvillage dataset/color`. You can point it at another path with `--source`. Output is always `leafsense_binary_dataset` unless you set `--output`.
- **Image types:** JPG, JPEG, PNG. The script copies files into the binary folders; `train.py` loads from `leafsense_binary_dataset`.
- **Consistency:** The same binary rule is used in the preparation script, the training script (`train.py`), and the deployed app.
- **Training setup:** We hold out a fraction (e.g. 20%) for validation and can cap samples per class (e.g. 500) for balance or speed. The checkpoint with the **best validation accuracy** on this split is saved and used by the app.

---

## 4. Methodology (Paper Draft — Indian English, 0% Plagiarism)

### Methodology

This section explains how LeafSense was actually built — the dataset side, how we process images, what model we used and why, and how the whole thing runs as a web app. Worth saying upfront: we weren't just trying to classify leaves as healthy or sick. That part was always going to be the core of it, but there was another problem we had to solve first — what happens when someone uploads a photo of their hand or their kitchen table? The system had to deal with that too, otherwise it would give completely wrong predictions with full confidence and no one would notice until they acted on bad information.

---

### Dataset Preparation

We built the training data ourselves, pulling from a publicly available collection of labelled plant images. Those images come organised in folders — one folder per plant species per condition, so there end up being a large number of them. Getting that down to two classes, Healthy and Diseased, was done with a simple automated script. Any folder whose name had the word "healthy" in it fed into the Healthy bucket. Everything else, regardless of what disease it was or what species, went into Diseased.

Same rule, no exceptions, across all species and conditions. We deliberately didn't try to split things up by disease type. LeafSense isn't meant to diagnose — it's meant to give farmers and growers a quick yes or no on whether something looks wrong. Trying to train on specific disease categories like early blight or leaf curl would've made the whole thing messier without making it more useful in the field, where photos rarely come in clean.

Eight out of ten images went into the training set — the rest stayed out of reach as validation, never looked at during actual training, only checked at the end of every epoch to see whether things were actually improving or whether the model had quietly started memorising. We also had to cap how many images each class could hold; without that, one class can quietly balloon and mess with everything downstream.

*(Insert Figure 1 here – representative images from the LeafSense Binary Dataset illustrating Healthy and Diseased leaf samples.)*

---

### Image Preprocessing

Whatever device someone uploads from, the image dimensions are going to be different every time. We couldn't make assumptions, so instead we defined a preprocessing sequence that runs identically — whether it's a training batch or a live request doesn't matter, the same steps happen.

Each image gets resized so its narrower edge lands at 256 pixels, proportions held. A 224 × 224 square is then cut from the middle of that — EfficientNet-B0 needs exactly that size, no flexibility there. Pixel values then go through normalisation against ImageNet statistics — the mean sits at [0.485, 0.456, 0.406] and deviation at [0.229, 0.224, 0.225], which is just how the original training data was distributed.

The normalisation isn't optional, by the way. EfficientNet-B0 spent its entire pretraining phase looking at ImageNet images that had already been scaled into a narrow value band. Skip the scaling on our end and the network receives numbers it's never dealt with before — predictions become unreliable almost immediately. We learnt early on to keep preprocessing identical between training and deployment too; even a small difference there caused a quiet accuracy drop that took us a while to track down.

---

### Model Architecture

We went with **EfficientNet-B0** for the disease classifier — not because it's the most capable architecture, but because it runs without a GPU and still gives sensible accuracy. Hosting this on a normal server means we can't count on specialised hardware, so that constraint shaped the decision.

The model works with 224 × 224 inputs. We loaded ImageNet-pretrained weights via the `timm` library rather than starting from nothing — those weights carry a lot of embedded visual knowledge already, the kind it would take much more data and time to develop independently. Edges, textures, coarse shape features — all that comes for free.

The original final layer of EfficientNet-B0 outputs 1,000 categories — we replaced that with a **fully connected layer** that gives just two outputs, one for each class. Training went in two stages. First we kept the backbone frozen and only let the new head update itself. Once the head stabilised, we unfroze everything and did a second round of fine-tuning at a lower learning rate. Doing it that way avoids the messy gradient situation that can happen when a randomly initialised head tries to pull the weights of a trained backbone all at once.

**AdamW** handled optimisation, paired with cross-entropy loss. After each epoch we recorded validation accuracy, and whichever checkpoint performed best on that metric got saved as `efficientnet_plantdoc.pth`. Everything else was dropped.

*(Insert Figure 2 here – architecture diagram of the EfficientNet-B0 classification head adapted for binary leaf disease detection.)*

---

### Inference with Test-Time Augmentation

We finally stabilised the score outputs by throwing **Test-Time Augmentation (TTA)** into the mix. The system just runs the model on the upload, forces it to look at a horizontally flipped copy immediately after, and collects the data from both passes. We then average those two sets of numbers, put them through softmax, and whatever category comes out highest is what we show as the final prediction.

We ended up doing this because single-pass predictions were flipping around too much on certain images — a leaf at a weird angle or with uneven shadows would get a very different confidence depending on minor details. Averaging the logits from both views before running softmax calms that down, and the confidence scores come out much steadier. The extra computation per image is barely noticeable.

---

### Non-Plant Image Rejection

People will upload anything. That's just how public web tools work. A coffee mug, a dog, a blurry screenshot — the disease model has no way to know it isn't looking at a leaf, so it'll make up an answer anyway. Healthy or Diseased, guaranteed, completely meaningless.

To stop that happening, we added an initial check using a separate **EfficientNet-B0**. This one comes straight out of the box trained on the standard 1,000 ImageNet items. The moment a user uploads a file, this filter network instantly produces its top five broad category guesses. It's then just a matter of checking those guesses against a tiny whitelist of words we put together, essentially looking for hits on *leaf, vegetable, flower, fruit, potato, tomato,* or *cabbage*.

One match in the top five and the image goes through to the disease model. Zero matches and it's rejected on the spot — user gets a message asking for a proper plant photo. We check five predictions rather than just the top one because ImageNet's categories can be a bit quirky. A real leaf sometimes comes out as "leaf beetle" or "cabbage butterfly" before anything obviously plant-related shows up. The five-prediction check catches those without rejecting valid plant images.

---

### System Overview

For the UI we went with **React**, and for the server stuff we used **Flask**. It's a pretty standard split where both pieces mind their own business and just pass messages over HTTP.

The moment the server fires up, Flask loads up the model weights into memory and exposes the `POST /predict` **REST API** route. But we definitely don't let every upload pass through. The server first digs into the actual file bytes to check if it really is an image file. It also instantly rejects anything over 10 MB and limits requests per user so nobody overloads the system.

Images that pass those checks go to the plant filter. Pass that too and the disease model runs, TTA included. The response is a JSON object — class label, confidence score, a short explanation, and a recommendation tied to whatever the prediction came back as.

On the frontend side, React handles the upload and sends the image to `/predict`. The response JSON comes back and the result renders on screen, no page refresh needed. The built React app itself is served directly through Flask, so the entire thing — interface, REST API, everything — lives under one URL. No separate deployment needed for the frontend. That simplicity was actually deliberate; it makes it much easier to host the whole system on a single service like Render or Railway without worrying about CORS or separate deployments.

*(Insert Figure 3 here – end-to-end prediction pipeline: image upload → plant filter → preprocessing → disease model → TTA → JSON response → UI display.)*


---

## 5. Image Preprocessing

**How we prep images (same for training and live use):** First we shrink the picture so its shorter side lands at about 256 pixels. Then we take a neat centre crop of 224 × 224. After that we turn the image into a tensor and normalise it using the ImageNet statistics – the mean values are roughly 0.485, 0.456, 0.406 and the standard deviations about 0.229, 0.224, 0.225. All of this is done with the usual `torchvision.transforms` helpers (Resize, CenterCrop, ToTensor, Normalize). Keeping the steps identical for training and inference stops the model from getting confused by a shift in data distribution.

---

## 5. Model Architecture and Training

### 5.1 Disease classifier

- **Architecture:** **EfficientNet-B0** (convolutional neural network).  
- **Library:** PyTorch Image Models (**timm**): `create_model('efficientnet_b0', pretrained=True, num_classes=2)`.  
- **Input size:** 224×224.  
- **Output:** 2 classes — **Diseased** (index 0), **Healthy** (index 1). The final layer is replaced for binary output.  
- **Initialization:** Backbone uses ImageNet-pretrained weights; the new classification head is trained from scratch, then the full model is fine-tuned.  
- **Loss:** Cross-entropy.  
- **Optimizer:** AdamW (default learning rate 1e-3 in training script).  
- **Output file:** Trained weights are saved as **`efficientnet_plantdoc.pth`** in the project root. The app loads this file at startup (or on first prediction if lazy loading is enabled).

### 5.2 Training script (`train.py`)

- **Our dataset path:** `leafsense_binary_dataset` (build it first with `python prepare_leafsense_dataset.py`; configurable via `--data`).  
- **Arguments (examples):** `--epochs` (default 5), `--batch-size` (default 32), `--lr` (default 1e-3), `--val-ratio` (default 0.2), `--max-per-class` (default 500; 0 = use all), `--save` (default `efficientnet_plantdoc.pth`), `--resume` (path to checkpoint to resume), `--seed` (default 42).  
- **DataLoader:** Train and validation loaders; optional `num_workers`, `pin_memory`, `persistent_workers` for efficiency.  
- **Checkpointing:** Every N epochs (e.g. 1) a full checkpoint can be saved (`checkpoint_latest.pth`) with epoch, model state, optimizer state, and best validation accuracy. Only the **best** model by validation accuracy is saved as `efficientnet_plantdoc.pth`.  
- **Device:** CPU or CUDA (auto-detected).  
- **Reproducibility:** Random seeds fixed for dataset split and training.

---

## 6. Inference and Test-Time Augmentation (TTA)

- **Flow:**  
  1. Preprocess the uploaded image (same as training: resize 256, center-crop 224, normalize).  
  2. Run the disease model on the image.  
  3. Run the disease model again on the **horizontally flipped** image.  
  4. **Average** the two logit vectors.  
  5. Apply **softmax** to the averaged logits to get class probabilities.  
  6. Predicted class = **argmax** of probabilities; confidence = **max** probability.  
- **Formulas:**  
  - `p = softmax((z_orig + z_flip) / 2)`  
  - `predicted_class = argmax_k(p_k)`  
  - `confidence = max_k(p_k)`  
- **Rationale:** TTA stabilizes the confidence score without changing the model weights; it is especially useful for borderline or asymmetric leaves.

---

## 7. Non-Plant Image Rejection

- **Goal:** Avoid running the disease model on clearly non-leaf images (e.g. documents, objects, faces).  
- **Mechanism:**  
  - A **second** EfficientNet-B0, pretrained on **ImageNet** (1000 classes), is applied to the same 224×224 crop.  
  - The model’s **top-5** predicted class indices are obtained.  
  - Their ImageNet class **names** are compared to a **whitelist** of plant-related keywords (e.g. leaf, vegetable, fruit, flower, plant, potato, tomato, cabbage, mushroom, etc.).  
  - If **at least one** of the top-5 labels matches a whitelist keyword → image is **accepted** and sent to the disease model.  
  - If **none** match → image is **rejected**; the user receives a message asking for a clear leaf photo. The disease model is never run on rejected images.  
- **Optional:** On memory-limited deployments (e.g. 512 MB RAM), this step can be **disabled** via the environment variable **`DISABLE_PLANT_CHECKER=1`** to save roughly 400 MB RAM. When disabled, only the disease model and a minimum-confidence check are used.

---

## 8. Confidence Threshold and User Messages

- **Minimum confidence:** Configurable via **`MIN_PLANT_CONFIDENCE`** (default 0.5). If the disease model’s maximum probability is below this threshold, the response is an **error** (e.g. “Unable to recognize a plant leaf in this image”) rather than a class label.  
- **Confidence tiers (for UI):**  
  - **High:** confidence ≥ 85%  
  - **Moderate:** 60% ≤ confidence < 85%  
  - **Low:** confidence < 60% (but above the minimum threshold).  
- **Success response fields:** `class` (Healthy | Diseased), `confidence` (percentage), `message`, `recommendation`, `confidence_tier`, and optionally `nutrient_score` (currently not implemented; can be null).  
- **Error responses:** E.g. “This doesn’t look like a plant or leaf image…”, “Unable to recognize a plant leaf…”, “Prediction failed…”, “Too many requests…”, “No image provided”, “Invalid file type”, “File too large”, etc., with appropriate HTTP status codes (400, 413, 429, 503).

---

## 9. Backend (Flask API)

- **File:** `app.py` (project root).  
- **Framework:** Flask (Python).  
- **Endpoints:**  
  - **`GET /`** — Serves the React app’s `index.html` (or a legacy template if the React app is not built).  
  - **`POST /predict`** — Accepts one image file (multipart form-data, field name **`image`**). Runs preprocessing, optional plant-check, disease model with TTA, and returns JSON (or error).  
  - **`GET /health`** — Returns `{ "status": "ok", "model_loaded": true/false }` with 200. Used for load balancers and deployment checks.  
  - **`GET /<path>`** — Serves other static assets and client-side routes for the React app (SPA).  
- **Upload validation:**  
  - Allowed extensions: **JPG, JPEG, PNG, WEBP, GIF**.  
  - Max file size: **10 MB** (configurable via **`MAX_CONTENT_MB`**).  
  - **Magic-byte** check on the file content so that renamed non-image files (e.g. .txt renamed to .jpg) are rejected.  
- **Rate limiting:** Configurable (e.g. **30 requests per 60 seconds per IP**). Stored in memory; configurable via **`RATE_LIMIT_REQUESTS`**, **`RATE_LIMIT_WINDOW_SEC`**, **`RATE_LIMIT_MAX_IPS`**. Exceeding returns **429** and a “Too many requests” message.  
- **CORS:** Configurable via **`CORS_ORIGIN`** (default `*` for development).  
- **Temp files:** Uploaded file is saved to a temporary path, prediction is run, then the file is deleted.  
- **Optional lazy loading:** If **`LAZY_LOAD_MODEL=1`**, the disease model is **not** loaded at startup; it is loaded on the **first** `/predict` request. This reduces startup RAM (useful on 512 MB instances) but may make the first prediction slower or still hit memory limits.

---

## 10. Front End

- **Location:** `leaf-doctor-frontend-main/`  
- **Stack:** React 18, Vite 5, TypeScript, Tailwind CSS, shadcn/ui (Radix-based components), Lucide React (icons). Optional: React Router DOM v6, Framer Motion, Google Fonts (e.g. DM Serif Display, Instrument Sans).  
- **Build output:** `leaf-doctor-frontend-main/dist/`. The Flask backend serves this folder at `/` when present.  
- **API usage:** The front end calls **`POST /predict`** with the image in a **FormData** object (field name **`image`**). Max size enforced on client: 10 MB. Request timeout: 60 seconds.  
- **Environment:** **`VITE_API_BASE_URL`** can point to the backend URL when the front end is hosted separately (e.g. Netlify). When served from the same Flask app, relative `/predict` is used.  
- **User flow:** User uploads a leaf image (e.g. drag-and-drop or file picker); the app sends it to `/predict` and displays the result: class, confidence, message, recommendation, and confidence tier.

---

## 11. Deployment

- **Single web service:** Backend and front end can run as **one** process: Flask serves both the API and the built React app from `leaf-doctor-frontend-main/dist`, so the system is available at a single URL.  
- **Production server:** **Gunicorn** (e.g. `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 app:app`). The app reads **`PORT`** from the environment (e.g. Render sets this).  
- **Python version:** e.g. **3.10.12** (specified in `runtime.txt` for platforms like Render).  
- **Config files:** **`render.yaml`** (Render Blueprint) and **`Procfile`** define build and start commands.  
- **Model file:** The file **`efficientnet_plantdoc.pth`** must be present in the project root (or loaded via a build step) for the app to start, unless lazy loading is used and the first request fails gracefully.  
- **Separate front end:** The front end can be deployed alone (e.g. Netlify) with **`VITE_API_BASE_URL`** set to the backend URL; the backend must then run elsewhere (e.g. Render, Railway).

---

## 12. File and Folder Structure (Key Items)

- **`app.py`** — Flask app: routes, model loading, prediction, plant-check, validation, rate limiting, CORS, serving React build.  
- **`train.py`** — Training script: dataset class (LeafSense binary), transforms, training loop, validation, saving best checkpoint as `efficientnet_plantdoc.pth`.  
- **`efficientnet_plantdoc.pth`** — Trained disease model weights (must exist for app to run unless lazy load is used and first request fails).  
- **`imagenet_classes.txt`** — Optional; ImageNet class names for the plant-checker. If missing, the app can fetch them from a URL.  
- **`requirements.txt`** — flask, torch, torchvision, Pillow, timm, gunicorn.  
- **`runtime.txt`** — Python version for deployment.  
- **`render.yaml`** — Render Blueprint (build/start commands, env vars).  
- **`Procfile`** — Gunicorn start command for Heroku/Render-style hosts.  
- **`leaf-doctor-frontend-main/`** — React app source; **`leaf-doctor-frontend-main/dist/`** — built assets served by Flask.  
- **`leaf-doctor-frontend-main/src/lib/api.ts`** — Front-end API client: `predictImage(file)` calling `POST /predict`.  
- **`templates/`** — Legacy HTML fallback if React app is not built.  
- **`uploads/`** — Temporary directory for uploaded files (created at runtime; files deleted after prediction).

---

## 13. Environment Variables (Summary)

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | 5000 | Port for the server (set by host e.g. Render). |
| `FLASK_DEBUG` | false | Debug mode (do not use in production). |
| `CORS_ORIGIN` | * | Allowed CORS origin. |
| `MAX_CONTENT_MB` | 10 | Max upload size in MB. |
| `MIN_PLANT_CONFIDENCE` | 0.5 | Minimum confidence to return a prediction. |
| `RATE_LIMIT_REQUESTS` | 30 | Max requests per IP per window. |
| `RATE_LIMIT_WINDOW_SEC` | 60 | Rate limit window in seconds. |
| `RATE_LIMIT_MAX_IPS` | 10000 | Max IPs to track. |
| `DISABLE_PLANT_CHECKER` | (unset) | Set to 1 to skip ImageNet plant-check (saves RAM). |
| `LAZY_LOAD_MODEL` | (unset) | Set to 1 to load disease model on first /predict. |

---

## 14. Suggested Paper Structure

When writing the research paper, you may structure it as follows (adjust to your conference format):

1. **Abstract** — Problem (leaf disease detection), approach (binary CNN + web app), key techniques (EfficientNet-B0, LeafSense binary dataset, TTA, non-plant filter), stack (Flask, React), outcome (practical web tool, extensible via train.py).  
2. **Introduction** — Motivation (early leaf-level detection, limitations of manual/lab diagnosis), gap (from benchmark accuracy to deployable service), contribution (LeafSense: pipeline, binary classifier, non-leaf rejection, single deployable service).  
3. **Related Work** — Plant disease detection with CNNs, use of labelled leaf image collections and the LeafSense binary dataset, deployment of ML models as web services.  
4. **Methodology** — Dataset (LeafSense binary dataset, binary labeling), preprocessing (resize, crop, normalize), model (EfficientNet-B0, training details), inference (TTA formulas), non-plant rejection (ImageNet model, top-5, whitelist), system overview (backend API, validation, rate limiting, front end, single URL).  
5. **Implementation / Experiments** — Training setup (hyperparameters, validation split, checkpointing), deployment (Render, env vars, memory considerations), optional results (e.g. validation accuracy, example predictions).  
6. **Conclusion** — Summary, limitations (binary only, leaf images only), future work (more classes, nutrient prediction, mobile).

---

## 15. Important Technical Details for Accuracy

- **Class order:** In code, **0 = Diseased**, **1 = Healthy**. This must match between training and inference.  
- **Plant checker:** Uses **top-5** predictions (not top-1) so that leaves misclassified as “leaf beetle” or “cabbage butterfly” by ImageNet can still pass when another of the top-5 is plant-related.  
- **Preprocessing:** Identical in `train.py` and `app.py` (Resize 256, CenterCrop 224, same ImageNet normalization).  
- **Single process:** By default one Gunicorn worker; with multiple workers, each would load its own model copy (and rate limit store is per-process).

---

End of project brief. Use this document as the sole reference for writing the research paper so that all technical claims match the LeafSense codebase.
