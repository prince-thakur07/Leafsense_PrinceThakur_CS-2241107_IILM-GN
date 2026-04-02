# LeafSense — Technologies & ML Algorithms Used

## GitHub Repository
**Link:** https://github.com/abhishekhbihari007/leafSense

---

## 1. Machine Learning / Deep Learning

| Item | Technology / Algorithm |
|------|-------------------------|
| **Main model (disease detection)** | **EfficientNet-B0** (CNN) — binary classification: Healthy vs Diseased |
| **Model creation & loading** | **timm** (PyTorch Image Models) — `create_model('efficientnet_b0', num_classes=2)` |
| **Plant vs non-plant check** | **EfficientNet-B0** pretrained on **ImageNet** (1000 classes) — used to reject non-leaf images (e.g. bench, document, person) |
| **Framework** | **PyTorch** (`torch`) |
| **Image preprocessing** | **torchvision.transforms** — Resize(256), CenterCrop(224), ToTensor, ImageNet normalization |
| **Inference trick** | **Test-Time Augmentation (TTA)** — average logits from original + horizontally flipped image for better confidence |
| **Output** | **Softmax** for confidence; **argmax** for class (Healthy / Diseased) |

### Model files
- `efficientnet_plantdoc.pth` — trained weights for disease model (2 classes)
- `imagenet_classes.txt` — ImageNet class names for plant-checker logic

---

## 2. Backend

| Item | Technology |
|------|------------|
| **API server** | **Flask** (Python) |
| **Image handling** | **Pillow (PIL)** — load, convert RGB |
| **Dependencies** | See `requirements.txt`: flask, torch, torchvision, Pillow, timm |

---

## 3. Frontend

| Item | Technology |
|------|------------|
| **UI framework** | **React 18** |
| **Build tool** | **Vite 5** |
| **Language** | **TypeScript** |
| **Styling** | **Tailwind CSS**, **tailwindcss-animate** |
| **UI components** | **Radix UI** (shadcn/ui style), **Lucide React** (icons) |
| **Routing** | **React Router DOM v6** |
| **Animations** | **Framer Motion** |
| **Fonts** | **DM Serif Display** (headings), **Instrument Sans** (body) — Google Fonts |

---

## 4. Summary for Research / Report

- **Dataset:** LeafSense binary dataset (Healthy / Diseased), built from a source folder of labelled leaf images
- **ML algorithm:** EfficientNet-B0 (Convolutional Neural Network)
- **Task:** Binary image classification (Healthy leaf / Diseased leaf)
- **Extra ML use:** ImageNet-pretrained EfficientNet-B0 for “plant vs non-plant” filtering
- **Libraries:** PyTorch, torchvision, timm, Flask, Pillow
- **Frontend stack:** React, TypeScript, Vite, Tailwind CSS

---

*Last updated with project code as pushed to GitHub.*
