import os
import torch
import torch.nn.functional as F
from timm import create_model
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from train import LeafSenseBinary, get_transforms, NUM_CLASSES, DEVICE


DATA_DIR = "leafsense_binary_dataset"
MODEL_PATH = "efficientnet_plantdoc.pth"
VAL_RATIO = 0.2
SEED = 42


def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    if not os.path.isdir(DATA_DIR):
        print(f"Dataset folder not found: {DATA_DIR}")
        return

    full_dataset = LeafSenseBinary(DATA_DIR, transform=get_transforms(), max_per_class=None)
    n_total = len(full_dataset)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val

    print(f"Total samples: {n_total}  (train={n_train}, val={n_val})")

    gen = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=gen
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1]  # probability of Healthy (class 1)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Binary predictions with threshold 0.5 on Healthy probability
    preds = [1 if p >= 0.5 else 0 for p in all_probs]

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels, preds))

    print("\nClassification report (precision, recall, F1):")
    print(classification_report(all_labels, preds, target_names=["Diseased", "Healthy"]))

    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"\nROC AUC: {auc:.4f}")
    except ValueError as e:
        print(f"\nROC AUC could not be computed: {e}")


if __name__ == "__main__":
    main()

