"""
Build the LeafSense binary leaf dataset from a source folder of labelled leaf images.

Reads from a source folder with one subfolder per class (folder name containing
'healthy' -> Healthy, else -> Diseased), copies images into:
  leafsense_binary_dataset/Healthy/
  leafsense_binary_dataset/Diseased/

Run this once before training. Then point train.py at leafsense_binary_dataset.
"""
import os
import argparse
import shutil
from pathlib import Path

def is_healthy(folder_name: str) -> bool:
    return "healthy" in folder_name.lower()

def main():
    parser = argparse.ArgumentParser(description="Build LeafSense binary dataset from source leaf images.")
    parser.add_argument(
        "--source",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plantvillage dataset", "color"),
        help="Source folder with one subfolder per class (default: plantvillage dataset/color)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "leafsense_binary_dataset"),
        help="Output root folder (default: leafsense_binary_dataset)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Max images per binary class (default: all). Use e.g. 500 for faster prep.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    out_root = Path(args.output)
    if not source.is_dir():
        print(f"Error: Source folder not found: {source}")
        print("Place the source leaf images there and run again.")
        return 1

    healthy_dir = out_root / "Healthy"
    diseased_dir = out_root / "Diseased"
    healthy_dir.mkdir(parents=True, exist_ok=True)
    diseased_dir.mkdir(parents=True, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    healthy_paths = []
    diseased_paths = []

    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            if f.suffix in exts:
                if is_healthy(class_dir.name):
                    healthy_paths.append(f)
                else:
                    diseased_paths.append(f)

    if args.max_per_class and args.max_per_class > 0:
        import random
        random.seed(42)
        random.shuffle(healthy_paths)
        random.shuffle(diseased_paths)
        healthy_paths = healthy_paths[: args.max_per_class]
        diseased_paths = diseased_paths[: args.max_per_class]

    def copy_into(paths, dest_dir, prefix="img"):
        for i, src in enumerate(paths):
            name = f"{prefix}_{i:05d}{src.suffix}"
            shutil.copy2(src, dest_dir / name)

    copy_into(healthy_paths, healthy_dir, "healthy")
    copy_into(diseased_paths, diseased_dir, "diseased")

    n_healthy = len(healthy_paths)
    n_diseased = len(diseased_paths)
    print(f"LeafSense binary dataset written to: {out_root}")
    print(f"  Healthy:  {n_healthy} images -> {healthy_dir}")
    print(f"  Diseased: {n_diseased} images -> {diseased_dir}")
    print(f"Total: {n_healthy + n_diseased} images.")
    print("Train with: python train.py --data", str(out_root))
    return 0

if __name__ == "__main__":
    exit(main())
