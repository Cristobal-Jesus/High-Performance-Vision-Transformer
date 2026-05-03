# resize_dataset.py
import os
from pathlib import Path
from multiprocessing import Pool
from PIL import Image

SRC = Path("/home/bejeque/nhernang/Cristobal/pytorch_models/assets/dataset")
DST = Path("/home/bejeque/nhernang/Cristobal/pytorch_models/assets/dataset_256")

def resize_image(args):
    src_path, dst_path = args
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize((256, 256), Image.LANCZOS)
            img.save(dst_path, "JPEG", quality=90)
    except Exception as e:
        print(f"ERROR {src_path}: {e}")

def collect_tasks():
    tasks = []
    for class_dir in sorted(SRC.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                dst = DST / class_dir.name / img_path.name
                tasks.append((img_path, dst))
    return tasks

if __name__ == "__main__":
    tasks = collect_tasks()
    print(f"Total images: {len(tasks)}")
    with Pool(processes=32) as pool:
        for i, _ in enumerate(pool.imap_unordered(resize_image, tasks, chunksize=64)):
            if i % 10000 == 0:
                print(f"  {i}/{len(tasks)}")
    print("Done.")