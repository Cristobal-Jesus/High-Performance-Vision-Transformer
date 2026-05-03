import os
import shutil

# =========================================================
# CONFIGURACIÓN
# =========================================================

DATASET_DIR = "dataset2"   # carpeta con las clases
TEST_DIR = "Test"         # carpeta destino
IMAGES_PER_CLASS = 1000   # imágenes a mover por clase

# =========================================================
# NO TOCAR NADA A PARTIR DE AQUÍ
# =========================================================

os.makedirs(TEST_DIR, exist_ok=True)

for class_name in sorted(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"\nProcesando clase: {class_name}")

    images = sorted([
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not images:
        print("  ⚠️ No hay imágenes")
        continue

    if len(images) < IMAGES_PER_CLASS:
        print(f"  ⚠️ Solo hay {len(images)} imágenes, se moverán todas")

    selected = images[:IMAGES_PER_CLASS]

    for idx, img_name in enumerate(selected, start=1):
        src = os.path.join(class_path, img_name)
        ext = os.path.splitext(img_name)[1]
        new_name = f"{class_name}_{idx:04d}{ext}"
        dst = os.path.join(TEST_DIR, new_name)

        shutil.move(src, dst)

    print(f"Movidas {len(selected)} imágenes a Test")

print("\nCarpeta Test creada y dataset actualizado")
