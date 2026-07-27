import os
import cv2
import shutil

# --- DIRECTORIES ---
train_img_dir = r"D:\VSCode\Fine Tune Model\backup_data\train\images"
train_lbl_dir = r"D:\VSCode\Fine Tune Model\backup_data\train\labels"

print("🚀 Executing production-grade data augmentation pipeline...")

img_files = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
augmented_count = 0

for img_file in img_files:
    base_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(train_img_dir, img_file)
    lbl_path = os.path.join(train_lbl_dir, f"{base_name}.txt")
    
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    # --- 1. LOW-PASS FILTERING (GAUSSIAN BLUR) ---
    blur_img = cv2.GaussianBlur(img, (5, 5), 0)
    blur_name = f"{base_name}_aug_blur"
    cv2.imwrite(os.path.join(train_img_dir, f"{blur_name}.jpg"), blur_img)
    if os.path.exists(lbl_path):
        shutil.copy(lbl_path, os.path.join(train_lbl_dir, f"{blur_name}.txt"))
        
    # --- 2. LINEAR PIXEL INTENSITY SCALING (CONTRAST) ---
    contrast_img = cv2.convertScaleAbs(img, alpha=1.25, beta=0)
    contrast_name = f"{base_name}_aug_contrast"
    cv2.imwrite(os.path.join(train_img_dir, f"{contrast_name}.jpg"), contrast_img)
    if os.path.exists(lbl_path):
        shutil.copy(lbl_path, os.path.join(train_lbl_dir, f"{contrast_name}.txt"))

    # --- 3. CHROMATIC DIMENSIONALITY REDUCTION (GRAYSCALE) ---
    gray_channels = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.merge([gray_channels, gray_channels, gray_channels])
    gray_name = f"{base_name}_aug_gray"
    cv2.imwrite(os.path.join(train_img_dir, f"{gray_name}.jpg"), gray_img)
    if os.path.exists(lbl_path):
        shutil.copy(lbl_path, os.path.join(train_lbl_dir, f"{gray_name}.txt"))

    # --- 4. SPATIAL GEOMETRIC TRANSFORMATION (HORIZONTAL FLIP) ---
    flip_img = cv2.flip(img, 1)  # 1 indicates horizontal flipping axis, i avoided vertical bcoz its bad data
    flip_name = f"{base_name}_aug_flip"
    cv2.imwrite(os.path.join(train_img_dir, f"{flip_name}.jpg"), flip_img)
    
    # Mathematical transformation of YOLO bounding boxes for horizontal reflection
    if os.path.exists(lbl_path):
        flipped_bboxes = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = parts[0]
                    cx, cy, bw, bh = map(float, parts[1:])
                    # Invert the normalized horizontal center point relative to the boundary
                    new_cx = 1.0 - cx
                    flipped_bboxes.append(f"{cls_id} {new_cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    
        with open(os.path.join(train_lbl_dir, f"{flip_name}.txt"), "w") as f:
            f.writelines(flipped_bboxes)

    augmented_count += 4

print(f"\n✨ Data Augmentation Engine Completed Successfully!")
print(f"   • Baseline Dataset Size: {len(img_files)}")
print(f"   • Synthesized Augmentations Added: {augmented_count}")
print(f"   • Total Expanded Training Pipeline: {len(os.listdir(train_img_dir))} files.")