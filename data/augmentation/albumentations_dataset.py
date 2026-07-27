"""
===================================================================================================
Pipeline Title: Automated Dataset Augmentation & YOLO Label Map Sync (Fixed Bounds & Float Class)
Description:
    Processes batch directories for train/valid images and synchronizes YOLO bounding box labels.
===================================================================================================
"""
import os
import cv2
import glob
import albumentations as A

# 1. Directory Path Routing Configurations
SRC_DIR = r"D:\VSCode\Fine Tune Model\data\unaugmented_cropped_split_256_overlap_50"
OUT_DIR = r"D:\VSCode\Fine Tune Model\data\augmented_cropped_split_256_overlap_50"


# 2. Production Calibrated Albumentations Pipeline
augmentation_pipeline = A.Compose([
    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.10, 0.25), p=0.4),
    A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.3),
    A.ImageCompression(quality_range=(50, 85), compression_type="jpeg", p=0.4),
    #A.Downscale(scale_range=(0.40, 0.75), interpolation_pair={"downscale": cv2.INTER_LINEAR, "upscale": cv2.INTER_LINEAR}, p=0.4),
    A.ChannelShuffle(p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_yolo_labels(label_path):
    """Parses raw text labels into bounding boxes and clamps them securely between 0.0 and 1.0."""
    bboxes = []
    class_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(float(parts[0])) # Safe float string parsing
                    x_c, y_c, w, h = map(float, parts[1:])
                    
                    x_min = x_c - (w / 2)
                    y_min = y_c - (h / 2)
                    x_max = x_c + (w / 2)
                    y_max = y_c + (h / 2)
                    
                    x_min = max(0.0, min(x_min, 1.0))
                    y_min = max(0.0, min(y_min, 1.0))
                    x_max = max(0.0, min(x_max, 1.0))
                    y_max = max(0.0, min(y_max, 1.0))
                    
                    x_c = (x_min + x_max) / 2.0
                    y_c = (y_min + y_max) / 2.0
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    if w > 0.001 and h > 0.001:
                        bboxes.append([x_c, y_c, w, h])
                        class_labels.append(cls_id)
    return bboxes, class_labels

def save_yolo_labels(out_path, bboxes, class_labels):
    """Writes updated coordinate locations back to the output folder structure with strict integer classes."""
    with open(out_path, 'w') as f:
        for bbox, cls_id in zip(bboxes, class_labels):
            clean_cls = int(float(cls_id)) # Force strict integer mapping (e.g., 1 instead of 1.0)
            f.write(f"{clean_cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

# 3. Main Data Execution Engine Loop (only train)
for split in ['train']:
    img_src_folder = os.path.join(SRC_DIR, split, 'images')
    lbl_src_folder = os.path.join(SRC_DIR, split, 'labels')
    
    img_out_folder = os.path.join(OUT_DIR, split, 'images')
    lbl_out_folder = os.path.join(OUT_DIR, split, 'labels')
    
    os.makedirs(img_out_folder, exist_ok=True)
    os.makedirs(lbl_out_folder, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(img_src_folder, "*.jpg")) + glob.glob(os.path.join(img_src_folder, "*.png"))
    print(f"🚀 Processing Split Location: [{split}] | Target Size: {len(image_paths)} images found.")
    
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_src_folder, f"{base_name}.txt")
        
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = read_yolo_labels(lbl_path)
        
        if len(bboxes) == 0:
            cv2.imwrite(os.path.join(img_out_folder, f"aug_{base_name}.jpg"), image)
            save_yolo_labels(os.path.join(lbl_out_folder, f"aug_{base_name}.txt"), [], [])
            continue
            
        try:
            augmented = augmentation_pipeline(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
            aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            aug_boxes = augmented['bboxes']
            aug_classes = augmented['class_labels']
            
            cv2.imwrite(os.path.join(img_out_folder, f"aug_{base_name}.jpg"), aug_img)
            save_yolo_labels(os.path.join(lbl_out_folder, f"aug_{base_name}.txt"), aug_boxes, aug_classes)
            
        except Exception as e:
            cv2.imwrite(os.path.join(img_out_folder, f"aug_{base_name}.jpg"), image)
            save_yolo_labels(os.path.join(lbl_out_folder, f"aug_{base_name}.txt"), bboxes, class_labels)

print("🎉 Batch dataset compilation completed successfully with strict integer classes!")