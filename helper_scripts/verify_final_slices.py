import os
import cv2
import shutil

# --- PATHS ---
images_dir = r"D:\VSCode\Fine Tune Model\backup_data\train\images"
labels_dir = r"D:\VSCode\Fine Tune Model\backup_data\train\labels"
output_dir = r"D:\VSCode\Fine Tune Model\All inspections\inspections_check_FINAL_with aug"

# Clean old check folder
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# VisDrone class mapping array
class_map = {0: "Human", 1: "Car", 2: "Motor"}
color_map = {
    0: (0, 0, 255),    # Red for Human
    1: (0, 255, 0),    # Green for Car
    2: (255, 255, 0)   # Cyan for Motor
}

# Collect all text files that explicitly belong to the horizontal reflection split
all_labels = os.listdir(labels_dir)
flip_labels = [f for f in all_labels if "_aug_flip" in f and f.lower().endswith('.txt')]

# Filter out empty files so we only inspect frames with active targets
populated_flip_labels = [f for f in flip_labels if os.path.getsize(os.path.join(labels_dir, f)) > 0]

# Pick the first 3 populated flipped files to inspect
sample_size = min(3, len(populated_flip_labels))
selected_samples = populated_flip_labels[:sample_size]

print(f"🎯 Specifically targeting {sample_size} horizontal reflection (flip) tiles for verification...")

for lbl_file in selected_samples:
    base_name = os.path.splitext(lbl_file)[0]
    img_file = f"{base_name}.jpg"
    
    img_path = os.path.join(images_dir, img_file)
    lbl_path = os.path.join(labels_dir, lbl_file)
    
    if not os.path.exists(img_path):
        continue
        
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w, _ = img.shape
    
    with open(lbl_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            
            # Convert normalized decimal values back to absolute image pixels
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            label_text = class_map.get(cls_id, f"Unknown ({cls_id})")
            color = color_map.get(cls_id, (255, 0, 0))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
    output_path = os.path.join(output_dir, f"check_{img_file}")
    cv2.imwrite(output_path, img)

print(f"✅ targeted check files generated! Open this folder to review the flips: {output_dir}")