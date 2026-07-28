"""
===================================================================================================
Script Title: Augmented Dataset Label Inspector Engine (Robust Parsing)
Description:
    Loads augmented image assets alongside recalculated YOLO coordinate annotations, 
    rendering bounding boxes visually to verify boundary and sync integrity.
===================================================================================================
"""
import os
import cv2
import glob
import random

# 1. Pipeline Routing Configurations
AUG_DIR = r"D:\VSCode\Fine Tune Model\data\augmented_cropped_split\train"
CLASS_MAP = {0: "human", 1: "car", 2: "motor"}
COLOR_MAP = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)} # Red=Human, Green=Car, Blue=Motor

def verify_augmented_labels(num_samples=3):
    img_folder = os.path.join(AUG_DIR, "images")
    lbl_folder = os.path.join(AUG_DIR, "labels")
    
    image_paths = glob.glob(os.path.join(img_folder, "*.jpg"))
    if not image_paths:
        print(f"⚠️ No images found at target destination: {img_folder}")
        return
        
    print(f"👁️ Randomly inspecting {num_samples} frames from the augmented folder pool...")
    samples = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    for idx, img_path in enumerate(samples):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_folder, f"{base_name}.txt")
        
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(float(parts[0])) # Fixed: parse float strings cleanly
                        x_c, y_c, box_w, box_h = map(float, parts[1:])
                        
                        x1 = int((x_c - box_w / 2) * w)
                        y1 = int((y_c - box_h / 2) * h)
                        x2 = int((x_c + box_w / 2) * w)
                        y2 = int((y_c + box_h / 2) * h)
                        
                        color = COLOR_MAP.get(cls_id, (255, 255, 255))
                        label_text = CLASS_MAP.get(cls_id, f"Class {cls_id}")
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label_text, (x1, max(15, y1 - 5)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        output_filename = f"inspect_check_{idx}.jpg"
        cv2.imwrite(output_filename, img)
        print(f"💾 Rendered test target output file generated at: D:\\VSCode\\Fine Tune Model\\src\\utils\\{output_filename}")

if __name__ == "__main__":
    verify_augmented_labels(num_samples=3)