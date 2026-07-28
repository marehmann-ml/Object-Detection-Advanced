
"""
===================================================================================================
Title: High-Altitude Drone Multiclass Object Detection via SAHI & OpenCV

Description:
    This script addresses the challenge of scale variance (objects changing drastically in size due 
    to camera distance) on high-altitude drone footage. Standard models downsample full frames, 
    causing severe feature degradation (loss of fine pixel details) for tiny, distant targets.

    To resolve this, we leverage:
    1. OpenCV (cv2): Manually handles native frame extraction, real-time video streaming metadata 
       configuration, and custom, lightweight bounding box and label rendering.
    2. SAHI AutoDetectionModel: Packages our custom fine-tuned YOLO26s weight matrix safely inside 
       the slicing wrapper engine.
    3. SAHI get_sliced_prediction: Generates temporary overlapping window slices (512x512 tiles 
       with 20% redundancy) across the frame. This allows the model to process tiny objects at 
       their original, high-resolution pixel scale. A tailored postprocess match threshold (0.45 IoU) 
       is used to prevent NMS (Non-Maximum Suppression) from suppressing closely packed, adjacent 
       targets in high-density traffic scenes.
===================================================================================================
"""


import os
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def main():
    model_path = r"D:\VSCode\Fine Tune Model\runs\detect\yolo26s_raw_finetuned_model\weights\best.pt"
    video_source_path = r"C:\Users\abdur.rehman\Downloads\front_lawn.mp4"
    output_video_path = r"D:\VSCode\Fine Tune Model\outputs\sahi_inference\Close View\front_lawn_CV_sahi.mp4"

    cap = cv2.VideoCapture(video_source_path)
    if not cap.isOpened(): 
        print("❌ Error: Cannot open source video.")
        return

    # Extracting video metadata & output config
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=model_path, confidence_threshold=0.35, device="cuda:0"
    )

    color_map = {"car": (0, 255, 0), "human": (0, 0, 255), "motor": (255, 255, 0)}
    
    # Static rendering properties
    box_thickness = 2
    scale = 0.55 
    txt_th = 2   

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        res = get_sliced_prediction(
            image=frame, detection_model=model, slice_height=512, slice_width=512,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2, auto_slice_resolution=False, verbose=0,
            postprocess_match_threshold=0.45,
        )

        for p in res.object_prediction_list:
            b = p.bbox
            x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
            color = color_map.get(p.category.name.lower(), (255, 0, 0))

            # Render bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            
            txt = f"{p.category.name} {float(p.score.value):.2f}"
            # FIX: Renamed variable to text_height to prevent overwriting the thickness property
            (tw, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, txt_th)
            
            ty = max(0, y1 - text_height - 10)
            cv2.rectangle(frame, (x1, ty), (x1 + tw, y1), color, -1)
            cv2.putText(
                frame, txt, (x1, y1 - 5 if y1 - text_height - 10 > 0 else y1 + text_height + 5),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), txt_th
            )

        out.write(frame)
        count += 1
        print(f"⏳ Progress: {count}/{total}", end="\r")


    cap.release()
    out.release()
    print("\n✨ Inference complete! File processed without variable clashing.")

if __name__ == "__main__":
    main()