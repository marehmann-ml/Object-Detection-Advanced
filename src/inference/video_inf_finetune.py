
import os
import cv2
from ultralytics import YOLO

def main():
    # 1. Paths matching your exact instructions
    model_path = r"D:\VSCode\OpenCV\multiclass_det_2\runs\detect\robo&visd_finetune_yolo26s_final_run\weights\best.pt"
    video_path = r"C:\Users\abdur.rehman\Downloads\road_det_new.mp4"
    output_dir = r"D:\VSCode\OpenCV\multiclass_det_2\finetuned model detection"
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "det_02.mp4")

    # Initialize the YOLO model instance inside main
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Keeps giant background boxes out
    max_w_pixels = width * 0.15
    max_h_pixels = height * 0.15

    # Color Mapping Match
    color_map = {
        "Car": (0, 255, 0),       # Vibrant Green
        "Human": (0, 0, 255),     # Clear Red
        "Motor": (255, 255, 0)    # Bright Cyan
    }
    default_color = (255, 0, 0) 

    print("🚀 Running prediction with Case-Insensitive Overlap Recovery active...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Running prediction at imgsz=768
        results = model.predict(frame, imgsz=512, conf=0.05, iou=0.70, verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Instantly drop the large static background blocks
                if box_width > max_w_pixels or box_height > max_h_pixels:
                    continue 
                
                cls_id = int(box.cls[0])
                
                # Case-insensitivity fix: .capitalize() forces 'human' -> 'Human'
                raw_label = model.names[cls_id]
                label = raw_label.capitalize() 
                
                conf = float(box.conf[0])
                
                # Keep vehicles clean at 0.50, humans bypass down to absolute limits
                if label != "Human" and conf < 0.25:
                    continue
                if label == "Human" and conf < 0.05:
                    continue
                
                color = color_map.get(label, default_color)
                
                # Render bounding box annotations onto frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"✅ Detections unlocked! Output successfully saved to: {output_path}")

if __name__ == "__main__":
    main()