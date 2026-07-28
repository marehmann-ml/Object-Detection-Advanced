"""
===================================================================================================
Pipeline Title: Native SAHI Video Inference Pipeline
Description:
    Processes video data natively using SAHI's built-in 'predict_video' utility.
    Bypasses custom cv2 frame loops entirely, utilizing only native documentation API protocols.
===================================================================================================
"""

from sahi.predict import predict

def run_native_video_inference():
    
    model_path = r"D:\VSCode\Fine Tune Model\runs\detect\yolo26s_raw_finetuned_model_best\weights\best.pt"
    video_source_path = r"C:\Users\abdur.rehman\Downloads\road_det_new.mp4"
    target_project_dir = r"D:\VSCode\Fine Tune Model\outputs\sahi_inference\SAHI_runs"
    run_name = "Road_Run"

    # 3. Execute Pure Video Slicing Inference
    # SAHI automatically detects .mp4 in the 'source' and outputs a video natively.
    predict(
        model_type="ultralytics",  
        model_path=model_path,
        model_confidence_threshold=0.35,  
        model_device="cuda:0",
        source=video_source_path,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.20,
        overlap_width_ratio=0.20,
        postprocess_match_threshold=0.45,
        project=target_project_dir,
        name=run_name,
        visual_bbox_thickness=2,
        visual_text_thickness=2,
        visual_text_size=0.7  
    )

if __name__ == "__main__":
    run_native_video_inference()