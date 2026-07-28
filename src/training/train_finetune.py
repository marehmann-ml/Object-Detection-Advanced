
"""
===================================================================================================
Pipeline Title: Stage-2 Advanced Fine-Tuning Script with Isolated Hyperparameters
Description:
    Loads pre-trained checkpoint weights from prev training run,
    points to the newly augmented data folder, and overrides hyperparameter arguments 
    using our clean isolated configuration file.
===================================================================================================
"""
import os
from ultralytics import YOLO

# 1. Strict Path Configurations
YESTERDAY_BEST_WEIGHTS = r"D:\VSCode\Fine Tune Model\weights\train-4\weights\best.pt"
#r"D:\VSCode\Fine Tune Model\runs\detect\yolo26s_raw_finetuned_model_best\weights\best.pt"
                        
       
DATASET_CONFIG_YAML = r"D:\VSCode\Fine Tune Model\src\config\data_config.yaml"  
TUNED_HYPERPARAMETERS_YAML = r"D:\VSCode\Fine Tune Model\src\config\hyperpara.yaml"

def run_production_tuning(): 
    # Verify that yesterday's checkpoint weights exist before wasting execution overhead
    if not os.path.exists(YESTERDAY_BEST_WEIGHTS):
        raise FileNotFoundError(f"⚠️ Target checkpoint weights not found at: {YESTERDAY_BEST_WEIGHTS}")
        
    if not os.path.exists(TUNED_HYPERPARAMETERS_YAML):
        raise FileNotFoundError(f"⚠️ Custom hyperparameters configuration file missing at: {TUNED_HYPERPARAMETERS_YAML}")

    print("🚀 Initializing Stage-2 Fine-Tuning Lab Pipeline...")
    print(f"📦 Seeding Base Model Weights From Yesterday's Run: {YESTERDAY_BEST_WEIGHTS}")
    
    # Load yesterday's best weights checkpoint structure
    model = YOLO(YESTERDAY_BEST_WEIGHTS)
    
    print("🔥 Starting Model Training Loop with custom parameters...")
    # Execute the training run pointing to your custom configs
    model.train(
        data=DATASET_CONFIG_YAML,
        cfg=TUNED_HYPERPARAMETERS_YAML,     # Loads your isolated freeze=4, multi_scale=True, loss settings
        project=r"D:\VSCode\Fine Tune Model\runs\detect",
        name="yolo26s_augmented_finetuned_256_50_dataset_768"  
    )

if __name__ == "__main__":
    run_production_tuning()

