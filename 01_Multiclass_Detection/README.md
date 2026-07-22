# Multi-Class Object Detection

This module focuses on transitioning from single-class object detection to a multi-class drone object detection pipeline using the VisDrone dataset.

The objective was to detect three target classes from aerial imagery:

- Human
- Car
- Motor

The complete workflow involved dataset preparation, class mapping, preprocessing, model training, and performance evaluation.

---

## Project Objectives

- Convert the original VisDrone annotations into a custom three-class dataset.
- Develop preprocessing scripts for automated dataset preparation.
- Configure YOLO-compatible dataset structure.
- Train and evaluate multiple object detection models.
- Improve small object detection performance through iterative experimentation.

---

## Dataset

Dataset Used:

- VisDrone Dataset

Target Classes

| Original Class | New Class |
|---------------|-----------|
| pedestrian | Human |
| people | Human |
| car | Car |
| motor | Motor |

---

## Workflow

VisDrone Dataset

↓

Class Mapping

↓

Dataset Cleaning

↓

YOLO Dataset Generation

↓

Training

↓

Evaluation

---

## Repository Contents

```text
01_Multiclass_Detection/

README.md

scripts/

configs/

outputs/

images/
```

---

## Key Learnings

- Multi-class dataset preparation
- YOLO dataset formatting
- Class mapping
- Data preprocessing
- Model evaluation
- Small object detection

---

## Results

This module established the foundation for all subsequent work including:

- Custom dataset creation
- Fine-tuning
- SAHI inference
- Office-road object detection

Those advanced workflows are covered in the later modules of this repository.
