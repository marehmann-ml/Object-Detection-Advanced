# Object-Detection-Advanced
Advanced object detection workflows using YOLO, SAHI, custom datasets, fine-tuning, and computer vision techniques.

Object Detection Advanced

Advanced object detection workflows developed during my Machine Learning Internship, focusing on small object detection from drone imagery using modern Computer Vision techniques.

This repository documents my learning and implementation journey beyond basic object detection, covering custom dataset creation, annotation workflows, image slicing, data augmentation, fine-tuning, SAHI inference, and model evaluation.

The work presented here was developed through continuous experimentation, discussions with senior engineers, and iterative improvements while working on real-world aerial imagery.

Project Overview

This repository demonstrates an end-to-end object detection pipeline for detecting small objects from high-altitude drone imagery.

The complete workflow includes:

Multi-Class Object Detection
Custom Office Dataset Preparation
Roboflow Annotation
Dataset Cleaning & Verification
Image Slicing
Data Augmentation
Fine-Tuning YOLO Models
SAHI Inference
Performance Evaluation
Utility Scripts
Workflow
Drone Video
      │
      ▼
Frame Extraction
      │
      ▼
Annotation (Roboflow)
      │
      ▼
Dataset Cleaning
      │
      ▼
Utility Scripts
      │
      ▼
Image Slicing
      │
      ▼
Data Augmentation
      │
      ▼
Fine-Tuning
      │
      ▼
SAHI Inference
      │
      ▼
Evaluation & Results
Repository Structure
Object-Detection-Advanced/

│── 01_Multiclass_Detection
│── 02_Custom_Dataset
│── 03_Roboflow_Annotation
│── 04_Data_Augmentation
│── 05_Fine_Tuning
│── 06_SAHI_Inference
│── 07_Utilities
│── 08_Results
│
├── README.md
└── LICENSE
Technologies Used
Python
OpenCV
Ultralytics YOLO
SAHI
Roboflow
Albumentations
Streamlit
NumPy
Matplotlib
Key Learning Areas

During this project I gained practical experience in:

Small Object Detection
Drone Vision
Multi-Class Detection
Dataset Engineering
Annotation Workflows
Fine-Tuning
SAHI-based Inference
Data Augmentation
Performance Evaluation
Computer Vision Pipeline Development
Repository Status

This repository is continuously updated as new experiments, training pipelines, and evaluation techniques are developed throughout my internship.
