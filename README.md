# Computer Vision Laboratory Assignments

This repository contains three lab assignments from the Spring 2024–2025 BBM 418 Computer Vision course at Hacettepe University. Each assignment explores a key concept in computer vision: document rectification, image classification with CNNs, and object detection using YOLO.

---

## Assignment 1: Document Rectification with Hough Transform and RANSAC

**Objective:**  
Correct geometric distortions in document images (e.g., perspective skew, folds, curvature) using classical computer vision techniques.

**Approach:**
- **Edge Detection:** Preprocessed input images with Gaussian blur and Canny edge detection.
- **Line Detection:** Applied Hough Transform to detect potential document edges.
- **Line Refinement:** Used RANSAC to robustly fit straight lines by eliminating outliers.
- **Corner Detection:** Intersected lines to find document corners.
- **Geometric Transformation:** Computed homography to rectify distorted images.
- **Evaluation:** Compared rectified images to ground-truth using Structural Similarity Index (SSIM).

**Results:**
- Six distortion categories evaluated: curved, fold, incomplete, perspective, random, rotate.
- Best SSIM observed in *perspective* category, showing strong performance on planar distortions.
- Lower performance in *curved* and *rotate* categories, due to limitations of single homography on non-planar surfaces.

---

## Assignment 2: Image Classification with CNNs

**Objective:**  
Classify food images into 11 categories using CNNs, trained from scratch and via transfer learning.

**Part 1 – From Scratch:**
- Implemented two CNNs:
  - Plain 5-layer CNN
  - 5-layer CNN with residual connections
- Tuned learning rate and batch size (grid search).
- Applied dropout for regularization.
- Evaluated using training/validation/test accuracy and confusion matrices.

**Part 2 – Transfer Learning:**
- Used MobileNetV2 pretrained on ImageNet.
- Tested two strategies:
  - Fine-tune only the FC layer.
  - Fine-tune FC + last two convolutional layers.
- Achieved higher accuracy and better generalization with transfer learning.

**Results:**
- Residual CNN (from scratch) outperformed plain CNN.
- Transfer learning with MobileNetV2 yielded best test accuracy (~74%) and reduced overfitting.
- Model performance analyzed through loss curves and confusion matrices.

---

## Assignment 3: Object Detection and Counting with YOLOv8

**Objective:**  
Detect and count cars in drone-based parking lot images using YOLOv8.

**Approach:**
- Trained YOLOv8n model under four configurations:
  1. Freeze first 5 blocks
  2. Freeze first 10 blocks
  3. Freeze first 21 blocks
  4. Train entire model
- Compared predictions with ground truth using:
  - Exact Match Accuracy
  - Mean Squared Error (MSE)
  - Precision and Recall
- Visualized bounding box predictions and analyzed missed detections.

**Results:**
- Full model training achieved the best detection accuracy.
- Freezing fewer layers led to better fine-tuning and object localization.
- Detection performance depended heavily on chosen hyperparameters and training strategy.

---

## Notes

Each assignment includes a corresponding report with detailed methodology, experimental setup, and results. All models were implemented using Python, NumPy, PyTorch, and Ultralytics YOLOv8 where appropriate.

