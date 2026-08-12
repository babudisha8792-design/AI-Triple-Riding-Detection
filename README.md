# PyTorch Three-Person-on-One-Bike Detection System - Setup & User Guide

A complete Python Web Application Platform and Standalone CLI Inference system powered by **PyTorch/YOLO (`best.pt`)**, OpenCV frame annotation, Flask, and a modern glassmorphism dark UI.

The system is designed to detect motorcycles/bikes and count the number of people riding on each bike.

---

## 🚨 Main Detection Rule

The primary purpose of this project is to detect **three or more people travelling on a single bike**.

The system follows this logic:

```text
1 Person on Bike   → ✓ SAFE (LOW Risk, Violation: NO)
2 Persons on Bike  → ✓ SAFE (LOW Risk, Violation: NO)
3 Persons on Bike  → ⚠ THREAT DETECTED (HIGH Risk, Violation: YES)
4+ Persons on Bike → ⚠ THREAT DETECTED (HIGH Risk, Violation: YES)
```

When **3 or more people are detected on the same bike**, the system displays:

> **⚠ THREE PERSONS ON ONE BIKE — THREAT DETECTED**

When only 1 or 2 people are detected:

> **✓ SAFE — NO VIOLATION DETECTED**

The system displays the detected people using bounding boxes and provides the detection count and confidence score.

---

## 📁 Project Structure

```text
three_person_bike_detection_pytorch_app/
│
├── best.pt                    <-- Your trained YOLO/PyTorch model (Place here)
├── app.py                     <-- Flask Web Server & REST API
├── detect.py                  <-- PyTorch/YOLO Inference Engine & CLI Tool
├── create_samples.py          <-- Utility to generate instant demo test images
├── requirements.txt           <-- Python Package Dependencies
│
├── templates/
│   └── index.html             <-- Modern Glassmorphism Dashboard UI
│
├── static/
│   ├── style.css              <-- Dark Glassmorphism Theme Styling
│   ├── script.js              <-- Frontend REST API & AI Scanner Interface
│   ├── samples/               <-- Generated instant sample test images
│   └── uploads/               <-- Processed annotated images & videos
│
└── README.md                  <-- Setup & Usage Guide
```

---

## ⚡ Quick Setup Instructions

### Step 1: Clone or Copy Project Files

Ensure all files are placed in your project root directory.

### Step 2: Add Your `best.pt` Model

Copy your trained PyTorch/YOLO model into the project root folder:

```text
three_person_bike_detection_pytorch_app/
│
├── best.pt
├── app.py
├── detect.py
└── ...
```

The model should ideally be trained to detect:
* `person` / `rider`
* `motorcycle` / `motorbike` / `bike`

If `best.pt` is not provided, the application automatically attempts to use standard pretrained `yolov8n.pt` weights or enters **SIMULATION / DEMO MODE** with synthetic bounding boxes so you can immediately test the interface.

---

### Step 3: Install Python Dependencies

Open Command Prompt or Terminal, navigate to the project directory, and run:

```bash
pip install -r requirements.txt
```

#### Dependencies in `requirements.txt`:
* `flask`
* `ultralytics`
* `opencv-python`
* `torch`
* `torchvision`
* `numpy`
* `Pillow`

---

## 🚀 How to Run the Website Platform

Launch the Flask server:

```bash
python app.py
```

Then open your browser and navigate to:

```text
http://127.0.0.1:5000
```

### Dashboard Features:
* **Drag-and-Drop Uploader**: Upload any image or video file.
* **Instant Test Samples**: One-click triggers for 1-person (SAFE), 2-person (SAFE), and 3-person (THREAT) test cases.
* **AI Vision Scanner Screen**: Animated laser line scanner showing real-time spatial analysis.
* **Annotated Visual Display**: Neon green/red bounding boxes with confidence scores.
* **Result Information Panel**: Comprehensive breakdown of vehicles, riders, threat status, and violation flags.
* **Live Statistics Dashboard**: Tracks total checked, safe count, violations, threat alerts, and overall riders.
* **Recent Detection History Table**: Real-time log displaying recent detections with preview thumbnails.

---

## 💻 Standalone CLI Inference Commands

### 1. Single Image Detection

```bash
python detect.py --image path/to/bike_image.jpg
```

Processes the image, draws bounding boxes, prints JSON detection summary, and saves annotated output to `static/uploads/result_bike_image.jpg`.

### 2. Video Detection

```bash
python detect.py --video path/to/bike_video.mp4
```

Processes video frame-by-frame, annotates riders and threat banners, and saves processed video to `static/uploads/result_bike_video.mp4`.

### 3. Live Webcam Detection

```bash
python detect.py --webcam
```

Opens your computer's webcam feed, performs real-time detection, counts riders per motorcycle, and displays live overlay warning banners.  
**Press 'q' or 'ESC' to exit webcam window.**

---

## 🧠 Bounding Box Spatial Association Logic

The system does **NOT** simply count every person visible in the entire image.

Instead, it associates pedestrians and riders with individual motorcycles using bounding-box spatial geometry:

```text
Image Matrix
     ↓
Detect Motorcycle Bounding Boxes [bx1, by1, bx2, by2]
     ↓
Detect Person Bounding Boxes [px1, py1, px2, py2]
     ↓
Expand Motorcycle BBox Vertically (Rider Seating Zone)
     ↓
Check Spatial Overlap / Containment of Persons in Rider Zone
     ↓
Assign Persons to the Associated Motorcycle
     ↓
Count Riders per Bike
     ↓
If Riders >= 3 → ⚠ THREAT DETECTED (HIGH RISK, VIOLATION: YES)
Else           → ✓ SAFE (LOW RISK, VIOLATION: NO)
```

---

## 🌐 Flask REST API Specifications

### 1. `POST /api/detect` (Image Detection)
* **Form Field**: `file` (Image file)
* **Sample Response**:
```json
{
    "status": "THREAT DETECTED",
    "risk_level": "HIGH",
    "violation": true,
    "vehicles_detected": 1,
    "persons_detected": 3,
    "violating_bikes_count": 1,
    "max_persons_on_single_bike": 3,
    "confidence": 0.96,
    "annotated_image_url": "/static/uploads/res_image.jpg",
    "message": "Three persons detected on one bike (1 vehicle(s))"
}
```

### 2. `POST /api/sample/<sample_name>`
* **Params**: `1person`, `2persons`, `3persons`
* Runs instant detection on synthetic test images.

### 3. `GET /api/stats`
* Returns current dashboard counters and recent history records.

---

## ⚙ Model Class Customization

In `detect.py`, you can easily adjust class names if your custom model uses different labels:

```python
PERSON_CLASSES = ["person", "rider", "passenger"]
BIKE_CLASSES = ["motorcycle", "motorbike", "bike", "scooter"]
```

---

## 📌 Disclaimer & Project Note

> This project demonstrates an AI-based traffic violation detection concept suitable for academic or prototype demonstrations. The production accuracy of the system depends on the quality of the trained model, dataset, camera angle, lighting, object detection performance, and person-to-bike association algorithm.
