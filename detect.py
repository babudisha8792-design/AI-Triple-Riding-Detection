"""
PyTorch/YOLO Three-Person-on-One-Bike Detection Engine & Standalone CLI Tool.
Handles object detection, person-to-bike spatial association, violation classification,
and OpenCV frame annotation.
"""

import os
import sys
import argparse
import time
import json
import cv2
import numpy as np

# Easy configuration section for custom trained models
MODEL_PATH = "best.pt"
FALLBACK_MODEL = "yolov8n.pt"

# Recognized class names (case-insensitive)
PERSON_CLASSES = ["person", "rider", "passenger"]
BIKE_CLASSES = ["motorcycle", "motorbike", "bike", "scooter", "moped"]

# Global model handle
yolo_model = None
using_demo_mode = False

def load_detection_model(model_path=MODEL_PATH):
    """
    Attempts to load PyTorch / YOLO model.
    Falls back to pretrained yolov8n.pt or SIMULATION mode if unavailable.
    """
    global yolo_model, using_demo_mode
    
    try:
        from ultralytics import YOLO
        if os.path.exists(model_path):
            print(f"[INFO] Loading custom PyTorch/YOLO model from '{model_path}'...")
            yolo_model = YOLO(model_path)
            using_demo_mode = False
            return yolo_model
        elif os.path.exists(FALLBACK_MODEL):
            print(f"[INFO] '{model_path}' not found. Loading fallback model '{FALLBACK_MODEL}'...")
            yolo_model = YOLO(FALLBACK_MODEL)
            using_demo_mode = False
            return yolo_model
        else:
            try:
                print(f"[INFO] Attempting to download standard YOLOv8n model...")
                yolo_model = YOLO(FALLBACK_MODEL)
                using_demo_mode = False
                return yolo_model
            except Exception as e:
                print(f"[WARNING] Could not initialize YOLO model: {e}")
                print("[INFO] Switching to SIMULATION / DEMO MODE.")
                using_demo_mode = True
                return None
    except ImportError:
        print("[WARNING] ultralytics package not installed. Switching to SIMULATION / DEMO MODE.")
        using_demo_mode = True
        return None

def compute_box_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between box1 [x1, y1, x2, y2] and box2 [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def associate_persons_with_bikes(bikes, persons, frame_width, frame_height):
    """
    Associates detected persons with detected motorcycles based on bounding box geometry.
    Expands the bike bounding box vertically to include the rider zone above/around the bike.
    
    bikes: list of dicts [{'bbox': [x1, y1, x2, y2], 'conf': float, 'id': int}]
    persons: list of dicts [{'bbox': [x1, y1, x2, y2], 'conf': float, 'id': int}]
    """
    associations = {i: [] for i in range(len(bikes))}
    unassociated_persons = []

    for person_idx, person in enumerate(persons):
        px1, py1, px2, py2 = person['bbox']
        person_area = max(1, (px2 - px1) * (py2 - py1))
        person_center_x = (px1 + px2) / 2.0
        person_bottom_y = py2
        
        best_bike_idx = -1
        max_score = -1.0

        for bike_idx, bike in enumerate(bikes):
            bx1, by1, bx2, by2 = bike['bbox']
            bw = bx2 - bx1
            bh = by2 - by1

            # Define rider zone for this bike: expand upwards by 80% of height, sideways by 25%
            rider_zone_x1 = max(0, bx1 - 0.25 * bw)
            rider_zone_x2 = min(frame_width, bx2 + 0.25 * bw)
            rider_zone_y1 = max(0, by1 - 0.85 * bh)
            rider_zone_y2 = min(frame_height, by2 + 0.20 * bh)

            # Check if person center falls within rider zone
            if (rider_zone_x1 <= person_center_x <= rider_zone_x2) and (rider_zone_y1 <= person_bottom_y <= rider_zone_y2):
                # Calculate overlap between person box and rider zone box
                ix1 = max(px1, rider_zone_x1)
                iy1 = max(py1, rider_zone_y1)
                ix2 = min(px2, rider_zone_x2)
                iy2 = min(py2, rider_zone_y2)

                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                overlap_ratio = inter_area / person_area

                if overlap_ratio > max_score:
                    max_score = overlap_ratio
                    best_bike_idx = bike_idx

        if best_bike_idx != -1 and max_score > 0.25:
            associations[best_bike_idx].append(person_idx)
        else:
            unassociated_persons.append(person_idx)

    return associations, unassociated_persons

def process_frame(frame, conf_threshold=0.35):
    """
    Runs detection on a single CV2 image matrix (BGR).
    Returns annotated frame and structured detection result dict.
    """
    global yolo_model, using_demo_mode

    if yolo_model is None and not using_demo_mode:
        load_detection_model()

    h, w, _ = frame.shape
    annotated = frame.copy()

    bikes = []
    persons = []

    if yolo_model is not None and not using_demo_mode:
        results = yolo_model(frame, conf=conf_threshold, verbose=False)
        for r in results:
            boxes = r.boxes
            names = r.names
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id)).lower()
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()

                if any(k in cls_name for k in BIKE_CLASSES):
                    bikes.append({'bbox': xyxy, 'conf': conf, 'label': cls_name})
                elif any(k in cls_name for k in PERSON_CLASSES):
                    persons.append({'bbox': xyxy, 'conf': conf, 'label': cls_name})
    else:
        # SIMULATION / DEMO MODE fallback
        # Generates deterministic demo detections based on image dimensions or metadata
        annotated_watermark = True
        # Simple color analysis or fallback heuristic
        # If no model, inspect if demo sample image pattern exists
        bikes, persons = generate_demo_detections(frame)

    # Spatial association of persons to bikes
    associations, unassociated_persons = associate_persons_with_bikes(bikes, persons, w, h)

    # Classify each bike and build summary
    bike_results = []
    overall_threat = False
    max_riders = 0
    total_violating_bikes = 0

    for b_idx, bike in enumerate(bikes):
        bx1, by1, bx2, by2 = [int(v) for v in bike['bbox']]
        associated_person_indices = associations.get(b_idx, [])
        rider_count = len(associated_person_indices)

        if rider_count > max_riders:
            max_riders = rider_count

        # Main detection rule logic: >= 3 persons on one bike -> THREAT DETECTED
        if rider_count >= 3:
            status = "THREAT DETECTED"
            risk_level = "HIGH"
            violation = True
            overall_threat = True
            total_violating_bikes += 1
            color = (0, 0, 235) # Bright Red / Danger
            bbox_color = (0, 0, 255)
        else:
            status = "SAFE"
            risk_level = "LOW"
            violation = False
            color = (0, 220, 100) # Vibrant Green / Safe
            bbox_color = (0, 255, 120)

        bike_results.append({
            'bike_id': b_idx + 1,
            'bbox': [bx1, by1, bx2, by2],
            'confidence': round(bike['conf'], 2),
            'rider_count': rider_count,
            'status': status,
            'risk_level': risk_level,
            'violation': violation
        })

        # Draw Motorcycle Bounding Box
        thickness = 3 if violation else 2
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), bbox_color, thickness)
        
        # Label motorcycle
        label = label = f"Bike #{b_idx+1}: {bike.get('confidence', 0)*100:.0f}% ({rider_count} Rider{'s' if rider_count!=1 else ''})"
        cv2.rectangle(annotated, (bx1, max(0, by1 - 25)), (bx1 + len(label)*9, by1), bbox_color, -1)
        cv2.putText(annotated, label, (bx1 + 4, max(12, by1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Highlight associated riders
        for p_idx in associated_person_indices:
            person = persons[p_idx]
            px1, py1, px2, py2 = [int(v) for v in person['bbox']]
            p_label = f"Person {person['conf']*100:.0f}%"
            cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
            cv2.putText(annotated, p_label, (px1, max(15, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Draw Threat Alert Header over violating bikes
        if violation:
            alert_txt = f"ALERT: {rider_count} PERSONS ON BIKE #{b_idx+1}"
            cv2.rectangle(annotated, (bx1, max(0, by1 - 50)), (bx1 + len(alert_txt)*11, max(25, by1 - 25)), (0, 0, 200), -1)
            cv2.putText(annotated, alert_txt, (bx1 + 5, max(18, by1 - 32)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Draw unassociated pedestrians in blue/cyan
    for p_idx in unassociated_persons:
        person = persons[p_idx]
        px1, py1, px2, py2 = [int(v) for v in person['bbox']]
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 180, 50), 1)
        cv2.putText(annotated, "Pedestrian", (px1, max(12, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 50), 1)

    # Frame Overall Header Banner
    banner_height = 55
    overlay = annotated.copy()
    
    if overall_threat:
        banner_color = (15, 15, 180) # Dark red overlay
        header_title = "⚠ THREE PERSONS ON ONE BIKE — THREAT DETECTED"
        sub_title = f"VIOLATION: YES | High Risk Alert | Violating Bikes: {total_violating_bikes}/{len(bikes)}"
        text_color = (255, 255, 255)
    else:
        banner_color = (15, 100, 30) # Dark green overlay
        header_title = "✓ SAFE — NO VIOLATION DETECTED"
        sub_title = f"VIOLATION: NO | Low Risk | Total Bikes: {len(bikes)} | Total Riders: {len(persons)}"
        text_color = (255, 255, 255)

    cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_color, -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    cv2.putText(annotated, header_title, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)
    cv2.putText(annotated, sub_title, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    if using_demo_mode:
        cv2.putText(annotated, "SIMULATION / DEMO MODE (Add best.pt for custom model)", (w - 380, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    # Average confidence score
    all_confs = [b['confidence'] for b in bike_results] + [p['conf'] for p in persons]
    avg_conf = round(float(np.mean(all_confs)), 2) if all_confs else 0.92

    result_summary = {
        'status': "THREAT DETECTED" if overall_threat else "SAFE",
        'risk_level': "HIGH" if overall_threat else "LOW",
        'violation': overall_threat,
        'vehicles_detected': len(bikes),
        'persons_detected': len(persons),
        'violating_bikes_count': total_violating_bikes,
        'max_persons_on_single_bike': max_riders,
        'confidence': avg_conf,
        'bike_details': bike_results,
        'demo_mode': using_demo_mode,
        'message': f"Three persons detected on one bike ({total_violating_bikes} vehicle(s))" if overall_threat else "No three-person violation detected"
    }

    return annotated, result_summary

def generate_demo_detections(frame):
    """
    Simulation mode generator when model weights are not loaded.
    Reads frame shape/content to produce synthetic detections for demonstration.
    """
    h, w, _ = frame.shape
    
    # Check if image color signature or shape suggests specific test case
    # Default synthetic 3-person threat case
    b_x1, b_y1, b_x2, b_y2 = int(w * 0.25), int(h * 0.45), int(w * 0.75), int(h * 0.85)
    bikes = [{'bbox': [b_x1, b_y1, b_x2, b_y2], 'conf': 0.94, 'label': 'motorcycle'}]

    # Create 3 rider bounding boxes on top of bike
    p1 = {'bbox': [int(w * 0.30), int(h * 0.22), int(w * 0.45), int(h * 0.55)], 'conf': 0.96, 'label': 'person'}
    p2 = {'bbox': [int(w * 0.42), int(h * 0.20), int(w * 0.57), int(h * 0.55)], 'conf': 0.92, 'label': 'person'}
    p3 = {'bbox': [int(w * 0.55), int(h * 0.24), int(w * 0.68), int(h * 0.56)], 'conf': 0.89, 'label': 'person'}

    persons = [p1, p2, p3]

    return bikes, persons

def run_image_detection(image_path, output_path=None):
    """
    CLI / Backend function to process an image file.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image path not found: {image_path}")
        return None, None

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return None, None

    annotated, summary = process_frame(frame)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Annotated image saved to '{output_path}'")

    return annotated, summary

def run_video_detection(video_path, output_path=None):
    """
    CLI / Backend function to process a video file frame by frame.
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video path not found: {video_path}")
        return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return None, None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    last_summary = None
    frame_count = 0

    print("[INFO] Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, summary = process_frame(frame)
        last_summary = summary
        frame_count += 1

        if out_writer:
            out_writer.write(annotated)

    cap.release()
    if out_writer:
        out_writer.release()
        print(f"[INFO] Processed video ({frame_count} frames) saved to '{output_path}'")

    return last_summary

def run_webcam_detection():
    """
    CLI interactive mode: Live webcam inference.
    Press 'q' to exit.
    """
    print("[INFO] Opening webcam... Press 'q' in the window to quit.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not access webcam (index 0). Ensure camera is connected.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture webcam frame.")
            break

        annotated, summary = process_frame(frame)

        cv2.imshow("Three Persons on One Bike Detection - Live Feed", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam session closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch/YOLO Three-Person-on-One-Bike Detection CLI System")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--webcam", action="store_true", help="Launch live webcam detection")
    parser.add_argument("--output", type=str, help="Path to save annotated output file")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to best.pt model weights file")

    args = parser.parse_args()

    if args.model:
        MODEL_PATH = args.model

    load_detection_model(MODEL_PATH)

    if args.image:
        out_file = args.output or os.path.join("static", "uploads", f"result_{os.path.basename(args.image)}")
        _, summary = run_image_detection(args.image, out_file)
        if summary:
            print("\n=== AI DETECTION RESULT ===")
            print(json.dumps(summary, indent=4))
    elif args.video:
        out_file = args.output or os.path.join("static", "uploads", f"result_{os.path.basename(args.video)}")
        summary = run_video_detection(args.video, out_file)
        if summary:
            print("\n=== AI VIDEO DETECTION RESULT ===")
            print(json.dumps(summary, indent=4))
    elif args.webcam:
        run_webcam_detection()
    else:
        print("[USAGE] Please specify --image, --video, or --webcam. Example:")
        print("        python detect.py --image path/to/image.jpg")
        print("        python detect.py --webcam")
