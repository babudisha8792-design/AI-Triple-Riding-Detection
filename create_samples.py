"""
Utility script to generate synthetic test images for 1-rider (SAFE), 2-riders (SAFE),
and 3-riders (THREAT DETECTED) to enable instant testing in the web dashboard.
"""

import os
import cv2
import numpy as np

def generate_sample_images():
    samples_dir = os.path.join("static", "samples")
    os.makedirs(samples_dir, exist_ok=True)

    test_cases = [
        {"filename": "sample_1person.jpg", "riders": 1, "label": "1 Person on Bike (SAFE)"},
        {"filename": "sample_2persons.jpg", "riders": 2, "label": "2 Persons on Bike (SAFE)"},
        {"filename": "sample_3persons.jpg", "riders": 3, "label": "3 Persons on Bike (THREAT)"},
    ]

    for tc in test_cases:
        # Create dark road background canvas (800x500)
        img = np.zeros((500, 800, 3), dtype=np.uint8)
        
        # Draw road and background gradient
        img[:320] = [40, 35, 30] # Sky / Background
        img[320:] = [70, 70, 70] # Asphalt road
        
        # Road lane line
        cv2.line(img, (0, 420), (800, 420), (255, 255, 255), 4)

        # Draw Motorcycle silhouette (around center)
        b_x1, b_y1, b_x2, b_y2 = 250, 240, 550, 420
        # Wheels
        cv2.circle(img, (290, 400), 45, (20, 20, 20), -1)
        cv2.circle(img, (290, 400), 45, (150, 150, 150), 6)
        cv2.circle(img, (510, 400), 45, (20, 20, 20), -1)
        cv2.circle(img, (510, 400), 45, (150, 150, 150), 6)
        # Bike body frame
        cv2.rectangle(img, (310, 310), (490, 380), (30, 90, 200), -1) # Blue motorcycle body
        cv2.line(img, (470, 320), (510, 270), (200, 200, 200), 8)   # Handlebars

        # Draw riders on top of motorcycle
        rider_count = tc['riders']
        rider_colors = [(0, 180, 255), (0, 220, 120), (220, 100, 255)] # Unique rider jacket colors
        
        spacing = 65 if rider_count > 1 else 0
        start_x = 340 - int((rider_count - 1) * spacing / 2.0)

        for i in range(rider_count):
            rx = start_x + i * spacing
            ry = 180 + (i % 2) * 10
            
            # Head / Helmet
            cv2.circle(img, (rx + 20, ry), 22, (50, 50, 50), -1)
            cv2.circle(img, (rx + 20, ry), 20, (230, 230, 230), -1)
            cv2.rectangle(img, (rx + 10, ry - 5), (rx + 30, ry + 5), (20, 20, 20), -1) # Visor

            # Body / Jacket
            cv2.rectangle(img, (rx, ry + 22), (rx + 40, ry + 120), rider_colors[i % 3], -1)

        # Title text overlay on image
        cv2.putText(img, f"DEMO TEST IMAGE: {tc['label']}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(img, "PyTorch/YOLO Bike Violation AI Benchmark", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

        filepath = os.path.join(samples_dir, tc['filename'])
        cv2.imwrite(filepath, img)
        print(f"[INFO] Generated synthetic sample image: {filepath}")

if __name__ == "__main__":
    generate_sample_images()
