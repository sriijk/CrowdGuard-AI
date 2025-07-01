from ultralytics import YOLO
import cv2
import numpy as np
from app_utils.detection import detect_people
from app_utils.zone_analysis import get_zone_id, draw_zone_grid

# Load YOLOv5n model
model = YOLO("model/yolov8n.pt")  # make sure yolov5n.pt is in the 'model' folder

# Initialize webcam
cap = cv2.VideoCapture(0)

# Grid config
GRID_ROWS, GRID_COLS = 3, 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    frame_h, frame_w = frame.shape[:2]

    # Detect people
    detections = detect_people(model, frame)

    # Zone counts
    zone_counts = [0] * (GRID_ROWS * GRID_COLS)

    # Draw grid
    draw_zone_grid(frame, GRID_ROWS, GRID_COLS)

    for det in detections:
        x1, y1, x2, y2, conf = det
        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
        zone_id = get_zone_id(xc, yc, frame_w, frame_h, GRID_ROWS, GRID_COLS)
        zone_counts[zone_id] += 1

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (xc, yc), 3, (0, 0, 255), -1)

    # Display counts
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            zone_id = i * GRID_COLS + j
            x = j * frame_w // GRID_COLS + 5
            y = i * frame_h // GRID_ROWS + 20
            cv2.putText(frame, f'{zone_counts[zone_id]}', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Crowd Monitor", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
