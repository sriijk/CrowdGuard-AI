import cv2

# ------------------ PERSON DETECTION ------------------
def detect_people(model, frame, conf=0.5):
    """
    Detect people using YOLO model and return bounding boxes.

    Args:
        model: YOLO model object.
        frame: Image frame (BGR).
        conf (float): Confidence threshold (default 0.5).

    Returns:
        List of tuples: (x1, y1, x2, y2, confidence)
    """
    results = model.predict(source=frame, conf=conf, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # Class 0 = person (COCO)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                detections.append((x1, y1, x2, y2, confidence))

    return detections

# ------------------ ZONE GRID UTILITIES ------------------
def get_zone_id(x, y, frame_w, frame_h, rows, cols):
    """
    Get zone ID based on grid coordinates.

    Args:
        x, y: Center point of person.
        frame_w, frame_h: Frame dimensions.
        rows, cols: Grid size.

    Returns:
        Integer zone ID.
    """
    zone_w = frame_w // cols
    zone_h = frame_h // rows
    col = int(x / zone_w)
    row = int(y / zone_h)
    return row * cols + col

def draw_zone_grid(frame, rows, cols):
    """
    Draw a grid overlay on the frame.

    Args:
        frame: Image frame.
        rows, cols: Number of rows and columns in the grid.
    """
    h, w = frame.shape[:2]
    for i in range(1, rows):
        y = i * h // rows
        cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1)
    for j in range(1, cols):
        x = j * w // cols
        cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1)
