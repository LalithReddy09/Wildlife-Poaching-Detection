import cv2
from ultralytics import YOLO
from alert import trigger_local_alert  # from your alert.py

# Load YOLO model
model = YOLO("models/best.pt")  # or yolov8n.pt if you're starting fresh

# Load video
cap = cv2.VideoCapture("data/sample_video.mp4")

frame_rate = 30  # Check every 30 frames
frame_count = 0

print("🔍 Starting wildlife detection... Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_rate != 0:
        continue

    results = model(frame)[0]  # YOLOv8 inference

    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        print(f"Detected: {class_name} ({confidence:.2f})")

        if class_name.lower() in ["gun", "weapon"] and confidence > 0.5:
            trigger_local_alert(frame, class_name, confidence)

    annotated_frame = results.plot()
    cv2.imshow("Wildlife AI Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
