import time
import cv2
import os
from playsound import playsound

ALERT_SOUND = "alert.mp3"  # Add any free alert sound file to root directory
LOG_FILE = "results/detections_log.txt"
FRAME_SAVE_DIR = "results/detected_frames/"

os.makedirs(FRAME_SAVE_DIR, exist_ok=True)

def trigger_local_alert(frame, class_name, confidence):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] ALERT: Detected {class_name} ({confidence:.2f})"
    print(log_msg)

    # Save to log
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

    # Save frame as evidence
    frame_path = os.path.join(FRAME_SAVE_DIR, f"{timestamp.replace(':', '-')}.jpg")
    cv2.imwrite(frame_path, frame)

    # Play alert sound
    playsound(ALERT_SOUND)
