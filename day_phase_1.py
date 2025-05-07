import cv2
import numpy as np
import requests
from ultralytics import YOLO
from datetime import datetime
import os
import time

# === ESP32-CAM Frame Capture URL ===
esp_url = "http://192.168.198.242/stream"

# === Load YOLO model ===
model = YOLO("runs/detect/traffic_light_model2/weights/best.pt")
names = model.names

# === Setup Variables ===
recording = False
frame_buffer = []
displacement = 4
prev_status = "NONE"
transition_to_none_from_red = False
frame_count = 0
fps = 10  # approximate processing frame rate
frame_width, frame_height = 128, 128  # ESP32 default QVGA resolution

# === Create violation output folder ===
os.makedirs("violations", exist_ok=True)

# === Optimize: Reuse session ===
session = requests.Session()

def get_signal_status(labels):
    if "red" in labels:
        return "RED"
    elif "green" in labels or "yellow" in labels:
        return "GREENYELLOW"
    else:
        return "NONE"

def save_violation_clip(buffer):
    if not buffer:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = f"violations/violation_{timestamp}.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    for f in buffer:
        out.write(f)
    out.release()
    print(f"📹 Violation saved: {out_path}")

# === Main Loop ===
while True:
    try:
        # === Faster: Reuse session to capture frame ===
        response = session.get(esp_url, timeout=4)
        img_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frame_count += 1
    except Exception as e:
        print(f"⚠️ Failed to get image from ESP32: {e}")
        time.sleep(0.2)  # slight wait before retry
        continue

    # === Run detection ===
    results = model(frame, verbose=False)
    labels = [names[int(b.cls[0])] for b in results[0].boxes]
    current_status = get_signal_status(labels)

    print(f"\n🟡 Frame {frame_count}: Detected Signal → {current_status}")

    # === Transitions ===
    if prev_status != current_status:
        print(f"🔄 Transition: {prev_status} → {current_status}")
        if prev_status in ["GREENYELLOW", "GREEN"] and current_status == "RED":
            print("🚨 RED light appeared after GREEN/YELLOW")
        elif prev_status == "RED" and current_status == "NONE":
            print("🛑 RED transitioned to NONE, checking violation...")
            transition_to_none_from_red = True
        elif prev_status == "RED" and current_status == "GREEN":
            print("✅ RED → GREEN — discard recording")
            recording = False
            frame_buffer = []
        elif prev_status == "GREENYELLOW" and current_status == "NONE":
            print("⚠️ No signal detected after GREENYELLOW")
    prev_status = current_status

    # === Handle RED
    if current_status == "RED":
        if not recording:
            print("🔴 START recording due to RED light.")
            recording = True
            frame_buffer = []
        displacement = 4
        print(f"📏 Displacement: {displacement:.1f} meters")
    else:
        if not transition_to_none_from_red:
            displacement = 4

    # === Record if RED
    if recording:
        frame_buffer.append(frame)
        print("💾 Frame added to buffer")

    # === Check for Violation
    if transition_to_none_from_red:
        print(f"🔎 Checking displacement: {displacement:.1f} m")
        if displacement > 3:
            print("🚨 Violation Detected!")
            save_violation_clip(frame_buffer)
        else:
            print("✅ No violation, recording discarded.")
        transition_to_none_from_red = False
        recording = False
        frame_buffer = []
        displacement = 4

    # === Show Annotated Frame ===
    annotated_frame = results[0].plot()
    display_text = f"Signal: {current_status} | Displacement: {displacement:.1f} m"
    cv2.putText(annotated_frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("ESP32-CAM Traffic Light Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quit requested.")
        break

    # === Throttle a bit to reduce network load & CPU usage ===
    time.sleep(0.05)

cv2.destroyAllWindows()
print("✅ Processing complete.")
