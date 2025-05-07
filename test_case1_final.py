import requests
import time
import cv2
import numpy as np
from ultralytics import YOLO

# === ESP32 IP Setup ===
esp_ip = "192.168.244.61"
esp_capture_url = f"http://{esp_ip}/capture"
esp_command_url = f"http://{esp_ip}"
esp_buzzer_url = f"http://{esp_ip}/buzz"

# === YOLO Model Setup ===
model = YOLO("runs/detect/traffic_light_model2/weights/best.pt")
names = model.names

# === States ===
vehicle_state = "idle"
last_signal_stopped = None
last_signal_forced = None

# === Helpers ===
def get_signal_status(labels):
    if "red" in labels:
        return "RED"
    elif "green" in labels or "yellow" in labels:
        return "GREEN"
    else:
        return "NONE"

def capture_frame():
    try:
        response = requests.get(esp_capture_url, timeout=2)
        img_array = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except:
        print("❌ Failed to capture frame")
        return None

def trigger_buzzer(duration):
    try:
        res = requests.get(f"{esp_buzzer_url}?time={duration}", timeout=2)
        print(f"🔔 Buzzer Triggered for {duration}s → {res.text}")
    except:
        print("❌ Failed to trigger buzzer")

def monitor_signal(state_type):
    print(f"📸 Monitoring signal ({state_type.upper()})...")

    while True:
        frame = capture_frame()
        if frame is None:
            continue

        results = model(frame, verbose=False)
        labels = [names[int(b.cls[0])] for b in results[0].boxes]
        current_status = get_signal_status(labels)

        # Show frame
        annotated = results[0].plot()
        cv2.imshow(f"{state_type.upper()} - Detecting...", annotated)

        # If signal detected
        if current_status != "NONE":
            print(f"✅ Detected: {current_status}")
            cv2.destroyAllWindows()
            return current_status

        # Auto-close on ESC or 'q' (optional)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            print("🛑 Manually closed preview")
            cv2.destroyAllWindows()
            return "NONE"


# === Main Loop ===
while True:
    cmd = input("Enter command (f: forward, b: backward, s: stop, r: forced, d: exit): ").strip().lower()

    if cmd == 'd':
        print("👋 Exiting...")
        break

    # Send motor command
    if cmd in ['f', 'b', 's']:
        try:
            requests.get(f"{esp_command_url}/{cmd.upper()}", timeout=2)
        except:
            print("⚠️ Failed to send motor command")

    # Update state
    if cmd in ['f', 'b']:
        vehicle_state = "running"
    elif cmd == 's':
        vehicle_state = "stopped"
    elif cmd == 'r':
        vehicle_state = "forced"
    else:
        vehicle_state = "idle"

    print(f"🚗 Vehicle State: {vehicle_state}")

    # Stopped State
    if vehicle_state == "stopped" and last_signal_stopped is None:
        last_signal_stopped = monitor_signal("stopped")
        print(f"🛑 STOPPED → Signal: {last_signal_stopped}")
        if last_signal_stopped == "RED":
            trigger_buzzer(5)
        elif last_signal_stopped == "GREEN":
            trigger_buzzer(2)

    # Forced State
    elif vehicle_state == "forced" and last_signal_forced is None:
        last_signal_forced = monitor_signal("forced")
        print(f"🚨 FORCED → Signal: {last_signal_forced}")
        if last_signal_forced == "RED":
            trigger_buzzer(100)
        else:
            print("✅ No Violation")

    # Reset on state change
    if vehicle_state != "stopped":
        last_signal_stopped = None
    if vehicle_state != "forced":
        last_signal_forced = None
