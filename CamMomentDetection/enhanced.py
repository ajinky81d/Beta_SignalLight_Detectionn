import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Warning counter
warnings = 0
max_warnings = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate magnitude of movement
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_movement = np.mean(magnitude)
    
    # Detect camera adjustment
    if avg_movement > 2:  # Adjust sensitivity if needed
        warnings += 1
        print(f"Warning {warnings}/3: Camera movement detected! Adjusting the frame.")
    
    # Show movement status on screen
    status = "Stable" if avg_movement < 2 else f"Moving (Warning {warnings}/3)"
    cv2.putText(frame, f"Status: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Camera Movement Detection", frame)
    
    prev_gray = gray

    # If warnings reach 3, close the camera
    if warnings >= max_warnings:
        print("Camera Closed: Excessive movement detected (Someone adjusting the frame).")
        break

    # Press 'q' to exit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Camera Closed: User manually exited.")
        break

cap.release()
cv2.destroyAllWindows()
