import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened
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
    
    # Display movement status
    status = "Stable" if avg_movement < 2 else "Moving"
    cv2.putText(frame, f"Status: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Camera Movement Detection", frame)
    
    prev_gray = gray

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
