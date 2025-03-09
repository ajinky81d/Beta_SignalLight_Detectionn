import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

# Convert first frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Get frame size
height, width = prev_gray.shape[:2]

# Define frame boundary points (corners)
prev_pts = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=np.float32)

warnings = 0
max_warnings = 3
angle_threshold = 15  # Degree threshold for camera movement

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow to track boundary points
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

    if new_pts is not None and status is not None and all(status):
        # Compute transformation matrix
        M, _ = cv2.findHomography(prev_pts, new_pts, cv2.RANSAC)

        if M is not None:
            # Extract rotation angle (camera tilt)
            angle_x = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)

            # Convert float points to integer tuples
            pt1, pt2, pt3, pt4 = [tuple(map(int, p.ravel())) for p in new_pts]

            # Draw only boundary lines (no cross lines)
            cv2.line(frame, pt1, pt2, (0, 0, 255), 4)  # Top border
            cv2.line(frame, pt1, pt3, (0, 0, 255), 4)  # Left border
            cv2.line(frame, pt2, pt4, (0, 0, 255), 4)  # Right border
            cv2.line(frame, pt3, pt4, (0, 0, 255), 4)  # Bottom border

            # Detect excessive frame movement (camera adjustment)
            if abs(angle_x) > angle_threshold:
                warnings += 1
                print(f"Warning {warnings}/3: Camera adjusted more than {angle_threshold}°!")

    # Display warning count on screen
    status_text = f"Status: {'Stable' if warnings < 3 else 'Camera Adjusted!'}"
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Camera Movement Detection", frame)

    # Close camera after 3 warnings
    if warnings >= max_warnings:
        print("Camera Closed: Excessive frame movement detected!")
        break

    # Press 'q' to exit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Camera Closed: User manually exited.")
        break

    # Update previous frame
    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()








# import cv2
# import numpy as np

# # Open webcam
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Read the first frame
# ret, prev_frame = cap.read()
# if not ret:
#     print("Error: Could not read frame.")
#     cap.release()
#     exit()

# # Convert to grayscale
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# # Define initial boundary points (corners of the frame)
# height, width = prev_frame.shape[:2]
# prev_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

# warnings = 0
# max_warnings = 3
# movement_threshold = 15  # Threshold changed from 20° to 15°

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert current frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Track frame corner movement using Optical Flow
#     new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

#     if new_pts is not None and status is not None:
#         # Compute transformation matrix (Homography)
#         M, _ = cv2.findHomography(prev_pts, new_pts, cv2.RANSAC)

#         if M is not None:
#             # Extract rotation angle
#             angle_x = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)

#             # Convert float points to integer tuples
#             pt1 = tuple(map(int, new_pts[0].ravel()))
#             pt2 = tuple(map(int, new_pts[1].ravel()))
#             pt3 = tuple(map(int, new_pts[2].ravel()))
#             pt4 = tuple(map(int, new_pts[3].ravel()))

#             # Draw a **RED** boundary outline (without cross lines)
#             cv2.line(frame, pt1, pt2, (0, 0, 255), 4)  # Top border
#             cv2.line(frame, pt1, pt3, (0, 0, 255), 4)  # Left border
#             cv2.line(frame, pt2, pt4, (0, 0, 255), 4)  # Right border
#             cv2.line(frame, pt3, pt4, (0, 0, 255), 4)  # Bottom border

#             # Detect excessive frame movement
#             if abs(angle_x) > movement_threshold:
#                 warnings += 1
#                 print(f"Warning {warnings}/3: Camera tilted more than {movement_threshold}°!")

#     # Show warning count and status
#     status_text = f"Status: {'Stable' if warnings < 3 else 'Camera Adjusted!'}"
#     cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

#     cv2.imshow("Camera Frame Tilt Detection", frame)

#     # If warnings reach max, close the camera
#     if warnings >= max_warnings:
#         print("Camera Closed: Excessive frame movement detected!")
#         break

#     # Press 'q' to exit manually
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Camera Closed: User manually exited.")
#         break

#     # Update previous frame
#     prev_gray = gray.copy()

# cap.release()
# cv2.destroyAllWindows()














# import cv2
# import numpy as np

# # Open webcam
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Read the first frame
# ret, prev_frame = cap.read()
# if not ret:
#     print("Error: Could not read frame.")
#     cap.release()
#     exit()

# # Convert to grayscale
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# # Define frame boundary points (corners)
# height, width = prev_frame.shape[:2]
# prev_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

# warnings = 0
# max_warnings = 3

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Track the movement of frame corner points
#     new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

#     if new_pts is not None and status is not None:
#         # Compute transformation matrix (Homography)
#         M, _ = cv2.findHomography(prev_pts, new_pts, cv2.RANSAC)

#         if M is not None:
#             # Extract rotation angle
#             angle_x = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)

#             # Draw frame boundary (RED outline)
#             cv2.polylines(frame, [np.int32(new_pts)], isClosed=True, color=(0, 0, 255), thickness=3)

#             # Detect excessive frame movement
#             if abs(angle_x) > 20:
#                 warnings += 1
#                 print(f"Warning {warnings}/3: Camera tilted more than 20°!")

#     # Show warning count and status
#     status_text = f"Status: {'Stable' if warnings < 3 else 'Camera Adjusted!'}"
#     cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

#     cv2.imshow("Camera Frame Tilt Detection", frame)

#     # If warnings reach max, close the camera
#     if warnings >= max_warnings:
#         print("Camera Closed: Excessive frame movement detected!")
#         break

#     # Press 'q' to exit manually
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Camera Closed: User manually exited.")
#         break

#     # Update previous frame
#     prev_gray = gray.copy()

# cap.release()
# cv2.destroyAllWindows()



