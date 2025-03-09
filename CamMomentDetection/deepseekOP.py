import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read first frame and get dimensions
ret, frame = cap.read()
height, width = frame.shape[:2]

# Define fixed boundary points (proper 2D format)
edge_margin = 20
boundary_points = np.array([
    [edge_margin, edge_margin],          # Top-left
    [width-edge_margin, edge_margin],    # Top-right
    [edge_margin, height-edge_margin],   # Bottom-left
    [width-edge_margin, height-edge_margin], # Bottom-right
    [width//2, edge_margin],             # Top-center
    [width//2, height-edge_margin],      # Bottom-center
    [edge_margin, height//2],            # Left-center
    [width-edge_margin, height//2]       # Right-center
], dtype=np.float32).reshape(-1, 1, 2)  # Correct 3D shape for optical flow

# Detection parameters
warnings = 0
max_warnings = 3
translation_threshold = 15
rotation_threshold = 5
min_boundary_points = 4

# Initialize previous frame
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Optical flow parameters
lk_params = dict(winSize=(25, 25),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Track boundary points
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, boundary_points, None, **lk_params)
    
    movement_detected = False
    if new_points is not None and np.sum(status) >= min_boundary_points:
        # Reshape points for transformation calculation
        valid_old = boundary_points[status==1].reshape(-1, 2)
        valid_new = new_points[status==1].reshape(-1, 2)
        
        M, _ = cv2.estimateAffinePartial2D(valid_old, valid_new)
        
        if M is not None:
            # Calculate movement metrics
            dx = M[0, 2]
            dy = M[1, 2]
            rotation = np.arctan2(M[1, 0], M[0, 0]) * 180/np.pi
            translation = np.sqrt(dx**2 + dy**2)
            
            if translation > translation_threshold or abs(rotation) > rotation_threshold:
                movement_detected = True
                warnings += 1
                print(f"Warning {warnings}/3: Frame moved {translation:.1f}px, rotated {rotation:.1f}°")

    # Visual feedback (fixed point formatting)
    for i in range(len(boundary_points)):
        if status[i]:
            x_old, y_old = boundary_points[i][0].astype(int)
            x_new, y_new = new_points[i][0].astype(int)
            cv2.line(frame, (x_old, y_old), (x_new, y_new), (0, 255, 0), 2)
    
    # Draw boundary rectangle
    cv2.rectangle(frame, 
                 (edge_margin, edge_margin), 
                 (width-edge_margin, height-edge_margin), 
                 (0, 0, 255), 2)

    # Status display
    cv2.putText(frame, f"Active Points: {np.sum(status)}/{len(status)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Warnings: {warnings}/3", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if movement_detected:
        cv2.putText(frame, "FRAME MOVEMENT DETECTED!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Frame Boundary Tracker", frame)

    # Update previous data
    prev_gray = gray.copy()
    boundary_points[status==1] = new_points[status==1]

    if warnings >= max_warnings:
        print("SECURITY ALERT: Camera frame adjusted!")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Initialize webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Read first frame and prepare
# ret, prev_frame = cap.read()
# if not ret:
#     print("Error: Could not read initial frame.")
#     cap.release()
#     exit()

# height, width = prev_frame.shape[:2]
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# # Detect initial good features to track
# feature_params = dict(maxCorners=100,
#                       qualityLevel=0.3,
#                       minDistance=7,
#                       blockSize=7)
# prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# # Create mask for visualization
# mask = np.zeros_like(prev_frame)

# # Detection parameters
# warnings = 0
# max_warnings = 3
# translation_threshold = 10  # pixels
# rotation_threshold = 3      # degrees
# min_points = 15

# # Optical flow parameters
# lk_params = dict(winSize=(25, 25),
#                 maxLevel=3,
#                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Calculate optical flow
#     new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
    
#     # Select good points
#     if new_pts is not None:
#         good_new = new_pts[status == 1]
#         good_old = prev_pts[status == 1]
    
#     movement_detected = False
#     translation = 0
#     rotation = 0
    
#     if len(good_new) >= min_points:
#         # Find transformation matrix
#         M, _ = cv2.estimateAffinePartial2D(good_old, good_new)
        
#         if M is not None:
#             # Calculate movement metrics
#             dx = M[0, 2]
#             dy = M[1, 2]
#             rotation = np.arctan2(M[1, 0], M[0, 0]) * 180/np.pi
#             translation = np.sqrt(dx**2 + dy**2)
            
#             # Check thresholds
#             if translation > translation_threshold or abs(rotation) > rotation_threshold:
#                 movement_detected = True
#                 warnings += 1
#                 print(f"Warning {warnings}/3: Move {translation:.1f}px, Rotate {rotation:.1f}°")

#     # Visualization
#     mask.fill(0)
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
#         frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
#     img = cv2.add(frame, mask)

#     # Status overlay
#     cv2.putText(img, f"Tracked Points: {len(good_new)}/{len(prev_pts)}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     cv2.putText(img, f"Warnings: {warnings}/3", (10, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
#     if movement_detected:
#         cv2.putText(img, "CAMERA MOVEMENT DETECTED!", (50, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#         cv2.putText(img, f"Translation: {translation:.1f}px", (50, 140),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#         cv2.putText(img, f"Rotation: {rotation:.1f}°", (50, 180),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     cv2.imshow("Camera Movement Tracker", img)

#     # Update previous points
#     prev_gray = gray.copy()
#     prev_pts = good_new.reshape(-1, 1, 2)
    
#     # Add new features if points are getting low
#     if len(prev_pts) < min_points * 2:
#         new_features = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
#         if new_features is not None:
#             prev_pts = np.vstack((prev_pts, new_features))

#     # Security trigger
#     if warnings >= max_warnings:
#         print("SECURITY ALERT: Camera tampering detected!")
#         # Add buzzer activation code here
#         break

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()








# import cv2
# import numpy as np

# # Initialize webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Read first frame and get dimensions
# ret, prev_frame = cap.read()
# if not ret:
#     print("Error: Could not read initial frame.")
#     cap.release()
#     exit()

# height, width = prev_frame.shape[:2]
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# # Fixed boundary points (corners + center)
# boundary_points = np.array([
#     [[0, 0]], [[width-1, 0]],          # Top points
#     [[0, height-1]], [[width-1, height-1]],  # Bottom points
#     [[width//2, 0]], [[width//2, height-1]],  # Vertical center
#     [[0, height//2]], [[width-1, height//2]]  # Horizontal center
# ], dtype=np.float32)

# # Detection parameters
# warnings = 0
# max_warnings = 3
# translation_threshold = 15  # Reduced from 50px
# rotation_threshold = 5      # Reduced from 15°
# min_points_for_detection = 4

# # Optical flow parameters
# lk_params = dict(winSize=(25, 25),
#                 maxLevel=3,
#                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Process current frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Track boundary points using optical flow
#     new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, boundary_points, None, **lk_params)
    
#     movement_detected = False
#     if new_pts is not None and status is not None:
#         # Fix array dimensions
#         status_flat = status.flatten()
#         valid_indices = np.where(status_flat == 1)[0]
        
#         if len(valid_indices) >= min_points_for_detection:
#             # Get valid corresponding points
#             valid_prev = boundary_points[valid_indices].reshape(-1, 2)
#             valid_new = new_pts[valid_indices].reshape(-1, 2)
            
#             # Calculate transformation matrix
#             M, inliers = cv2.estimateAffinePartial2D(valid_prev, valid_new)
            
#             if M is not None:
#                 # Extract movement parameters
#                 dx = M[0, 2]
#                 dy = M[1, 2]
#                 rotation = np.arctan2(M[1, 0], M[0, 0]) * 180/np.pi
#                 translation = np.sqrt(dx**2 + dy**2)
                
#                 # Visual debugging
#                 for i in valid_indices:
#                     a, b = new_pts[i][0]
#                     c, d = boundary_points[i][0]
#                     cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
#                     cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

#                 # Check movement thresholds
#                 if translation > translation_threshold or abs(rotation) > rotation_threshold:
#                     movement_detected = True
#                     warnings += 1
#                     print(f"Warning {warnings}/3: Translation: {translation:.1f}px, Rotation: {rotation:.1f}°")

#     # Visual feedback
#     cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 2)
    
#     if movement_detected:
#         cv2.putText(frame, "MOVEMENT DETECTED!", (50, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#         cv2.putText(frame, f"Translation: {translation:.1f}px", (50, 150),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#         cv2.putText(frame, f"Rotation: {rotation:.1f}°", (50, 200),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     # Status display
#     status_text = f"Warnings: {warnings}/3 | Tracking points: {len(valid_indices) if 'valid_indices' in locals() else 0}"
#     cv2.putText(frame, status_text, (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
#     cv2.imshow("Camera Security System", frame)

#     # Update previous frame data
#     prev_gray = gray.copy()
#     if new_pts is not None and status is not None:
#         boundary_points[valid_indices] = new_pts[valid_indices]

#     # Check security triggers
#     if warnings >= max_warnings:
#         print("ALERT: Security protocol activated!")
#         # Add your buzzer activation code here
#         break

#     # Exit on 'q' press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()