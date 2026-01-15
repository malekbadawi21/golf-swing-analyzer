# Golf Swing Analyzer - Basic Video Player

import cv2
import sys
from pose_detector import PoseDetector

# Open video file
video_path = "videos/sample_swing.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    print("Make sure the file exists in the videos/ folder")
    sys.exit(1)

#Create pose detector 
detector = PoseDetector(0.7)

# Main video playback loop
print("Playing video... Press 'q' to quit")

while True:
    # Read next frame
    ret, frame = cap.read()

    # Check if frame was read successfully
    if not ret:
        print("End of video reached")
        break

    #Detect pose
    results= detector.detect_pose(frame)

    #Draw landmarks on the frame
    annotated_frame= detector.draw_landmarks(frame, results)

    # Display frame
    cv2.imshow('Golf Swing Analyzer', annotated_frame)

    # Wait and check for keypress (100ms = ~10 FPS playback)
    key = cv2.waitKey(100) & 0xFF

    # Exit if 'q' pressed
    if key == ord('q'):
        print("Playback stopped by user")
        break

# Cleanup - release resources
cap.release()
cv2.destroyAllWindows()

print("Video player closed successfully")