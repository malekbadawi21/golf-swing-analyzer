# Golf Swing Analyzer - Basic Video Player

import cv2
import sys
from pose_detector import PoseDetector
from swing_analyzer import SwingAnalyzer

# Open video file
video_path = "videos/sample_swing3.mov"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    print("Make sure the file exists in the videos/ folder")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

#Create pose detector 
detector = PoseDetector(0.5)
analyzer = SwingAnalyzer(fps)

#Playback control
paused = False
frame_number = 0
annotated_frame = None

# Main video playback loop
print("Playing video... Press SPACE to pause, Press 'q' to quit, use arrow keys to step through")

while True:
    # Read next frame
    if not paused:
        ret, frame = cap.read()

        # Check if frame was read successfully
        if not ret:
            print("End of video reached")
            break

        frame_number += 1

    #Detect pose and get preprocessed frame
    if not paused or annotated_frame is None:
        results, processed_frame = detector.detect_pose_hybrid(frame)
        #Draw landmarks on the preprocessed frame
        annotated_frame = detector.draw_landmarks_with_visibility(processed_frame, results, min_visibility=0.3)

    analyzer.analyze_frame(results, frame_number)

    # Display frame
    cv2.imshow('Golf Swing Analyzer', annotated_frame)

    # Wait and check for keypress (100ms = 10 FPS playback)
    key = cv2.waitKey(100) & 0xFF

    # Exit if 'q' pressed
    if key == ord('q'):
        print("Playback stopped by user")
        break
    elif key == ord(' '): 
        paused = not paused
        if paused: print ("Video paused - Use arrow keys to step through; Space to resume")
        else: print("Video resumed")
    elif key == 3: #right arrow key
        if paused: 
            ret, frame = cap.read()
            if ret:
                frame_number += 1
                annotated_frame = None
    elif key == 2: #left arrow key
        if paused and frame_number>1: 
            frame_number -= 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                annotated_frame= None
    elif key == ord('r'): 
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_number = 0
        paused = False
        print("Video Restarted")

# Cleanup - release resources
cap.release()
cv2.destroyAllWindows()

print("Video player closed successfully")