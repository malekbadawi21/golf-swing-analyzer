# Golf Swing Analyzer - Basic Video Player
# This script loads and displays a golf swing video frame by frame

import cv2  # OpenCV library for computer vision operations
import sys  # System library for program exit and system operations

# Step 1: Create a VideoCapture object to read the video file
# cv2.VideoCapture() opens the video and prepares it for reading
video_path = "videos/sample_swing.mp4"
cap = cv2.VideoCapture(video_path)

# Step 2: Check if the video opened successfully
# cap.isOpened() returns True if the video file was found and can be read
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    print("Make sure the file exists in the videos/ folder")
    sys.exit(1)  # Exit the program with error code 1

# Step 3: Main video playback loop
# This loop runs continuously, reading and displaying frames until the video ends or user quits
print("Playing video... Press 'q' to quit")

while True:
    # Read the next frame from the video
    # ret = True if frame was successfully read, False if video ended
    # frame = the actual image data as a NumPy array
    ret, frame = cap.read()

    # Check if we successfully read a frame
    # If ret is False, we've reached the end of the video
    if not ret:
        print("End of video reached")
        break  # Exit the while loop

    # Display the frame in a window
    # First argument: window name (creates window if it doesn't exist)
    # Second argument: the image/frame to display
    cv2.imshow('Golf Swing Analyzer', frame)

    # Wait for 25 milliseconds and check if a key was pressed
    # cv2.waitKey(25) returns the ASCII code of the key pressed, or -1 if no key
    # 25ms delay = ~40 frames per second (1000ms / 25ms = 40 FPS)
    # The & 0xFF masks the result to get only the last 8 bits (handles different systems)
    key = cv2.waitKey(100) & 0xFF

    # Check if the user pressed 'q' (ASCII code for 'q' is 113)
    # ord('q') converts the character 'q' to its ASCII code
    if key == ord('q'):
        print("Playback stopped by user")
        break  # Exit the while loop

# Step 4: Cleanup - Release resources and close windows
# This is CRITICAL for proper resource management

# Release the VideoCapture object
# This closes the video file and frees up memory
# Without this, the file might remain locked and memory might leak
cap.release()

# Close all OpenCV windows
# This destroys any windows created by cv2.imshow()
# Without this, windows might remain open even after the program ends
cv2.destroyAllWindows()

print("Video player closed successfully")