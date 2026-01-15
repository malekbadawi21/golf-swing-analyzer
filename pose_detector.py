import mediapipe as mp 
import cv2 

class PoseDetector: 
    def __init__ (self, confidence):
        self.confidence = confidence

        self.pose = mp.solutions.pose.Pose(
            #confidence required to say there is an individual detected
            min_detection_confidence=confidence,
            #confidence required to say the same individual has been tracked
            min_tracking_confidence=confidence 
        )

    #detect pose method takes in the frame as BGR, converts to RGB and processes the result
    def detect_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results
    
    #draw landmarks method takes in the frames, checks if person is detected and uses landmarks to connect points
    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )
        return frame