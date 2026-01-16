import mediapipe as mp 
import cv2 

class PoseDetector: 
    def __init__ (self, confidence):
        self.confidence = confidence

        self.pose = mp.solutions.pose.Pose(
            #confidence required to say there is an individual detected
            min_detection_confidence=confidence,
            #confidence required to say the same individual has been tracked
            min_tracking_confidence=confidence,
            model_complexity=2,
            smooth_landmarks=True
        )

    #normalize frame format for different video codecs
    def preprocess_frame(self, frame):
        if frame.dtype != 'uint8':
            frame = frame.astype('uint8')
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    
    #detect pose method takes in the frame as BGR, converts to RGB and processes the result
    def detect_pose(self, frame):
        frame = self.preprocess_frame(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results, frame
    
    #draw landmarks method takes in the frames, checks if person is detected and uses landmarks to connect points
    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )
        return frame
    
    def draw_landmarks_with_visibility(self, frame, results, min_visibility=0.3):
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0,255,0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0,255,0,), thickness=2
                )
            )

            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if landmark.visibility < min_visibility:
                    h, w, _ = frame.shape
                    cx, cy= int(landmark.x * w), int(landmark.y *h)
                    cv2.circle(frame, (cx, cy,), 5, (0,0,255), -1) 
        return frame