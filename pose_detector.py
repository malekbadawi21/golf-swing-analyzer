import mediapipe as mp 
import cv2 

class PoseDetector: 
    def __init__ (self, confidence):
        self.confidence = confidence

        self.pose_smooth = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence,
            model_complexity=2,
            smooth_landmarks=True
        )

        self.pose_static = mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=confidence,
            model_complexity=2,
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
        results = self.pose_smooth.process(rgb_frame)
        return results, frame
    
    def detect_pose_hybrid(self, frame, visibility_threshold=0.5):
        frame = self.preprocess_frame(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_smooth.process(rgb_frame)

        if not results.pose_landmarks:
            return results, frame
        
        lead_arm_indices = [
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
            mp.solutions.pose.PoseLandmark.LEFT_WRIST.value,
        ]

        needs_redetection = False
        for idx in lead_arm_indices:
            if results.pose_landmarks.landmark[idx].visibility < visibility_threshold:
                needs_redetection = True
                break
        
        if needs_redetection:
            static_results= self.pose_static.process(rgb_frame)
            if static_results.pose_landmarks:
                for idx in lead_arm_indices:
                    static_landmark = static_results.pose_landmarks.landmark[idx]
                    if static_landmark.visibility> results.pose_landmarks.landmark[idx].visibility:
                        results.pose_landmarks.landmark[idx].x = static_landmark.x
                        results.pose_landmarks.landmark[idx].y = static_landmark.y
                        results.pose_landmarks.landmark[idx].z = static_landmark.z
                        results.pose_landmarks.landmark[idx].visibility = static_landmark.visibility

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