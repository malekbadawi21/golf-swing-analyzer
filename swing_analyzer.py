import mediapipe as mp
import math

class SwingAnalyzer:
    def __init__(self, fps):
        self.fps = fps

        self.checkpoints = {
            'address': None,
            'takeaway': None,
            'lead_arm_parallel_back': None,
            'top': None,
            'lead_arm_parallel_down': None,
            'impact': None,
            'follow_through': None,
        }

        self.current_phase = 'address'
        self.prev_shoulder_rotation = None
        self.prev_wrist_y= None

    def calculate_angle(self, point1, point2, point3):
        vector1= [point1.x - point2.x,point1.y - point2.y]
        vector2= [point3.x - point2.x, point3.y - point2.y]

        dot_product= vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1= math.sqrt(vector1[0]**2 + vector1[1]**2)
        magnitude2= math.sqrt(vector2[0]**2 + vector2[1]**2)

        if magnitude1==0 or magnitude2==0:
            return 0
        
        cos_angle=dot_product/ (magnitude1*magnitude2)
        
        cos_angle=max(-1, min(1, cos_angle))

        angle = math.acos(cos_angle)
        return math.degrees(angle)
    
    def analyze_frame(self, results, frame_number):
        if not results.pose_landmarks:
            return
        landmarks=results.pose_landmarks.landmark

        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist= landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]

        if self.checkpoints['address'] is None:
            self.checkpoints['address'] = frame_number
            print(f"Address detected at frame {frame_number}")
        if (self.checkpoints['address'] is not None and 
            self.checkpoints['lead_arm_parallel_back'] is None):
            
            arm_angle=self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            arm_horizontal = abs(left_shoulder.y - left_wrist.y) <0.15

            if arm_angle>160 and arm_horizontal:
                self.checkpoints['lead_arm_parallel_back'] = frame_number
                print(f"Lead arm parallel (backswing) detected at frame {frame_number}")

        shoulder_rotation = left_shoulder.x-right_shoulder.x

        if (self.checkpoints['lead_arm_parallel_back'] is not None and
            self.checkpoints['top'] is None and 
            self.prev_shoulder_rotation is not None):

            if shoulder_rotation < self.prev_shoulder_rotation:
                self.checkpoints['top'] = frame_number
                print(f"Top of backswing detected at frame {frame_number}")

        if (self.checkpoints['top'] is not None and 
            self.checkpoints['lead_arm_parallel_down'] is None):
            
            arm_angle=self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            arm_horizontal = abs(left_shoulder.y - left_wrist.y) <0.15

            if arm_angle>160 and arm_horizontal:
                self.checkpoints['lead_arm_parallel_down'] = frame_number
                print(f"Lead arm parallel (downswing) detected at frame {frame_number}")

        if (self.checkpoints['lead_arm_parallel_down'] is not None and
            self.checkpoints['impact'] is None and 
            self.prev_wrist_y is not None):

            if left_wrist.y < self.prev_wrist_y:
                self.checkpoints['impact'] = frame_number
                print(f"Impact detected at frame {frame_number}")
        
        if (self.checkpoints['impact'] is not None and
            self.checkpoints['follow_through'] is None):

            if left_wrist.y <= left_shoulder.y:
                self.checkpoints['follow_through'] = frame_number
                print(f"Follow through detected at frame {frame_number}")

        self.prev_shoulder_rotation = shoulder_rotation
        self.prev_wrist_y = left_wrist.y