import cv2
import mediapipe as mp
from ultralytics import YOLO

class MediaPipeDetector():
    def __init__(self):
        self.deteced_landmarks = None

    def setup_detector(self):
        mp_pose = mp.solutions.pose
        self.pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

        # Initialize mediapipe drawing class - to draw the landmarks points.
        self.mp_drawing = mp.solutions.drawing_utils


    def getlandmarks(self,img):
        img_width = img.shape[1]
        img_height = img.shape[0]
        annotated_image, result = self.detectPose(img,False)
        deteced_landmarks = result.pose_landmarks.landmark
        # Initialize lists to store the x and y coordinates of all landmarks
        all_landmarks = []
        # prejde vsetky landmarky a ak nejaky chyba, da mu nejaku specificku hodnotu
        for landmark in deteced_landmarks:
            if landmark['presence']:
                landmark_x = int(landmark.x * img_width)
                landmark_y = int(landmark.y * img_height)
                all_landmarks.append((landmark_x, landmark_y))
            else:
                # TODO nejaka hodnota pre nedetekovane landmarky
                all_landmarks.append(('null', 'null'))
            # Check if the current landmark is the one you want to stop at
            if landmark == self.mp_pose.PoseLandmark.LEFT_HIP:
                break  # Stop the loop if the landmark is reached
        return all_landmarks

    def detectPose(self, image_input, draw=False):
        original_image = image_input.copy()

        image_in_RGB = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

        resultant = self.pose.process(image_in_RGB)

        if resultant.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(image=original_image, landmark_list=resultant.pose_landmarks,
                                      connections=self.mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                   thickness=3, circle_radius=3),
                                      connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(49, 125, 237),
                                                                                     thickness=2, circle_radius=2))

        return original_image, resultant



class YoloDetector():
    def __init__(self,model='yolov8n-pose.pt'):
        self.model = None
        self.model_type=model

    def setup(self):
        # Load a model
        self.model = YOLO(self.model_type)

    def get_landmarks(self,img):
        left_hip_index=12
        # Predict with the model
        results = self.model(source=img, show=False, save=True)
        landmarks = [landmark.cpu().numpy() for landmark in results[0].keypoints.xy]
        detected_landmarks = landmarks[0]
        all_landmarks = []
        # prejde vsetky landmarky a ak nejaky chyba, da mu nejaku specificku hodnotu
        for i,landmark in enumerate(detected_landmarks):
            if landmark[0] == 0 and landmark[1]==0:
                # TODO nejaka hodnota pre nedetekovane landmarky
                all_landmarks.append(('null', 'null'))
            else:
                landmark_x = int(landmark.x)
                landmark_y = int(landmark.y)
                all_landmarks.append((landmark_x, landmark_y))

            # Check if the current landmark is the one you want to stop at
            if i==left_hip_index:
                break  # Stop the loop if the landmark is reached
        return all_landmarks

