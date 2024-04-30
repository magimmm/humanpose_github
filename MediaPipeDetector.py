import cv2
import mediapipe as mp


class MediaPipeDetector:
    """
       A class for detecting landmarks using the MediaPipe library.
    """
    def __init__(self):
        self.mp_drawing = None
        self.pose_image = None
        self.mp_pose = None
        self.deteced_landmarks = None

    def setup_detector(self):
        """
            Set up the MediaPipe pose detection model.
        """
        self.mp_pose = mp.solutions.pose
        self.pose_image = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils

    def get_landmarks(self, img_path):
        """
            Detect landmarks in an image.

            Args:
                img_path (str): Path to the input image.

            Returns:
                list: Detected landmarks.
        """
        all_landmarks = []
        img = cv2.imread(img_path)
        annotated_image, result = self.detectPose(img, False)
        if not result.pose_landmarks or not result.pose_landmarks.landmark:
            print(img_path)
        else:
            deteced_landmarks = result.pose_landmarks.landmark
            img_width = img.shape[1]
            img_height = img.shape[0]
            index_left_hip = 24

            for (i, landmark) in enumerate(deteced_landmarks):
                if float(landmark.visibility) > 0.6:
                    landmark_x = int(landmark.x * img_width)
                    landmark_y = int(landmark.y * img_height)
                    all_landmarks.append([landmark_x, landmark_y])
                else:
                    all_landmarks.append([0, 0])
                if i == index_left_hip:
                    break
        return all_landmarks

    def detectPose(self, image_input, draw):
        """
            Detect pose in an image.

            Args:
                image_input (numpy.ndarray): Input image.
                draw (bool): Whether to draw landmarks on the image.

            Returns:
                tuple: Annotated image and result object containing pose landmarks.
        """
        original_image = image_input.copy()

        image_in_RGB = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

        result = self.pose_image.process(image_in_RGB)

        if result.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(image=original_image, landmark_list=result.pose_landmarks,
                                           connections=self.mp_pose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                             thickness=3,
                                                                                             circle_radius=3),
                                           connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(49, 125, 237),
                                                                                               thickness=2,
                                                                                               circle_radius=2))

        return original_image, result
