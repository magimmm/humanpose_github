from ultralytics import YOLO


class YoloDetector():
    """
        A class for detecting landmarks using YOLO.
    """
    def __init__(self, model):
        self.model = None
        self.model_type = model

    def setup(self):
        """
            Set up the Yolo detection model.
        """
        self.model = YOLO(self.model_type)

    def get_landmarks(self, img):
        """
            Detect landmarks in an image.

            Args:
                img: Input image.

            Returns:
                list: Detected landmarks.
        """
        left_hip_index = 12
        results = self.model(source=img, show=False, save=False, verbose=False)
        landmarks = [landmark.cpu().numpy() for landmark in results[0].keypoints.xy]
        detected_landmarks = landmarks[0]
        all_landmarks = []
        for i, landmark in enumerate(detected_landmarks):
            if landmark[0] == 0 and landmark[1] == 0:
                all_landmarks.append([0, 0])
            else:
                landmark_x = int(landmark[0])
                landmark_y = int(landmark[1])
                all_landmarks.append([landmark_x, landmark_y])
            if i == left_hip_index:
                break
        return all_landmarks
