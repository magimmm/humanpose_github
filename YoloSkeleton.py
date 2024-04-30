from Skeleton import Skeleton


class YoloSkeleton(Skeleton):
    """
yolo skeleton of images
    """

    def __init__(self, path=None):
        super().__init__()
        self.model_type = 'yolo'
        self.path = path

    def setup_from_detector(self, detected_landmarks):
        self.nose = detected_landmarks[0]
        self.left_eye = detected_landmarks[1]
        self.right_eye = detected_landmarks[2]
        self.left_ear = detected_landmarks[3]
        self.right_ear = detected_landmarks[4]
        self.left_shoulder = detected_landmarks[5]
        self.right_shoulder = detected_landmarks[6]
        self.left_elbow = detected_landmarks[7]
        self.right_elbow = detected_landmarks[8]
        self.left_wrist = detected_landmarks[9]
        self.right_wrist = detected_landmarks[10]
        self.left_hip = detected_landmarks[11]
        self.right_hip = detected_landmarks[12]
        self.setup_all_landmarks()

    def setup_from_annotation_file(self, keypoints, path):
        self.nose = (keypoints[0], keypoints[1])
        self.right_eye = (keypoints[6], keypoints[7])
        self.left_eye = (keypoints[15], keypoints[16])
        self.right_ear = (keypoints[21], keypoints[22])
        self.left_ear = (keypoints[24], keypoints[25])
        self.right_shoulder = (keypoints[33], keypoints[34])
        self.left_shoulder = (keypoints[36], keypoints[37])
        self.right_elbow = (keypoints[39], keypoints[40])
        self.left_elbow = (keypoints[42], keypoints[43])
        self.right_wrist = (keypoints[45], keypoints[46])
        self.left_wrist = (keypoints[48], keypoints[49])
        self.right_hip = (keypoints[69], keypoints[70])
        self.left_hip = (keypoints[72], keypoints[73])
        self.path = path
        self.setup_all_landmarks()

    def setup_all_landmarks(self):
        self.all_landmarks = [self.nose,
                              self.left_eye,
                              self.right_eye,
                              self.left_ear,
                              self.right_ear,
                              self.left_shoulder,
                              self.right_shoulder,
                              self.left_elbow,
                              self.right_elbow,
                              self.left_wrist,
                              self.right_wrist,
                              self.left_hip,
                              self.right_hip]
        self.body = [
            self.left_shoulder,
            self.right_shoulder,
            self.left_elbow,
            self.right_elbow,
            self.left_wrist,
            self.right_wrist,
            self.left_hip,
            self.right_hip]
