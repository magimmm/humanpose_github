from Skeleton import Skeleton


class MediaPipeSkeleton(Skeleton):
    """
      A class representing a skeleton detected using the MediaPipe library.
    """

    def __init__(self):
        super().__init__()
        self.mouth_left = None
        self.right_eye_outer = None
        self.left_eye_outer = None
        self.mouth_right = None
        self.left_eye_inner = None
        self.right_eye_inner = None
        self.right_eye_inner = None
        self.type = 'mediapipe'
        self.missing_landmarks_indexes_as_yolo = []

    def setup_from_detector(self, detected_landmarks):
        """
           Set up the skeleton based on detected landmarks from a detector.

           Args:
               detected_landmarks (list): Detected landmarks from a detector.
       """
        self.nose = detected_landmarks[0]
        self.left_eye_inner = detected_landmarks[1]
        self.left_eye = detected_landmarks[2]
        self.left_eye_outer = detected_landmarks[3]
        self.right_eye_inner = detected_landmarks[4]
        self.right_eye = detected_landmarks[5]
        self.right_eye_outer = detected_landmarks[6]
        self.left_ear = detected_landmarks[7]
        self.right_ear = detected_landmarks[8]
        self.mouth_left = detected_landmarks[9]
        self.mouth_right = detected_landmarks[10]
        self.left_shoulder = detected_landmarks[11]
        self.right_shoulder = detected_landmarks[12]
        self.left_elbow = detected_landmarks[13]
        self.right_elbow = detected_landmarks[14]
        self.left_wrist = detected_landmarks[15]
        self.right_wrist = detected_landmarks[16]
        self.left_pinky_finger = detected_landmarks[17]
        self.right_pinky_finger = detected_landmarks[18]
        self.left_index_finger = detected_landmarks[19]
        self.right_index_finger = detected_landmarks[20]
        self.left_thumb = detected_landmarks[21]
        self.right_thumb = detected_landmarks[22]
        self.left_hip = detected_landmarks[23]
        self.right_hip = detected_landmarks[24]

        self.setup_all_landmarks()

    def setup_from_annotation_file(self, keypoints, path):
        """
           Set up the skeleton based on annotated key-points from a file.

           Args:
               keypoints (list): Annotated key-points from a file.
               path (str): Path to the annotation file.
       """
        self.nose = (keypoints[0], keypoints[1])
        self.right_eye_inner = (keypoints[3], keypoints[4])
        self.right_eye = (keypoints[6], keypoints[7])
        self.right_eye_outer = (keypoints[9], keypoints[10])
        self.left_eye_inner = (keypoints[12], keypoints[13])
        self.left_eye = (keypoints[15], keypoints[16])
        self.left_eye_outer = (keypoints[18], keypoints[19])
        self.right_ear = (keypoints[21], keypoints[22])
        self.left_ear = (keypoints[24], keypoints[25])
        self.mouth_right = (keypoints[27], keypoints[28])
        self.mouth_left = (keypoints[30], keypoints[31])
        self.right_shoulder = (keypoints[33], keypoints[34])
        self.left_shoulder = (keypoints[36], keypoints[37])
        self.right_elbow = (keypoints[39], keypoints[40])
        self.left_elbow = (keypoints[42], keypoints[43])
        self.right_wrist = (keypoints[45], keypoints[46])
        self.left_wrist = (keypoints[48], keypoints[49])
        self.right_pinky_finger = (keypoints[51], keypoints[52])
        self.left_pinky_finger = (keypoints[54], keypoints[55])
        self.right_index_finger = (keypoints[57], keypoints[58])
        self.left_index_finger = (keypoints[60], keypoints[61])
        self.right_thumb = (keypoints[63], keypoints[64])
        self.left_thumb = (keypoints[66], keypoints[67])
        self.right_hip = (keypoints[69], keypoints[70])
        self.left_hip = (keypoints[72], keypoints[73])
        self.path = path
        self.setup_all_landmarks()

    def setup_all_landmarks(self):
        """
          Set up all landmarks and body parts.
        """
        self.all_landmarks = [self.nose, self.right_eye_inner, self.right_eye, self.right_eye_outer,
                              self.left_eye_inner,
                              self.left_eye, self.left_eye_outer, self.right_ear, self.left_ear, self.mouth_right,
                              self.mouth_left, self.right_shoulder, self.left_shoulder, self.right_elbow,
                              self.left_elbow,
                              self.right_wrist, self.left_wrist, self.right_pinky_finger, self.left_pinky_finger,
                              self.right_index_finger, self.left_index_finger, self.right_thumb, self.left_thumb,
                              self.right_hip, self.left_hip]

        self.body = [self.right_shoulder, self.left_shoulder, self.right_elbow,
                     self.left_elbow,
                     self.right_wrist, self.left_wrist, self.right_pinky_finger, self.left_pinky_finger,
                     self.right_index_finger, self.left_index_finger, self.right_thumb, self.left_thumb,
                     self.right_hip, self.left_hip]

    def find_missing_points_as_yolo(self):
        """
            Find missing landmarks from  using YOLO format (missing landmarks from these 13).
        """
        for i, landmark in enumerate(self.all_landmarks_as_yolo):
            if landmark == [0, 0]:
                self.missing_landmarks_indexes_as_yolo.append(i)
