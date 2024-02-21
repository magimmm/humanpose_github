class MediaPipeSkeleton:
    """
mediapipe skeleton for annotated images
    """

    def __init__(self, keypoints, path):
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
        self.type='mediapipe'

        self.all_landmarks = [self.nose, self.right_eye_inner, self.right_eye, self.right_eye_outer,
                              self.left_eye_inner,
                              self.left_eye, self.left_eye_outer, self.right_ear, self.left_ear, self.mouth_right,
                              self.mouth_left, self.right_shoulder, self.left_shoulder, self.right_elbow,
                              self.left_elbow,
                              self.right_wrist, self.left_wrist, self.right_pinky_finger, self.left_pinky_finger,
                              self.right_index_finger, self.left_index_finger, self.right_thumb, self.left_thumb,
                              self.right_hip, self.left_hip]
