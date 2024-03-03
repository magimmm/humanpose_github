class YoloSkeleton:
    """
yolo skeleton of annotated images
    """

    def __init__(self, keypoints, path):
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
        self.type='yolo'

        self.all_landmarks = [self.nose,
                              self.left_eye,
                              self.right_eye,
                              self.left_ear,
                              self.left_shoulder,
                              self.right_shoulder,
                              self.left_elbow,
                              self.right_elbow,
                              self.left_wrist,
                              self.right_wrist,
                              self.left_hip,
                              self.right_hip]
