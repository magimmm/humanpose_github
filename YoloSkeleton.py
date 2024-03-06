import math
class YoloSkeleton:
    """
yolo skeleton of annotated images
    """

    def __init__(self, keypoints, path):
        self.left_elbow_to_shoulder = None
        self.right_elbow_to_shoulder = None
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
        self.type = 'yolo'

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
        # TODO len ruky su
        # self.body = [
        #     self.left_shoulder,
        #     self.right_shoulder,
        #     self.left_elbow,
        #     self.right_elbow,
        #     self.left_wrist,
        #     self.right_wrist,
        #     self.left_hip,
        #     self.right_hip]

    def calculate_arms_distances(self):
        self.right_wrist_to_elbow=calculate_distance(self.right_wrist,self.right_elbow)
        self.right_elbow_to_shoulder=calculate_distance(self.right_elbow,self.right_shoulder)
        self.right_wrist_to_shoulder=calculate_distance(self.right_wrist,self.right_shoulder)
        self.left_wrist_to_elbow = calculate_distance(self.left_wrist, self.left_elbow)
        self.left_elbow_to_shoulder = calculate_distance(self.left_elbow, self.left_shoulder)
        self.left_wrist_to_shoulder = calculate_distance(self.left_wrist, self.left_shoulder)
        
def calculate_distance(point_one,point_two):
    x1,y1=point_one[0],point_one[1]
    x2,y2=point_two[0],point_two[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


