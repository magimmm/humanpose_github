import math
class YoloSkeleton:
    """
yolo skeleton of annotated images
    """

    def __init__(self):
        self.right_eye = None
        self.nose = None
        self.left_elbow_to_shoulder = None
        self.model_type = 'yolo'
        self.right_elbow_to_shoulder = None

    def setup_from_detector(self,detected_landmarks):
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

    def setup_from_annotation_file(self,keypoints,path):
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



    def calculate_limbs_distances(self):
        self.right_wrist_to_elbow = calculate_distance(self.right_wrist, self.right_elbow)
        self.right_elbow_to_shoulder = calculate_distance(self.right_elbow, self.right_shoulder)
        self.right_wrist_to_shoulder = calculate_distance(self.right_wrist, self.right_shoulder)
        self.left_wrist_to_elbow = calculate_distance(self.left_wrist, self.left_elbow)
        self.left_elbow_to_shoulder = calculate_distance(self.left_elbow, self.left_shoulder)
        self.left_wrist_to_shoulder = calculate_distance(self.left_wrist, self.left_shoulder)
        self.right_hip_to_shoulder=calculate_distance(self.right_hip,self.right_shoulder)
        self.left_hip_to_shoulder=calculate_distance(self.left_hip,self.left_shoulder)
        self.hip_to_hip=calculate_distance(self.left_hip,self.right_hip)
        self.shoulder_to_shoulder=calculate_distance(self.right_shoulder,self.left_shoulder)
    def calculate_arms_distances(self):
        self.right_wrist_to_elbow=calculate_distance(self.right_wrist,self.right_elbow)
        self.right_elbow_to_shoulder=calculate_distance(self.right_elbow,self.right_shoulder)
        self.right_wrist_to_shoulder=calculate_distance(self.right_wrist,self.right_shoulder)
        self.left_wrist_to_elbow = calculate_distance(self.left_wrist, self.left_elbow)
        self.left_elbow_to_shoulder = calculate_distance(self.left_elbow, self.left_shoulder)
        self.left_wrist_to_shoulder = calculate_distance(self.left_wrist, self.left_shoulder)

    def create_body_nn_feature_vector(self):
        self.calculate_arms_distances()
        self.features_vector=[self.right_wrist_to_elbow, self.right_elbow_to_shoulder, self.right_wrist_to_shoulder,
             self.left_wrist_to_elbow, self.left_elbow_to_shoulder, self.left_wrist_to_shoulder]

    def create_body_nn_feature_vector_whole_body(self):
        self.features_vector_whole_body = []
        for index, keypoint in enumerate(self.body):
            if index < len(self.body) - 1:
                distance = calculate_distance(keypoint, self.body[index + 1])
                self.features_vector_whole_body.append(distance)


def calculate_distance(point_one,point_two):
    x1,y1=point_one[0],point_one[1]
    x2,y2=point_two[0],point_two[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

