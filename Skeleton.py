import math
class Skeleton:
    def __init__(self):
        self.left_wrist_to_elbow = None
        self.right_elbow_to_shoulder = None
        self.nose = None
        self.right_eye = None
        self.left_eye = None
        self.right_ear = None
        self.left_ear = None
        self.right_shoulder = None
        self.left_shoulder = None
        self.right_elbow = None
        self.left_elbow = None
        self.right_wrist = None
        self.left_wrist = None
        self.right_pinky_finger = None
        self.left_pinky_finger = None
        self.right_index_finger = None
        self.left_index_finger = None
        self.right_thumb = None
        self.left_thumb = None
        self.right_hip = None
        self.left_hip = None
        self.path = None
        self.all_landmarks = []
        self.body = []
        self.limbs=[]
        self.limbs_indexes_in_body_list=[]
        self.missing_landmarks_indexes=[]

    def setup_from_detector(self, detected_landmarks):
        pass

    def setup_from_annotation_file(self, keypoints, path):
        pass

    def setup_all_landmarks(self):
        pass

    def calculate_limbs_distances(self):
        pass

    def calculate_arms_distances(self):
        pass

    def create_body_nn_feature_vector(self):
        pass

    def create_body_nn_feature_vector_whole_body(self):
        pass

    def find_missing_points(self):
        for i, landmark in enumerate(self.all_landmarks):
            if landmark == [0, 0]:
                self.missing_landmarks_indexes.append(i)

    def calculate_limbs_distances(self):
        self.right_wrist_to_elbow = calculate_distance(self.right_wrist, self.right_elbow)
        self.right_elbow_to_shoulder = calculate_distance(self.right_elbow, self.right_shoulder)
        self.right_wrist_to_shoulder = calculate_distance(self.right_wrist, self.right_shoulder)
        self.left_wrist_to_elbow = calculate_distance(self.left_wrist, self.left_elbow)
        self.left_elbow_to_shoulder = calculate_distance(self.left_elbow, self.left_shoulder)
        self.left_wrist_to_shoulder = calculate_distance(self.left_wrist, self.left_shoulder)
        self.right_hip_to_shoulder = calculate_distance(self.right_hip, self.right_shoulder)
        self.left_hip_to_shoulder = calculate_distance(self.left_hip, self.left_shoulder)
        self.hip_to_hip = calculate_distance(self.left_hip, self.right_hip)
        self.shoulder_to_shoulder = calculate_distance(self.right_shoulder, self.left_shoulder)
        self.wrist_to_wrist=calculate_distance(self.right_wrist,self.left_wrist)
        self.elbow_to_elbow=calculate_distance(self.right_elbow,self.left_elbow)
        self.right_wrist_hip=calculate_distance(self.right_wrist,self.right_hip)
        self.left_wrist_hip=calculate_distance(self.left_wrist,self.left_hip)
        self.right_wrist_left_hip=calculate_distance(self.right_wrist,self.left_hip)
        self.left_wrist_right_hip=calculate_distance(self.left_wrist,self.right_hip)
        self.right_wrist_left_shoulder=calculate_distance(self.right_wrist,self.left_shoulder)
        self.left_wrist_right_shoulder=calculate_distance(self.left_wrist,self.right_shoulder)

        self.limbs=[self.right_wrist_to_elbow,
                    self.right_elbow_to_shoulder,
                    self.left_wrist_to_elbow,
                    self.left_elbow_to_shoulder,
                    self.right_hip_to_shoulder,
                    self.left_hip_to_shoulder,
                    self.hip_to_hip,
                    self.shoulder_to_shoulder]
        self.limbs_indexes_in_body_list = [[10,8],[8,6],[9,7],[7,5],[12,6],[11,5],[11,12],[5,6]]
    def create_body_nn_feature_vector(self):
        self.calculate_limbs_distances()
        self.features_vector=[self.right_wrist_to_elbow,
                              self.right_elbow_to_shoulder,
                              self.right_wrist_to_shoulder,
                             self.left_wrist_to_elbow,
                              self.left_elbow_to_shoulder,
                              self.left_wrist_to_shoulder,
                              self.right_hip_to_shoulder,
                              self.left_hip_to_shoulder,
                            self.wrist_to_wrist,
                            self.elbow_to_elbow,
                            self.right_wrist_hip,
                            self.left_wrist_hip,
                            self.right_wrist_left_hip,
                            self.left_wrist_right_hip,
                              self.right_wrist_left_shoulder,
                              self.left_wrist_right_shoulder]

    def create_body_nn_feature_vector_whole_body(self):
        self.features_vector_whole_body = []
        for index, keypoint in enumerate(self.body):
            if index < len(self.body) - 1:
                distance = calculate_distance(keypoint, self.body[index + 1])
                self.features_vector_whole_body.append(distance)

    def preprocess_for_comparing(self):
        self.calculate_limbs_distances()
        self.find_missing_points()
        self.create_body_nn_feature_vector()
        self.create_body_nn_feature_vector_whole_body()

        self.all_landmarks_as_yolo = [self.nose,
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

def calculate_distance(point_one,point_two):
    x1,y1=point_one[0],point_one[1]
    x2,y2=point_two[0],point_two[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
