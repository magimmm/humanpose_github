import math


class Skeleton:
    """
    Class representing a skeleton detected or annotated in an image.
    """
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
        self.limbs = []
        self.limbs_indexes_in_body_list = []
        self.missing_landmarks_indexes = []

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
        """
        Find missing landmarks in the skeleton.
        """
        for i, landmark in enumerate(self.all_landmarks):
            if landmark == [0, 0]:
                self.missing_landmarks_indexes.append(i)

    def calculate_limbs_distances(self):
        """
        Calculate distances between different limbs of the skeleton.
        """
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
        self.wrist_to_wrist = calculate_distance(self.right_wrist, self.left_wrist)
        self.elbow_to_elbow = calculate_distance(self.right_elbow, self.left_elbow)
        self.right_wrist_hip = calculate_distance(self.right_wrist, self.right_hip)
        self.left_wrist_hip = calculate_distance(self.left_wrist, self.left_hip)
        self.right_wrist_left_hip = calculate_distance(self.right_wrist, self.left_hip)
        self.left_wrist_right_hip = calculate_distance(self.left_wrist, self.right_hip)
        self.right_wrist_left_shoulder = calculate_distance(self.right_wrist, self.left_shoulder)
        self.left_wrist_right_shoulder = calculate_distance(self.left_wrist, self.right_shoulder)

        self.limbs = [self.right_wrist_to_elbow,
                      self.right_elbow_to_shoulder,
                      self.left_wrist_to_elbow,
                      self.left_elbow_to_shoulder,
                      self.right_hip_to_shoulder,
                      self.left_hip_to_shoulder,
                      self.hip_to_hip,
                      self.shoulder_to_shoulder]
        self.limbs_indexes_in_body_list = [[10, 8], [8, 6], [9, 7], [7, 5], [12, 6], [11, 5], [11, 12], [5, 6]]

    def create_body_nn_feature_vector(self):
        """
        Create a feature vector for the body.
        """
        self.calculate_limbs_distances()
        self.features_vector = [self.right_wrist_to_elbow,
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
    def find_bounding_box(self):
        """
        Find the bounding box of the skeleton.
        """
        x_min = min(landmark[0] for landmark in self.all_landmarks)
        y_min = min(landmark[1] for landmark in self.all_landmarks)
        x_max = max(landmark[0] for landmark in self.all_landmarks)
        y_max = max(landmark[1] for landmark in self.all_landmarks)

        self.diagonal_bounding_box = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

    def preprocess_for_comparing(self):
        """
        Preprocess the skeleton for comparison.
        """
        self.calculate_limbs_distances()
        self.find_missing_points()
        self.create_body_nn_feature_vector()
        self.find_bounding_box()

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

    def find_face(self):
        """
        Find the face region of the skeleton.
        """
        landmarks = [landmark for landmark in
                     [self.nose, self.left_eye, self.right_eye, self.left_ear, self.right_ear, self.left_shoulder,
                      self.right_shoulder] if landmark != [0, 0]]

        # Calculate minimum and maximum coordinates using valid landmarks
        self.x_min = min(landmark[0] for landmark in landmarks)
        self.x_max = max(landmark[0] for landmark in landmarks)
        self.y_min = min(landmark[1] for landmark in landmarks)
        self.y_max = max(landmark[1] for landmark in landmarks)


def calculate_distance(point_one, point_two):
    """
    Calculate the Euclidean distance between two points.

    Args:
        point_one (tuple): Coordinates of the first point.
        point_two (tuple): Coordinates of the second point.

    Returns:
        float: Euclidean distance between the two points.
    """
    x1, y1 = point_one[0], point_one[1]
    x2, y2 = point_two[0], point_two[1]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
