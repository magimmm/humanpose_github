import copy
from SkeletonLoader import SkeletonsLoader
from SkeletonDetector import SkeletonDetector
from utils import calculate_distance
import numpy as np
import time
from ultralytics import YOLO
import cv2
import mediapipe as mp


class LandmarkTester:
    """
        A class for testing the accuracy of landmark detection methods.
    """

    def __init__(self):
        self.img_count = None
        self.loaded_skeletons_yolo = None
        self.loaded_skeletons_mediapipe = None
        self.detected_skeletons_mediapipe = []
        self.detected_skeletons_yolo = []
        self.images_paths = []

    def load_skeletons_from_annotation_files(self, path_normal_images, path_abnormal_images,
                                             path_annotation_file_normal, path_annotation_file_abnormal,
                                             show_yolo_landmarks, show_mediapipe_landmarks):
        """
        Load skeletons from annotation files.

        Args:
            path_normal_images (str): Path to normal images.
            path_abnormal_images (str): Path to abnormal images.
            path_annotation_file_normal (str): Path to the annotation file for normal images.
            path_annotation_file_abnormal (str): Path to the annotation file for abnormal images.
            show_yolo_landmarks (bool): Whether to show YOLO landmarks.
            show_mediapipe_landmarks (bool): Whether to show MediaPipe landmarks.
        """

        skeleton_loader = SkeletonsLoader(path_normal_images, path_abnormal_images, path_annotation_file_normal,
                                          path_annotation_file_abnormal)
        skeleton_loader.create_skeletons()

        if show_yolo_landmarks:
            skeleton_loader.view_skeletons_annotated('yolo')
        if show_mediapipe_landmarks:
            skeleton_loader.view_skeletons_annotated('mediapipe')

        self.loaded_skeletons_mediapipe = copy.deepcopy(skeleton_loader.skeletons_mediapipe)
        self.loaded_skeletons_yolo = copy.deepcopy(skeleton_loader.skeletons_yolo)
        self.create_path_lists_in_order()

    def create_path_lists_in_order(self):
        """
        Create lists of image paths.
        """
        for skeleton in self.loaded_skeletons_yolo:
            self.images_paths.append(skeleton.path)

    def create_skeletons_from_detections(self, images_paths):
        """
        Create skeletons from detected landmarks.

        Args:
            images_paths (list): List of image paths.
        """
        skeleton_detector = SkeletonDetector()
        skeleton_detector.detect_and_create_skeletons(images_paths)
        self.detected_skeletons_mediapipe = skeleton_detector.detected_skeletons_mediape
        self.detected_skeletons_yolo = skeleton_detector.detected_skeletons_yolo

    def compare(self):
        """
        Compare detected landmarks with annotated landmarks.
        """
        self.img_count = len(self.detected_skeletons_mediapipe)
        for detected_mp_skeleton, detected_y_skeleton, real_mp_skeleton, real_y_skeleton in zip(
                self.detected_skeletons_mediapipe, self.detected_skeletons_yolo, self.loaded_skeletons_mediapipe,
                self.loaded_skeletons_yolo):
            detected_mp_skeleton.preprocess_for_comparing()
            detected_y_skeleton.preprocess_for_comparing()
            real_mp_skeleton.preprocess_for_comparing()
            real_y_skeleton.preprocess_for_comparing()

        self.pcp()
        self.mpjpe()
        self.pck()
        self.mse()
        self.missing_landmarks()

    def measure_time(self):
        """
            Measure the time taken by detection methods.
        """
        time_mp = self.measure_time_mp()
        time_yolo = self.measure_time_yolo()
        print('new time Mediapipe, Yolo')
        print('Time: ', time_mp, time_yolo)
        self.images_paths

    def measure_time_mp(self):
        """
          Measure the time taken for detecting landmarks using MediaPipe.

          Returns:
              float: Time taken for detecting landmarks using MediaPipe.
        """
        start = time.time()
        mp_pose = mp.solutions.pose
        pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=0)
        # Initialize mediapipe drawing class - to draw the landmarks points.
        mp_drawing = mp.solutions.drawing_utils
        for path in self.images_paths:
            img = cv2.imread(path)
            image_in_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose_image.process(image_in_RGB)
            # deteced_landmarks = result.pose_landmarks.landmark
        end = time.time()
        duration = end - start
        return duration

    def measure_time_yolo(self):
        """
           Measure the time taken for detecting landmarks using YOLO.
           Note: Yolo gives outupt about time from processing each image

           Returns:
               float: Time taken for detecting landmarks using YOLO.
       """

        models = ['yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt']
        start = time.time()
        model = YOLO(models[0])
        for path in self.images_paths:
            results = model(source=path, show=False, save=False, verbose=False, max_det=1)
        end = time.time()
        duration = end - start
        return duration

    def calculate_pcp(self, detected_skeletons, real_skeletons):
        """
        Calculate the Percentage of Correct Parts (PCP) metric.

        Args:
            detected_skeletons (list): List of detected skeletons.
            real_skeletons (list): List of real skeletons.

        Returns:
            float: Percentage of Correct Parts.
        """
        limbs_count = len(detected_skeletons[0].limbs)
        total_percentage = 0
        for detected_skeleton, real_skeleton in zip(detected_skeletons, real_skeletons):
            percentage = self.calculate_pcp_one_img(detected_skeleton, real_skeleton, limbs_count)
            total_percentage += percentage
        return (percentage / self.img_count) * 100

    def calculate_pcp_one_img(self, detected_skeleton, real_skeleton, limbs_count):
        """
        Calculate the Percentage of Correct Parts (PCP) for one image.

        Args:
            detected_skeleton (Skeleton): Detected skeleton.
            real_skeleton (Skeleton): Real skeleton.
            limbs_count (int): Number of limbs.

        Returns:
            float: Percentage of Correct Parts for one image.
        """
        correct = 0
        for i, limb_keypoints in enumerate(detected_skeleton.limbs_indexes_in_body_list):
            limb_length = detected_skeleton.limbs[i]
            limb_start_real = real_skeleton.all_landmarks_as_yolo[limb_keypoints[0]]
            limb_end_real = real_skeleton.all_landmarks_as_yolo[limb_keypoints[1]]
            limb_start_detected = detected_skeleton.all_landmarks_as_yolo[limb_keypoints[0]]
            limb_end_detected = detected_skeleton.all_landmarks_as_yolo[limb_keypoints[1]]

            dist_limb_start = calculate_distance(limb_start_detected, limb_start_real)
            dist_limb_end = calculate_distance(limb_end_detected, limb_end_real)
            threshold = limb_length * 0.5
            if dist_limb_end < threshold and dist_limb_start < threshold:
                correct += 1

        return correct / limbs_count

    def pcp(self):
        """
        Calculate and print the Percentage of Correct Parts (PCP) for MediaPipe and YOLO.
        """
        percentage_mediapipe = self.calculate_pcp(self.detected_skeletons_mediapipe, self.loaded_skeletons_mediapipe)
        percentage_yolo = self.calculate_pcp(self.detected_skeletons_yolo, self.loaded_skeletons_yolo)
        print('PCP, meddiapipe, yolo: ', percentage_mediapipe * 100, percentage_yolo * 100)

    def pck(self):
        """
        Calculate and print the Percentage of Correct Keypoints (PCK) for MediaPipe, both reduced and all keypoints,
        and YOLO.
        """
        mp_all = 0
        mp_reduced = 0
        yolo_all = 0
        for detected_mp_skeleton, detected_y_skeleton, real_mp_skeleton, real_y_skeleton in zip(
                self.detected_skeletons_mediapipe, self.detected_skeletons_yolo, self.loaded_skeletons_mediapipe,
                self.loaded_skeletons_yolo):
            diagonal_mp_all = real_mp_skeleton.diagonal_bounding_box
            diagonal_yolo = real_y_skeleton.diagonal_bounding_box
            percentage_all_mediapipe = self.pck_one_img(detected_mp_skeleton.all_landmarks,
                                                        real_mp_skeleton.all_landmarks, diagonal_mp_all)
            percentage_reduced_mediapipe = self.pck_one_img(detected_mp_skeleton.all_landmarks_as_yolo,
                                                            real_mp_skeleton.all_landmarks_as_yolo, diagonal_yolo)
            percentage_yolo = self.pck_one_img(detected_y_skeleton.all_landmarks, real_y_skeleton.all_landmarks,
                                               diagonal_yolo)

            mp_all += percentage_all_mediapipe
            mp_reduced += percentage_reduced_mediapipe
            yolo_all += percentage_yolo

        mp_all = mp_all / self.img_count
        mp_reduced = mp_reduced / self.img_count
        yolo_all = yolo_all / self.img_count
        print('pck: mpreduced, mpall, yolo', mp_reduced * 100, mp_all * 100, yolo_all * 100)

    def pck_one_img(self, detected_landmarks, real_landmarks, diagonal):
        """
        Calculate the Percentage of Correct Keypoints (PCK) for one image.

        Args:
            detected_landmarks (list): Detected landmarks.
            real_landmarks (list): Real landmarks.
            diagonal (float): Diagonal of the bounding box.

        Returns:
            float: Percentage of Correct Keypoints for one image.
        """
        annotated_array = np.array(real_landmarks)
        detected_array = np.array(detected_landmarks)

        threshold = diagonal * 0.1

        non_zero_indices = np.all(detected_array != [0, 0], axis=1)
        annotated_non_zero = annotated_array[non_zero_indices]
        detected_non_zero = detected_array[non_zero_indices]

        distances = np.sqrt(np.sum(np.square(annotated_non_zero - detected_non_zero), axis=1))
        distances_in_threshold = np.sum(distances < threshold)
        return distances_in_threshold / len(non_zero_indices)

    def missing_landmarks(self):
        """
        Calculate and print the number of missing landmarks for MediaPipe, both reduced and all keypoints,
        and YOLO.
        """

        mp_all = 0
        mp_reduced = 0
        yolo_all = 0
        for detected_mp_skeleton, detected_y_skeleton in zip(
                self.detected_skeletons_mediapipe, self.detected_skeletons_yolo):
            detected_mp_skeleton.find_missing_points_as_yolo()
            mp_all += len(detected_mp_skeleton.missing_landmarks_indexes)
            mp_reduced += len(detected_mp_skeleton.missing_landmarks_indexes_as_yolo)
            yolo_all += len(detected_y_skeleton.missing_landmarks_indexes)

        mp_all = mp_all / self.img_count
        mp_reduced = mp_reduced / self.img_count
        yolo_all = yolo_all / self.img_count
        print('missing: mpreduced, mpall, yolo', mp_reduced, mp_all, yolo_all)

    def mse(self):
        """
        Calculate and print the Mean Squared Error (MSE) for MediaPipe, both reduced and all keypoints,
        and YOLO.
        """
        mp_all = 0
        mp_reduced = 0
        yolo_all = 0
        for detected_mp_skeleton, detected_y_skeleton, real_mp_skeleton, real_y_skeleton in zip(
                self.detected_skeletons_mediapipe, self.detected_skeletons_yolo, self.loaded_skeletons_mediapipe,
                self.loaded_skeletons_yolo):
            mse_all_mediapipe = self.calculate_mse(detected_mp_skeleton.all_landmarks,
                                                   real_mp_skeleton.all_landmarks)
            mse_reduced_mediapipe = self.calculate_mse(detected_mp_skeleton.all_landmarks_as_yolo,
                                                       real_mp_skeleton.all_landmarks_as_yolo)
            mse_yolo = self.calculate_mse(detected_y_skeleton.all_landmarks, real_y_skeleton.all_landmarks)

            mp_all += mse_all_mediapipe
            mp_reduced += mse_reduced_mediapipe
            yolo_all += mse_yolo

        mp_all = mp_all / self.img_count
        mp_reduced = mp_reduced / self.img_count
        yolo_all = yolo_all / self.img_count
        print('mse: mpreduced, mpall, yolo', mp_reduced, mp_all, yolo_all)

    def calculate_mse(self, detected_landmarks, annotated_landmarks):
        """
        Calculate the Mean Squared Error (MSE) for one image.

        Args:
            detected_landmarks (list): Detected landmarks.
            annotated_landmarks (list): Annotated landmarks.

        Returns:
            float: Mean Squared Error for one image.
        """
        annotated_array = np.array(annotated_landmarks)
        detected_array = np.array(detected_landmarks)

        # Filter out points with coordinates (0, 0)
        valid_indices = np.all(detected_array != 0, axis=1)
        annotated_array = annotated_array[valid_indices]
        detected_array = detected_array[valid_indices]

        # Calculate squared differences between corresponding coordinates
        squared_diffs = np.square(annotated_array - detected_array)

        # Calculate the mean squared error
        mse = np.mean(squared_diffs)

        return mse

    def mpjpe(self):
        """
        Calculate and print the Mean Per Joint Position Error (MPJPE) for MediaPipe, both reduced and all keypoints,
        and YOLO.
        """
        mp_all = 0
        mp_reduced = 0
        yolo_all = 0
        for detected_mp_skeleton, detected_y_skeleton, real_mp_skeleton, real_y_skeleton in zip(
                self.detected_skeletons_mediapipe, self.detected_skeletons_yolo, self.loaded_skeletons_mediapipe,
                self.loaded_skeletons_yolo):
            euclid_all_mediapipe = self.calculate_mpjpe_one_img(detected_mp_skeleton.all_landmarks,
                                                                real_mp_skeleton.all_landmarks)
            euclid_reduced_mediapipe = self.calculate_mpjpe_one_img(detected_mp_skeleton.all_landmarks_as_yolo,
                                                                    real_mp_skeleton.all_landmarks_as_yolo)
            euclid_yolo = self.calculate_mpjpe_one_img(detected_y_skeleton.all_landmarks, real_y_skeleton.all_landmarks)

            mp_all += euclid_all_mediapipe
            mp_reduced += euclid_reduced_mediapipe
            yolo_all += euclid_yolo

        mp_all = mp_all / self.img_count
        mp_reduced = mp_reduced / self.img_count
        yolo_all = yolo_all / self.img_count
        print('mpjpe: mpreduced, mpall, yolo', mp_reduced, mp_all, yolo_all)

    def calculate_mpjpe_one_img(self, detected_landmarks, real_landmarks):
        """
        Calculate the Mean Per Joint Position Error (MPJPE) for one image.

        Args:
            detected_landmarks (list): Detected landmarks.
            real_landmarks (list): Real landmarks.

        Returns:
            float: Mean Per Joint Position Error for one image.
        """
        # Convert landmarks to numpy arrays for easier computation
        annotated_array = np.array(real_landmarks)
        detected_array = np.array(detected_landmarks)

        # Filter out points with coordinates (0, 0)
        valid_indices = np.all(detected_array != 0, axis=1)
        annotated_array = annotated_array[valid_indices]
        detected_array = detected_array[valid_indices]

        # Calculate the Euclidean distance between each pair of corresponding landmarks
        distances = np.sqrt(np.sum(np.square(annotated_array - detected_array), axis=1))
        # Calculate the average Euclidean distance
        avg_distance = np.mean(distances)

        return avg_distance
