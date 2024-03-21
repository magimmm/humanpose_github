import copy
from SkeletonLoader import SkeletonsLoader
from SkeletonDetector import SkeletonDetector
from utils import calculate_distance
import numpy as np
from MediapipeSkeleton import MediaPipeSkeleton

class LandmarkTester():
    def __init__(self):
        self.loaded_skeletons_yolo = None
        self.loaded_skeletons_mediapipe = None
        self.detected_skeletons_mediapipe = []
        self.detected_skeletons_yolo = []
        self.time_yolo = 0
        self.time_mediapipe = 0
        self.images_paths = []

    def load_skeletons_from_annotation_files(self, path_normal_images, path_abnormal_images, path_annotation_file_normal, path_annotation_file_abnormal, train_new_model, show_yolo_landmarks, show_mediapipe_landmarks):
        skeleton_loader = SkeletonsLoader(path_normal_images, path_abnormal_images, path_annotation_file_normal, path_annotation_file_abnormal)
        skeleton_loader.create_skeletons()

        if train_new_model:
            skeleton_loader.train_body_neuron_network()

        if show_yolo_landmarks:
            skeleton_loader.view_skeletons_annotated('yolo')
        if show_mediapipe_landmarks:
            skeleton_loader.view_skeletons_annotated('mediapipe')

        self.loaded_skeletons_mediapipe = copy.deepcopy(skeleton_loader.skeletons_mediapipe)
        self.loaded_skeletons_yolo = copy.deepcopy(skeleton_loader.skeletons_yolo)

    def create_path_lists_in_order(self):
        # create lists with filenames so the detected and annotated will be in the same order
        for skeleton in self.loaded_skeletons_yolo:
            self.images_paths.append(skeleton.path)

    def create_skeletons_from_detections(self):
        self.create_path_lists_in_order()
        skeleton_detector = SkeletonDetector()
        skeleton_detector.detect_and_create_skeletons(self.images_paths)
        self.detected_skeletons_mediapipe = skeleton_detector.detected_skeletons_mediape
        self.detected_skeletons_yolo = skeleton_detector.detected_skeletons_yolo
        self.time_yolo = skeleton_detector.time_yolo
        self.time_mediapipe = skeleton_detector.time_mediapipe

    def compare(self):
        self.img_count=len(self.detected_skeletons_mediapipe)
        for detected_mp_skeleton,detected_y_skeleton, real_mp_skeleton,real_y_skeleton in zip(self.detected_skeletons_mediapipe,self.detected_skeletons_yolo,self.loaded_skeletons_mediapipe,self.loaded_skeletons_yolo):
            detected_mp_skeleton.preprocess_for_comparing()
            detected_y_skeleton.preprocess_for_comparing()
            real_mp_skeleton.preprocess_for_comparing()
            real_y_skeleton.preprocess_for_comparing()

        print('Mediapipe, Yolo')
        print('Time: ', self.time_mediapipe,self.time_yolo)
        self.pcp()
        self.mpjpe()
        self.pck()
        self.mse()
        self.missing_landmarks()

    def calculate_pcp(self, detected_skeletons,real_skeletons):
        limbs_count = len(detected_skeletons[0].limbs)
        total_percentage=0
        for detected_skeleton,real_skeleton in zip(detected_skeletons,real_skeletons):
            percentage=self.calculate_pcp_one_img(detected_skeleton,real_skeleton,limbs_count)
            total_percentage+=percentage
        return percentage/self.img_count

    def calculate_pcp_one_img(self,detected_skeleton,real_skeleton,limbs_count):
        correct = 0
        for i, limb_keypoints in enumerate(detected_skeleton.limbs_indexes_in_body_list):
            limb_length = detected_skeleton.limbs[i]
            limb_start_real = real_skeleton.all_landmarks_as_yolo[limb_keypoints[0]]
            limb_end_real = real_skeleton.all_landmarks_as_yolo[limb_keypoints[1]]
            limb_start_detected = detected_skeleton.all_landmarks_as_yolo[limb_keypoints[0]]
            limb_end_detected = detected_skeleton.all_landmarks_as_yolo[limb_keypoints[1]]

            dist_limb_start = calculate_distance(limb_start_detected, limb_start_real)
            dist_limb_end = calculate_distance(limb_end_detected, limb_end_real)

            #distance = dist_limb_end + dist_limb_start

            #if distance < limb_length:
                #correct += 1
            if dist_limb_end<limb_length/2 and dist_limb_start<limb_length/2:
                correct+=1
        return correct/limbs_count

    def pcp(self):
        percentage_mediapipe=self.calculate_pcp(self.detected_skeletons_mediapipe,self.loaded_skeletons_mediapipe)
        percentage_yolo=self.calculate_pcp(self.detected_skeletons_yolo,self.loaded_skeletons_yolo)
        print('PCP, meddiapipe, yolo: ',percentage_mediapipe*100,percentage_yolo*100)

    def pck(self):
        mp_all = 0
        mp_reduced = 0
        yolo_all = 0
        for detected_mp_skeleton, detected_y_skeleton, real_mp_skeleton, real_y_skeleton in zip(
                self.detected_skeletons_mediapipe, self.detected_skeletons_yolo, self.loaded_skeletons_mediapipe,
                self.loaded_skeletons_yolo):
            percentage_all_mediapipe = self.pck_one_img(detected_mp_skeleton.all_landmarks,
                                                                real_mp_skeleton.all_landmarks)
            percentage_reduced_mediapipe = self.pck_one_img(detected_mp_skeleton.all_landmarks_as_yolo,
                                                                    real_mp_skeleton.all_landmarks_as_yolo)
            percentage_yolo = self.pck_one_img(detected_y_skeleton.all_landmarks, real_y_skeleton.all_landmarks)

            mp_all += percentage_all_mediapipe
            mp_reduced += percentage_reduced_mediapipe
            yolo_all += percentage_yolo

        mp_all = mp_all / self.img_count
        mp_reduced = mp_reduced / self.img_count
        yolo_all = yolo_all / self.img_count
        print('pck: mpreduced, mpall, yolo', mp_reduced*100, mp_all*100, yolo_all*100)


    def pck_one_img(self,detected_landmarks, real_landmarks):
        annotated_array = np.array(real_landmarks)
        detected_array = np.array(detected_landmarks)

        # Calculate the Euclidean distance between each pair of corresponding landmarks
        distances = np.sqrt(np.sum(np.square(annotated_array - detected_array), axis=1))
        distances_in_threshold = np.sum(distances < 70)

        return distances_in_threshold/len(real_landmarks)

    def missing_landmarks(self):
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

    def calculate_mse(self,detected_landmarks, annotated_landmarks):
        # Convert landmarks to numpy arrays for easier computation
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
        mp_all= 0
        mp_reduced = 0
        yolo_all =0
        for detected_mp_skeleton, detected_y_skeleton, real_mp_skeleton, real_y_skeleton in zip(
                self.detected_skeletons_mediapipe, self.detected_skeletons_yolo, self.loaded_skeletons_mediapipe,
                self.loaded_skeletons_yolo):
            euclid_all_mediapipe=self.calculate_mpjpe_one_img(detected_mp_skeleton.all_landmarks,real_mp_skeleton.all_landmarks)
            euclid_reduced_mediapipe=self.calculate_mpjpe_one_img(detected_mp_skeleton.all_landmarks_as_yolo,real_mp_skeleton.all_landmarks_as_yolo)
            euclid_yolo=self.calculate_mpjpe_one_img(detected_y_skeleton.all_landmarks,real_y_skeleton.all_landmarks)

            mp_all+=euclid_all_mediapipe
            mp_reduced+=euclid_reduced_mediapipe
            yolo_all+=euclid_yolo

        mp_all=mp_all/self.img_count
        mp_reduced=mp_reduced/self.img_count
        yolo_all=yolo_all/self.img_count
        print('mpjpe: mpreduced, mpall, yolo',mp_reduced,mp_all,yolo_all)


    def calculate_mpjpe_one_img(self,detected_landmarks, real_landmarks):
        # Convert landmarks to numpy arrays for easier computation
        annotated_array = np.array(real_landmarks)
        detected_array = np.array(detected_landmarks)

        # Filter out points with coordinates (0, 0)
        valid_indices = np.all(detected_array != 0, axis=1)
        # print(valid_indices)
        annotated_array = annotated_array[valid_indices]
        detected_array = detected_array[valid_indices]

        # Calculate the Euclidean distance between each pair of corresponding landmarks
        distances = np.sqrt(np.sum(np.square(annotated_array - detected_array), axis=1))
        # Calculate the average Euclidean distance
        avg_distance = np.mean(distances)
        # print(avg_distance)

        return a