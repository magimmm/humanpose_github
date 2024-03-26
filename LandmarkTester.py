import copy
from SkeletonLoader import SkeletonsLoader
from SkeletonDetector import SkeletonDetector
from utils import calculate_distance
import numpy as np
import time
from ultralytics import YOLO
import cv2
import mediapipe as mp

class LandmarkTester():
    def __init__(self):
        self.loaded_skeletons_yolo = None
        self.loaded_skeletons_mediapipe = None
        self.detected_skeletons_mediapipe = []
        self.detected_skeletons_yolo = []
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
        self.create_path_lists_in_order()


    def create_path_lists_in_order(self):
        # create lists with filenames so the detected and annotated will be in the same order
        for skeleton in self.loaded_skeletons_yolo:
            self.images_paths.append(skeleton.path)

    def create_skeletons_from_detections(self,images_paths):
        skeleton_detector = SkeletonDetector()
        skeleton_detector.detect_and_create_skeletons(images_paths)
        self.detected_skeletons_mediapipe = skeleton_detector.detected_skeletons_mediape
        self.detected_skeletons_yolo = skeleton_detector.detected_skeletons_yolo

    def compare(self):
        # cv2.namedWindow('tesr',cv2.WINDOW_NORMAL)
        # img=cv2.imread(self.loaded_skeletons_yolo[0].path)
        # count=0
        # for d,r in zip(self.detected_skeletons_yolo[0].all_landmarks, self.loaded_skeletons_yolo[0].all_landmarks):
        #     wd = tuple(map(int, d))
        #     wr = tuple(map(int, r))
        #     cv2.circle(img, wd, 5, (255, 255, 255), -1)  # Green circle at limb start point
        #     cv2.circle(img, wr, 5, (0, 255, 0), -1)  # Green circle at limb end point
        #
        #     cv2.putText(img, str(count), (wd[0] + 10, wd[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        #     cv2.putText(img, str(count), (wr[0] + 10, wr[1] + 10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        #     count+=1
        # cv2.imshow('tesr', img)
        # cv2.waitKey(0)
        #




        self.img_count=len(self.detected_skeletons_mediapipe)
        for detected_mp_skeleton,detected_y_skeleton, real_mp_skeleton,real_y_skeleton in zip(self.detected_skeletons_mediapipe,self.detected_skeletons_yolo,self.loaded_skeletons_mediapipe,self.loaded_skeletons_yolo):
            detected_mp_skeleton.preprocess_for_comparing()
            detected_y_skeleton.preprocess_for_comparing()
            real_mp_skeleton.preprocess_for_comparing()
            real_y_skeleton.preprocess_for_comparing()

        # print('Mediapipe, Yolo')
        # print('Time: ', self.time_mediapipe,self.time_yolo)
        self.pcp()
        self.mpjpe()
        self.pck()
        self.mse()
        self.missing_landmarks()

    def measure_time(self):
        time_mp=self.measure_time_mp()
        time_yolo=self.measure_time_yolo()
        print('new time Mediapipe, Yolo')
        print('Time: ', time_mp, time_yolo)
        self.images_paths

    def measure_time_mp(self):
        start = time.time()
        mp_pose = mp.solutions.pose
        pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        # Initialize mediapipe drawing class - to draw the landmarks points.
        mp_drawing = mp.solutions.drawing_utils
        for path in self.images_paths:
            img = cv2.imread(path)
            image_in_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose_image.process(image_in_RGB)
            #deteced_landmarks = result.pose_landmarks.landmark
        end = time.time()
        duration = end - start
        return duration
    def measure_time_yolo(self):
        models = ['yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt']
        start = time.time()
        model=YOLO('yolov8n-pose.pt')
        for path in self.images_paths:
            results = model(source=path, show=False, save=False, verbose=False,max_det=1)
            #detected_landmarks = [landmark.cpu().numpy() for landmark in results[0].keypoints.xy][0]

        end=time.time()
        duration=end-start
        return duration
    def calculate_pcp(self, detected_skeletons,real_skeletons):
        limbs_count = len(detected_skeletons[0].limbs)
        total_percentage=0
        for detected_skeleton,real_skeleton in zip(detected_skeletons,real_skeletons):
            percentage=self.calculate_pcp_one_img(detected_skeleton,real_skeleton,limbs_count)
            total_percentage+=percentage
        return (percentage/self.img_count)*100

    def calculate_pcp_one_img(self,detected_skeleton,real_skeleton,limbs_count):
        correct = 0
        # cv2.namedWindow('tr', cv2.WINDOW_NORMAL)
        #
        # img=cv2.imread(real_skeleton.path)
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
            threshold=limb_length*0.5
            #print(dist_limb_end, dist_limb_start, limb_length,threshold)
            if dist_limb_end<threshold and dist_limb_start<threshold:
                correct+=1
        #         # Convert points to integer for drawing
        #     limb_start_detected = tuple(map(int, limb_start_detected))
        #     limb_end_detected = tuple(map(int, limb_end_detected))
        #     limb_start_real = tuple(map(int, limb_start_real))
        #     limb_end_real = tuple(map(int, limb_end_real))
        #     threshold = int(threshold)
        #
        #     wd = tuple(map(int, detected_skeleton.right_wrist))
        #     wr = tuple(map(int, real_skeleton.right_wrist))
        #     cv2.circle(img, wd, 5, (255, 255, 255), -1)  # Green circle at limb start point
        #     cv2.circle(img, wr, 5, (0, 255, 0), -1)  # Green circle at limb end point
        #
        #     # Draw the points
        #     # cv2.circle(img, limb_start_detected, 5, (0, 255, 0), -1)  # Green circle at limb start point
        #     # cv2.circle(img, limb_end_detected, 5, (0, 255, 0), -1)  # Green circle at limb end point
        #     #
        #     # cv2.circle(img, limb_start_real, 5, (0, 0, 255), -1)  # Green circle at limb start point
        #     # cv2.circle(img, limb_end_real, 5, (0, 0, 255), -1)  # Green circle at limb end point
        #
        #     # Draw circles around points with the threshold radius
        #     cv2.circle(img, limb_start_real, threshold, (255, 0, 0), 1)  # Blue circle around start point
        #     cv2.circle(img, limb_end_real, threshold, (255, 0, 0), 1)  # Blue circle around end point
        #     print(dist_limb_end, dist_limb_start, limb_length, threshold)
        #     print('cor', correct)
        #
        #     cv2.imshow('tr',img)
        #     cv2.waitKey(0)
        #
        #
        # print('pcp',correct/limbs_count)
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
        # TODO nezaratavat nulove !
        annotated_array = np.array(real_landmarks)
        detected_array = np.array(detected_landmarks)

        threshold=70

        non_zero_indices = np.all(detected_array != [0, 0], axis=1)
        annotated_non_zero = annotated_array[non_zero_indices]
        detected_non_zero = detected_array[non_zero_indices]

        # Calculate the Euclidean distance between each pair of corresponding landmarks
        distances = np.sqrt(np.sum(np.square(annotated_non_zero - detected_non_zero), axis=1))
        distances_in_threshold = np.sum(distances < threshold)
        return distances_in_threshold/len(non_zero_indices)

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

        return avg_distance