import json
import os
import sys

import mediapipe as mp
import numpy as np



import cv2
from MediapipeSkeleton import MediaPipeSkeleton
from YoloSkeleton import YoloSkeleton
from human_pose_landmarks_detectors import MediaPipeDetector, YoloDetector


def calculate_euclidean_distance(detected_landmarks, annotated_landmarks):
    # Convert landmarks to numpy arrays for easier computation
    annotated_array = np.array(annotated_landmarks)
    detected_array = np.array(detected_landmarks)

    # Filter out points with coordinates (0, 0)
    valid_indices = np.all(detected_array != 0, axis=1)
    #print(valid_indices)
    annotated_array = annotated_array[valid_indices]
    detected_array = detected_array[valid_indices]

    # Calculate the Euclidean distance between each pair of corresponding landmarks
    distances = np.sqrt(np.sum(np.square(annotated_array - detected_array), axis=1))

    # Calculate the average Euclidean distance
    avg_distance = np.mean(distances)
    #print(avg_distance)

    return avg_distance


def calculate_mse(detected_landmarks, annotated_landmarks):
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
def create_skeletons_from_annotations(annotation_file_path, images_paths,
                                      images_filenames, model
                                      ):
    skeletons_images = []
    for (i, img_filename) in enumerate(images_filenames):
        annotated_landmarks = get_landmarks_from_annotation_file(img_filename, annotation_file_path)
        # body zistene z anotovaneho suboru zoberie a vytvori z nich instanciu triedy
        if model == 'mediapipe':
            skeleton = MediaPipeSkeleton(annotated_landmarks, images_paths[i])
        elif model == 'yolo':
            skeleton = YoloSkeleton(annotated_landmarks, images_paths[i])
        skeletons_images.append(skeleton)
    return skeletons_images


def get_image_files_in_folder(folder_path):
    images_paths = []
    image_filenames = []
    for file_name in os.listdir(folder_path):
        # Check if the file is an image (you can add more extensions if needed)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            images_paths.append(os.path.join(folder_path, file_name))
            image_filenames.append(file_name)
    return images_paths, image_filenames


def get_landmarks_from_annotation_file(img_filename, annotation_file_path):
    # Load JSON data from a file
    with open(annotation_file_path, 'r') as file:
        data = json.load(file)

    annotations = data['annotations']
    images = data['images']
    for i in images:
        if i['file_name'] == img_filename:
            id_img = i['id']
            for a in annotations:
                if id_img == a['id']:
                    return a['keypoints']

    print("Image not found!")
    sys.exit(1)


class LandmarkTester():
    def __init__(self, path_images_normal, path_images_abnormal, path_annotation_file_normal,
                 path_annotation_file_abnormal):
        self.mediapipe_detector = None
        self.skeletons_mediapipe = None
        self.skeletons_yolo = None
        self.abnormal_images_filenames = None
        self.abnormal_images_paths = None
        self.normal_images_filenames = None
        self.normal_images_paths = None
        self.folder_images_normal = path_images_normal
        self.folder_images_abnormal = path_images_abnormal
        self.path_annotation_file_normal = path_annotation_file_normal
        self.path_annotation_file_abnormal = path_annotation_file_abnormal
        self.time_yolo=0
        self.time_mediapipe=0
        # self.all_images_filemames = self.normal_images_filenames + self.abnormal_images_filenames
        # self.all_images_paths = self.normal_images_paths + self.abnormal_images_paths

    def test(self):
        self.load_images_paths()
        self.create_skeletons()
        # TODO prememenovat tuto funkciu
        # TODO odkomentovat ak chceme vidiet ako sa zobrazuju naanotovane body
        #self.view_skeletons_annotated('mediapipe')
        #self.view_skeletons_annotated('yolo')
        self.run_and_compare()


    def load_images_paths(self):
        self.normal_images_paths, self.normal_images_filenames = get_image_files_in_folder(self.folder_images_normal)
        self.abnormal_images_paths, self.abnormal_images_filenames = get_image_files_in_folder(self.folder_images_abnormal)

    def create_skeletons(self):
        skeletons_normal_yolo = create_skeletons_from_annotations(self.path_annotation_file_normal,
                                                                  self.normal_images_paths,
                                                                  self.normal_images_filenames, model='yolo')

        skeletons_abnormal_yolo = create_skeletons_from_annotations(self.path_annotation_file_abnormal, self.abnormal_images_paths,self.abnormal_images_filenames, model='yolo')

        self.skeletons_yolo = skeletons_normal_yolo + skeletons_abnormal_yolo

        skeletons_normal_mediapipe = create_skeletons_from_annotations(self.path_annotation_file_normal,
                                                                       self.normal_images_paths,
                                                                       self.normal_images_filenames, model = 'mediapipe')

        skeletons_abnormal_mediapipe = create_skeletons_from_annotations(self.path_annotation_file_abnormal, self.abnormal_images_paths,self.abnormal_images_filenames,model='mediapipe')
        self.skeletons_mediapipe = skeletons_normal_mediapipe + skeletons_abnormal_mediapipe

    def view_skeletons_annotated(self, model):
        if model == 'mediapipe':
            selected_skeletons = self.skeletons_mediapipe
        elif model == 'yolo':
            selected_skeletons = self.skeletons_yolo

        for skeleton in selected_skeletons:
            img = cv2.imread(skeleton.path)
            for landmark in skeleton.all_landmarks:
                cv2.circle(img, (int(landmark[0]), int(landmark[1])), 4, (255, 120, 0), 1, 1)
            cv2.namedWindow('annotated_landmarks', cv2.WINDOW_NORMAL)
            cv2.imshow('annotated_landmarks', img)
            cv2.waitKey(0)

    def run_and_compare(self):
        self.run_detector('yolo')
        self.run_detector('mediapipe')


    def setup_mediapipe_detector(self):
        self.mediapipe_detector = MediaPipeDetector()
        self.mediapipe_detector.setup_detector()

    def setup_yolo_detector(self):
        # TODO yolo rozne modely
        models=['yolov8n-pose.pt','yolov8s-pose.pt','yolov8m-pose.pt','yolov8l-pose.pt','yolov8x-pose.pt']
        self.yolo_detector = YoloDetector(models[0])
        self.yolo_detector.setup()

    def run_detector(self, detector_type):
        if detector_type == 'yolo':
            print('YOLO results')
            self.setup_yolo_detector()
            skeletons = self.skeletons_yolo
        elif detector_type == 'mediapipe':
            print('MediaPipe results')
            self.setup_mediapipe_detector()
            skeletons = self.skeletons_mediapipe
        else:
            raise ValueError("Invalid detector type. Choose 'yolo' or 'mediapipe'.")

        total_mse = 0
        total_euclid = 0
        time_total= 0
        for annotated_skeleton in skeletons:
            detected_landmarks,time_detection = self.yolo_detector.get_landmarks(annotated_skeleton.path) if detector_type == 'yolo' else self.mediapipe_detector.get_landmarks(annotated_skeleton.path)
            euclid = calculate_euclidean_distance(detected_landmarks, annotated_skeleton.all_landmarks)
            mse = calculate_mse(detected_landmarks,annotated_skeleton.all_landmarks)
            total_mse += mse
            total_euclid += euclid
            time_total+=time_detection


        print('Mean squared error:', total_mse / len(skeletons))
        print('Mean Euclidean distance:', total_euclid / len(skeletons))
        print('Time: ',time_total)
        print()

    # Usage:
    # self.run_detector('yolo')
    # self.run_detector('mediapipe')


path_normal_images = 'Photos/normal_select'
path_abnormal_images = 'Photos/abnormal_select'
landmarktester = LandmarkTester(path_normal_images, path_images_abnormal=path_abnormal_images,
                                path_annotation_file_normal='annotation/normal_keypoints_photos.json',
                                path_annotation_file_abnormal='annotation/abnormal_keypoints_photos.json')
landmarktester.test()
