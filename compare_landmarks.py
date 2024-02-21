import json
import os
import mediapipe as mp
import numpy as np



import cv2
from MediapipeSkeleton import MediaPipeSkeleton
from YoloSkeleton import YoloSkeleton
from human_pose_landmarks_detectors import MediaPipeDetector, YoloDetector


def calculate_euclidean_distance(annotated_landmarks, detected_landmarks):
    # Convert landmarks to numpy arrays for easier computation
    annotated_array = np.array(annotated_landmarks)
    detected_array = np.array(detected_landmarks)

    # Calculate the Euclidean distance between each pair of corresponding landmarks
    distances = np.sqrt(np.sum(np.square(annotated_array - detected_array), axis=1))

    # Calculate the average Euclidean distance
    avg_distance = np.mean(distances)

    return avg_distance

def calculate_mse(annotated_landmarks, detected_landmarks):
    # Convert landmarks to numpy arrays for easier computation
    annotated_array = np.array(annotated_landmarks)
    detected_array = np.array(detected_landmarks)

    # Calculate squared differences between corresponding coordinates
    squared_diffs = np.square(annotated_array - detected_array)

    # Calculate the mean squared error
    mse = np.mean(squared_diffs)

    return mse

def create_skeletons_from_annotations(annotation_file_path, images_paths,
                                      images_filenames, model='mediapipe'
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

    images = data['images']
    for i in images:
        if i['file_name'] == img_filename:
            id_img = i['id']

    annotations = data['annotations']
    for a in annotations:
        if id_img == a['id']:
            return a['keypoints']

    return "Something went wrong"


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
        # self.all_images_filemames = self.normal_images_filenames + self.abnormal_images_filenames
        # self.all_images_paths = self.normal_images_paths + self.abnormal_images_paths

    def test(self):
        self.load_images_paths()
        self.create_skeletons()
        self.view_skeletons_annotated('yolo')

    def load_images_paths(self):
        self.normal_images_paths, self.normal_images_filenames = get_image_files_in_folder(self.folder_images_normal)
        # TODO odkomentovat ked sa pridaju dalsie obrazky
        # self.abnormal_images_paths, self.abnormal_images_filenames = get_image_files_in_folder(self.folder_images_abnormal)

    def create_skeletons(self):
        skeletons_normal_yolo = create_skeletons_from_annotations(self.path_annotation_file_normal,
                                                                  self.normal_images_paths,
                                                                  self.normal_images_filenames, model='yolo')

        # TODO odkomentovat ked sa pridaju dalsie obrazky
        skeletons_abnormal_yolo = []
        # skeletons_abnormal_yolo = create_skeletons_from_annotations(self.path_annotation_file_abnormal, self.normal_images_paths,self.normal_images_filenames, model='yolo')

        self.skeletons_yolo = skeletons_normal_yolo + skeletons_abnormal_yolo

        skeletons_normal_mediapipe = create_skeletons_from_annotations(self.path_annotation_file_normal,
                                                                       self.normal_images_paths,
                                                                       self.normal_images_filenames, model='mediapipe')

        # TODO odkomentovat ked sa pridaju dalsie obrazky
        skeletons_abnormal_mediapipe = []
        # skeletons_abnormal_mediapipe = create_skeletons_from_annotations(self.path_annotation_file_abnormal,self.normal_images_paths,self.normal_images_filenames,model='mediapipe')
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
            cv2.imshow('annotated_landmarks', img)
            cv2.waitKey(0)

    def run_and_compare(self):
        self.run_media_pipe()
        self.run_yolo()
        #TODO yolo rozne modely

    def run_media_pipe(self):
        self.setup_mediapipe_detector()
        total_mse=0
        total_euclid=0
        for annotated_skeleton in self.skeletons_mediapipe:
            detected_landmarks = self.mediapipe_detector.getlandmarks()
            euclid = calculate_euclidean_distance(detected_landmarks,annotated_skeleton.all_landmarks)
            mse =calculate_mse(annotated_skeleton.all_landmarks,detected_landmarks)
            total_mse+=mse
            total_euclid+=euclid
            #todo podel poctom bodov
        print(total_mse/len(self.skeletons_mediapipe))
        print(total_euclid/len(self.skeletons_mediapipe))

    def setup_mediapipe_detector(self):
        self.mediapipe_detector=MediaPipeDetector()
        self.mediapipe_detector.setup_detector()

    def setup_yolo_detector(self):
        models=['yolov8n-pose.pt','yolov8s-pose.pt','yolov8m-pose.pt','yolov8l-pose.pt','yolov8x-pose.pt']
        self.yolo_detector = YoloDetector(models[0])

    def run_yolo(self):
        self.setup_yolo_detector()
        total_mse = 0
        total_euclid = 0
        for annotated_skeleton in self.skeletons_yolo:
            detected_landmarks = self.yolo_detector.getlandmarks()
            euclid = calculate_euclidean_distance(detected_landmarks, annotated_skeleton.all_landmarks)
            mse = calculate_mse(annotated_skeleton.all_landmarks, detected_landmarks)
            total_mse += mse
            total_euclid += euclid
        print(total_mse / len(self.skeletons_yolo))
        print(total_euclid / len(self.skeletons_yolo))

path_normal_images = 'Photos/normal_select'
landmarktester = LandmarkTester('Photos/normal_select', path_images_abnormal='idk',
                                path_annotation_file_normal='annotation/person_keypoints_default.json',
                                path_annotation_file_abnormal='somepath')
landmarktester.test()
