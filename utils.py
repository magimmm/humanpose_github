import json
import os
import sys

import numpy as np

from MediapipeSkeleton import MediaPipeSkeleton
from YoloSkeleton import YoloSkeleton

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
            skeleton = MediaPipeSkeleton()
        elif model == 'yolo':
            skeleton = YoloSkeleton()
        skeleton.setup_from_annotation_file(annotated_landmarks, images_paths[i])
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

