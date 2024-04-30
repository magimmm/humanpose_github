import json
import os
import sys
import math
from MediapipeSkeleton import MediaPipeSkeleton
from YoloSkeleton import YoloSkeleton


def create_skeletons_from_annotations(annotation_file_path, images_paths,
                                      images_filenames, model
                                      ):
    """
    Create skeletons from annotation files.

    Args:
        annotation_file_path (str): Path to the annotation file.
        images_paths (list): List of paths to the images.
        images_filenames (list): List of image filenames.
        model (str): Model type ('mediapipe' or 'yolo').

    Returns:
        list: List of Skeleton objects.
    """
    skeletons_images = []
    for (i, img_filename) in enumerate(images_filenames):
        annotated_landmarks = get_landmarks_from_annotation_file(img_filename, annotation_file_path)
        if model == 'mediapipe':
            skeleton = MediaPipeSkeleton()
        elif model == 'yolo':
            skeleton = YoloSkeleton()
        skeleton.setup_from_annotation_file(annotated_landmarks, images_paths[i])
        skeletons_images.append(skeleton)
    return skeletons_images


def get_image_files_in_folder(folder_path):
    """
    Get paths and filenames of images in a folder.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        tuple: Tuple containing a list of image paths and a list of image filenames.
    """
    images_paths = []
    image_filenames = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            images_paths.append(os.path.join(folder_path, file_name))
            image_filenames.append(file_name)
    return images_paths, image_filenames


def get_landmarks_from_annotation_file(img_filename, annotation_file_path):
    """
    Get landmarks from an annotation file.

    Args:
        img_filename (str): Filename of the image.
        annotation_file_path (str): Path to the annotation file.

    Returns:
        list: List of detected landmarks.
    """
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


def calculate_distance(point_one, point_two):
    """
    Calculate the Euclidean distance between two points.

    Args:
        point_one (tuple): Coordinates of the first point (x1, y1).
        point_two (tuple): Coordinates of the second point (x2, y2).

    Returns:
        float: Euclidean distance between the two points.
    """
    x1, y1 = point_one[0], point_one[1]
    x2, y2 = point_two[0], point_two[1]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
