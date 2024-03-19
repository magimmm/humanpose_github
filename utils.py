import json
import os
import sys
import math

from MediapipeSkeleton import MediaPipeSkeleton
from YoloSkeleton import YoloSkeleton


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

def calculate_distance(point_one,point_two):
    x1,y1=point_one[0],point_one[1]
    x2,y2=point_two[0],point_two[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


