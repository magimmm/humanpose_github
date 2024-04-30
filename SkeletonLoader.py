import cv2
from utils import create_skeletons_from_annotations, get_image_files_in_folder


class SkeletonsLoader:
    """
    A class for loading and visualizing annotated skeletons from image data.
    """
    def __init__(self, path_images_normal, path_images_abnormal, path_annotation_file_normal,
                 path_annotation_file_abnormal):
        """
        Initializes the SkeletonsLoader with paths to image folders and annotation files.

        Args:
            path_images_normal (str): Path to the folder containing normal images.
            path_images_abnormal (str): Path to the folder containing abnormal images.
            path_annotation_file_normal (str): Path to the annotation file for normal images.
            path_annotation_file_abnormal (str): Path to the annotation file for abnormal images.
        """
        self.skeletons_mediapipe_abnormal = None
        self.skeletons_mediapipe_normal = None
        self.skeletons_yolo_abnormal = None
        self.skeletons_yolo_normal = None
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

    def load_images_paths(self):
        """
        Loads the paths and filenames of images from the specified folders.
        """
        self.normal_images_paths, self.normal_images_filenames = get_image_files_in_folder(self.folder_images_normal)
        self.abnormal_images_paths, self.abnormal_images_filenames = get_image_files_in_folder(
            self.folder_images_abnormal)

    def create_skeletons(self):
        """
        Creates skeletons from the annotation files and associates them with the corresponding images.
        """
        self.load_images_paths()
        skeletons_normal_yolo = create_skeletons_from_annotations(self.path_annotation_file_normal,
                                                                  self.normal_images_paths,
                                                                  self.normal_images_filenames, model='yolo')

        skeletons_abnormal_yolo = create_skeletons_from_annotations(self.path_annotation_file_abnormal,
                                                                    self.abnormal_images_paths,
                                                                    self.abnormal_images_filenames, model='yolo')

        self.skeletons_yolo_abnormal = skeletons_abnormal_yolo
        self.skeletons_yolo_normal = skeletons_normal_yolo
        self.skeletons_yolo = skeletons_normal_yolo + skeletons_abnormal_yolo

        skeletons_normal_mediapipe = create_skeletons_from_annotations(self.path_annotation_file_normal,
                                                                       self.normal_images_paths,
                                                                       self.normal_images_filenames, model='mediapipe')

        skeletons_abnormal_mediapipe = create_skeletons_from_annotations(self.path_annotation_file_abnormal,
                                                                         self.abnormal_images_paths,
                                                                         self.abnormal_images_filenames,
                                                                         model='mediapipe')
        self.skeletons_mediapipe_normal = skeletons_normal_mediapipe
        self.skeletons_mediapipe_abnormal = skeletons_abnormal_mediapipe
        self.skeletons_mediapipe = skeletons_normal_mediapipe + skeletons_abnormal_mediapipe

    def view_skeletons_annotated(self, model):
        """
        Visualizes annotated skeletons on their respective images.

        Args:
            model (str): The model used for annotations ('mediapipe' or 'yolo').
        """
        if model == 'mediapipe':
            selected_skeletons = self.skeletons_mediapipe
        elif model == 'yolo':
            selected_skeletons = self.skeletons_yolo

        for skeleton in selected_skeletons:
            img = cv2.imread(skeleton.path)
            for ind, landmark in enumerate(skeleton.all_landmarks):
                if ind == 1:
                    color = (255, 255, 255)
                else:
                    color = (255, 120, 0)
                cv2.circle(img, (int(landmark[0]), int(landmark[1])), 4, color, 1, 1)
            cv2.namedWindow('annotated_landmarks', cv2.WINDOW_NORMAL)
            cv2.imshow('annotated_landmarks', img)
            cv2.waitKey(0)
