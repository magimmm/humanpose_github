import cv2
from BodyNeuronNetwork import NeuronNetworkManager
from utils import create_skeletons_from_annotations,get_image_files_in_folder

class SkeletonsLoader():
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
    def load_images_paths(self):
        self.normal_images_paths, self.normal_images_filenames = get_image_files_in_folder(self.folder_images_normal)
        self.abnormal_images_paths, self.abnormal_images_filenames = get_image_files_in_folder(self.folder_images_abnormal)

    def create_skeletons(self):
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


    def train_body_neuron_network(self):
        self.body_nn_manager=NeuronNetworkManager()
        self.body_nn_manager.train_model(normal_skeletons=self.skeletons_yolo_normal, abnormal_skeletons=self.skeletons_yolo_abnormal,model='yolo')

    def view_skeletons_annotated(self, model):
        if model == 'mediapipe':
            selected_skeletons = self.skeletons_mediapipe
        elif model == 'yolo':
            selected_skeletons = self.skeletons_yolo

        for skeleton in selected_skeletons:
            img = cv2.imread(skeleton.path)
            for ind,landmark in enumerate(skeleton.all_landmarks):
                if ind== 1:
                    color = (255, 255, 255)
                else:
                    color = (255, 120, 0)
                cv2.circle(img, (int(landmark[0]), int(landmark[1])), 4, color, 1, 1)
            cv2.namedWindow('annotated_landmarks', cv2.WINDOW_NORMAL)
            cv2.imshow('annotated_landmarks', img)
            cv2.waitKey(0)

