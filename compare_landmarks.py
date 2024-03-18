import cv2
from MediapipeSkeleton import MediaPipeSkeleton
from YoloSkeleton import YoloSkeleton
from human_pose_landmarks_detectors import MediaPipeDetector, YoloDetector
from BodyNeuronNetwork import NeuronNetworkManager
from utils import calculate_mse,calculate_euclidean_distance,create_skeletons_from_annotations,get_image_files_in_folder

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

    def train_body_neuron_network(self):
        self.body_nn_manager=NeuronNetworkManager()
        self.body_nn_manager.train_model(normal_skeletons=self.skeletons_yolo_normal, abnormal_skeletons=self.skeletons_yolo_abnormal,model='yolo')

    def load_images_paths(self):
        self.normal_images_paths, self.normal_images_filenames = get_image_files_in_folder(self.folder_images_normal)
        self.abnormal_images_paths, self.abnormal_images_filenames = get_image_files_in_folder(self.folder_images_abnormal)

    def create_skeletons(self):
        skeletons_normal_yolo = create_skeletons_from_annotations(self.path_annotation_file_normal,
                                                                  self.normal_images_paths,
                                                                  self.normal_images_filenames, model='yolo')

        skeletons_abnormal_yolo = create_skeletons_from_annotations(self.path_annotation_file_abnormal, self.abnormal_images_paths,self.abnormal_images_filenames, model='yolo')

        self.skeletons_yolo_abnormal=skeletons_abnormal_yolo
        self.skeletons_yolo_normal=skeletons_normal_yolo
        self.skeletons_yolo = skeletons_normal_yolo + skeletons_abnormal_yolo

        skeletons_normal_mediapipe = create_skeletons_from_annotations(self.path_annotation_file_normal,
                                                                       self.normal_images_paths,
                                                                       self.normal_images_filenames, model = 'mediapipe')

        skeletons_abnormal_mediapipe = create_skeletons_from_annotations(self.path_annotation_file_abnormal, self.abnormal_images_paths,self.abnormal_images_filenames,model='mediapipe')
        self.skeletons_mediapipe_normal=skeletons_normal_mediapipe
        self.skeletons_mediapipe_abnormal=skeletons_abnormal_mediapipe
        self.skeletons_mediapipe = skeletons_normal_mediapipe + skeletons_abnormal_mediapipe

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

    def run_and_compare(self):
        #self.run_detector('yolo')
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
            raise ValueError("Invalid detector model_type. Choose 'yolo' or 'mediapipe'.")

        total_mse = 0
        total_euclid = 0
        time_total= 0
        for annotated_skeleton in skeletons:
            detected_landmarks,time_detection = self.yolo_detector.get_landmarks(annotated_skeleton.path) if detector_type == 'yolo' else self.mediapipe_detector.get_landmarks(annotated_skeleton.path)

            # img = cv2.imread(annotated_skeleton.path)
            # for ind_l,landmark in enumerate(detected_landmarks):
            #     if ind_l==1:
            #         color=(255, 255, 255)
            #     else:
            #         color=(255, 120, 0)
            #     cv2.circle(img, (int(landmark[0]), int(landmark[1])), 4, color, 1, 1)
            # cv2.namedWindow('annotated_landmarks', cv2.WINDOW_NORMAL)
            # cv2.imshow('detected_landmarks', img)
            # cv2.waitKey(0)

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
landmarktester.load_images_paths()
landmarktester.create_skeletons()#from annotations
# TODO uncomment in case of wanting to train the model again
#landmarktester.train_body_neuron_network()
# TODO odkomentovat ak chceme vidiet ako sa zobrazuju naanotovane body
# self.view_skeletons_annotated('mediapipe')
# self.view_skeletons_annotated('yolo')

landmarktester.run_and_compare()
# yolo  5.3566
# Mean squared error: 746.1339784812138
# Mean Euclidean distance: 29.75326993896032
# Time:  13.852840662002563
#
# mediPIPE
# Mean squared error: 55607.92410538614
# Mean Euclidean distance: 267.73781403977426
# Time:  8.255919933319092