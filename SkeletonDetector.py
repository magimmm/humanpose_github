from MediapipeSkeleton import MediaPipeSkeleton
from YoloSkeleton import YoloSkeleton
from YoloDetector import YoloDetector
from MediaPipeDetector import MediaPipeDetector

class SkeletonDetector():
    def __init__(self):
        self.detected_skeletons_mediape=[]
        self.detected_skeletons_yolo=[]
        self.time_yolo = 0
        self.time_mediapipe = 0

    def setup_yolo_detector(self):
        # TODO yolo rozne modely
        models = ['yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt']
        self.yolo_detector = YoloDetector(models[0])
        self.yolo_detector.setup()
    def setup_mediapipe_detector(self):
        self.mediapipe_detector = MediaPipeDetector()
        self.mediapipe_detector.setup_detector()

    def detect_and_create_skeletons(self, images_paths):
        self.setup_yolo_detector()
        self.setup_mediapipe_detector()
        for path in images_paths:
            yolo_detected_landmarks, yolo_time_detection = self.yolo_detector.get_landmarks(path)
            self.time_yolo=self.time_yolo+yolo_time_detection
            yolo_skeleton=YoloSkeleton()
            yolo_skeleton.setup_from_detector(yolo_detected_landmarks)
            self.detected_skeletons_yolo.append(yolo_skeleton)

            mediapipe_detected_landmarks, medipipe_time_detection = self.mediapipe_detector.get_landmarks(path)
            self.time_mediapipe=self.time_mediapipe+medipipe_time_detection
            mediapipe_skeleton=MediaPipeSkeleton()
            mediapipe_skeleton.setup_from_detector(mediapipe_detected_landmarks)
            self.detected_skeletons_mediape.append(mediapipe_skeleton)




