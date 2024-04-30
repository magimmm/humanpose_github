from MediapipeSkeleton import MediaPipeSkeleton
from YoloSkeleton import YoloSkeleton
from YoloDetector import YoloDetector
from MediaPipeDetector import MediaPipeDetector


class SkeletonDetector:
    """
    A class for detecting skeletons using both MediaPipe and YOLO detectors.
    """
    def __init__(self):
        self.detected_skeletons_mediape = []
        self.detected_skeletons_yolo = []

    def setup_yolo_detector(self):
        """
        Sets up the YOLO detector with predefined models.
        """
        models = ['yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt']
        self.yolo_detector = YoloDetector(models[0])
        self.yolo_detector.setup()

    def setup_mediapipe_detector(self):
        """
        Sets up the MediaPipe detector.
        """
        self.mediapipe_detector = MediaPipeDetector()
        self.mediapipe_detector.setup_detector()

    def detect_and_create_skeletons(self, images_paths):
        """
        Detects skeletons in the given images using both MediaPipe and YOLO detectors.

        Args:
            images_paths (list): List of file paths of images to detect skeletons in.
        """
        self.setup_yolo_detector()
        self.setup_mediapipe_detector()
        for path in images_paths:
            yolo_detected_landmarks = self.yolo_detector.get_landmarks(path)
            yolo_skeleton = YoloSkeleton(path)
            yolo_skeleton.setup_from_detector(yolo_detected_landmarks)
            self.detected_skeletons_yolo.append(yolo_skeleton)

            mediapipe_detected_landmarks = self.mediapipe_detector.get_landmarks(path)
            mediapipe_skeleton = MediaPipeSkeleton()
            mediapipe_skeleton.setup_from_detector(mediapipe_detected_landmarks)
            self.detected_skeletons_mediape.append(mediapipe_skeleton)
