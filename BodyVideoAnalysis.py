import cv2
from BodyNeuronNetwork import NeuronNetworkManager
from YoloDetector import YoloDetector
from MediaPipeDetector import MediaPipeDetector
from YoloSkeleton import YoloSkeleton
from MediapipeSkeleton import MediaPipeSkeleton


class Analyser:
    def __init__(self, detector_type):
        self.detector_type = detector_type
        self.threshold = 0.55
        self.false_positive = 0
        self.false_negative = 0
        self.true_positive = 0
        self.true_negative = 0
        self.face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

        self.face_cascade_front = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.abnormal_ranges = [
            (5404, 5455, "cough"),
            (5488, 5540, "scratch"),
            (5741, 5845, 'scratch'),
            (6120, 6219, "cough"),
            (6581, 6674, "cough"),
            (7019, 7070, "cough"),
            (7895, 8003, "cough"),
            (8399, 8443, "cough"),
            (8516, 8701, "scratch"),
            (8959, 9036, "cough"),
            (9399, 9510, "cough"),
            (9673, 9886, 'scratch'),
            (10352, 10466, "cough"),
            (10666, 10766, "cough"),
            (11252, 11360, "cough"),
            (11412, 11515, "cough"),
            (12264, 12327, "cough"),
            (12563, 12616, "cough"),
            (12932, 13123, 'yawn'),
            (13393, 13463, "cough"),
            (13681, 13950, "scratch"),
            (14342, 14525, "scratch"),
            (15060, 15097, "cough"),
            (15193, 15293, "cough"),
            (15628, 15700, "cough"),
            (16269, 16343, "cough"),
            (16460, 16547, "scratch"),
            (17105, 17170, "cough"),
            (17287, 17330, 'yawn'),
        ]

    def setup_mediapipe_detector(self):
        self.mediapipe_detector = MediaPipeDetector()
        self.mediapipe_detector.setup_detector()

    def setup_yolo_detector(self):
        models = ['yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt']
        self.yolo_detector = YoloDetector(models[0])
        self.yolo_detector.setup()

    def setup_model(self):
        self.body_nn_manager = NeuronNetworkManager()
        self.body_nn_manager.load_model()

    def process_frame(self, frame):
        detected_landmarks = self.yolo_detector.get_landmarks(
            frame) if self.detector_type == 'yolo' else self.mediapipe_detector.get_landmarks(
            frame)
        if self.detector_type == 'yolo':
            skeleton = YoloSkeleton()
            skeleton.setup_from_detector(detected_landmarks)
            skeleton.create_body_nn_feature_vector()
        else:
            skeleton = MediaPipeSkeleton()
            skeleton.setup_from_detector(detected_landmarks)
            skeleton.create_body_nn_feature_vector()

        result = self.body_nn_manager.predict_img(skeleton.features_vector)
        if result > 0.55:
            # not normal
            return True
        else:
            return False

    def run(self):
        if self.detector_type == 'yolo':
            self.setup_yolo_detector()
        elif self.detector_type == 'mediapipe':
            self.setup_mediapipe_detector()
        # Load the video
        video_path = 'Videos/anomal.mp4'
        cap = cv2.VideoCapture(video_path)

        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        frame_count = 5400
        # frame_skip=0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5400)
        # Process each frame
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # if frame_count%frame_skip==0:
            if 1:
                result=self.process_frame(frame)
                self.check_detection(result, frame_count)

            # Add text to the frame indicating the frame number
            # cv2.putText(frame, f'Driver state: {result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #
            # # Display the frame
            # cv2.imshow('Video', frame)
            #
            # # Check for key press to exit
            # if cv2.waitKey(5) & 0xFF == ord('q'):
            #     break

            frame_count += 1

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
        self.calculate_sensitivity_specificity()


    def calculate_sensitivity_specificity(self):
        total = self.true_negative + self.true_positive + self.false_negative + self.false_positive
        self.sensitivity = self.true_positive / (self.true_positive + self.false_negative)
        self.specificity = self.true_negative / (self.true_negative + self.false_positive)
        self.accuracy = (self.true_positive + self.true_negative) / total
        print("Specificity:", self.specificity)
        print("Sensitivity:", self.sensitivity)
        print("Accuracy:", self.accuracy)

    def check_detection(self, abnormal, frame_count):
        abnormal_real_state = False
        for start, end, action in self.abnormal_ranges:
            if start <= frame_count <= end:
                abnormal_real_state = True

        if abnormal:
            if abnormal_real_state:
                self.true_positive += 1
            else:
                self.false_positive += 1
        else:
            if abnormal_real_state:
                self.false_negative += 1

            else:
                self.true_negative += 1