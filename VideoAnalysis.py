import cv2
from BodyNeuronNetwork import NeuronNetworkManager
from YoloDetector import YoloDetector
from MediaPipeDetector import MediaPipeDetector
from YoloSkeleton import YoloSkeleton
from MediapipeSkeleton import MediaPipeSkeleton
class Analyser:
    def __init__(self,detecotr_type):
        self.detector_type=detecotr_type
        self.threshold=0.55

    def setup_mediapipe_detector(self):
        self.mediapipe_detector = MediaPipeDetector()
        self.mediapipe_detector.setup_detector()

    def setup_yolo_detector(self):
        # TODO yolo rozne modely
        models=['yolov8n-pose.pt','yolov8s-pose.pt','yolov8m-pose.pt','yolov8l-pose.pt','yolov8x-pose.pt']
        self.yolo_detector = YoloDetector(models[0])
        self.yolo_detector.setup()

    def setup_model(self):
        self.body_nn_manager = NeuronNetworkManager()
        self.body_nn_manager.load_model()

    def process_frame(self,frame):
        detected_landmarks, time_detection = self.yolo_detector.get_landmarks(frame) if self.detector_type == 'yolo' else self.mediapipe_detector.get_landmarks(
            frame)
        if self.detector_type=='yolo':
            skeleton=YoloSkeleton()
            skeleton.setup_from_detector(detected_landmarks)
            skeleton.create_body_nn_feature_vector()


        result= self.body_nn_manager.predict_img(skeleton.features_vector)
        if result<0.55:
            return 'Normal'
        else:
            return 'Not Normal'

    def run(self):
        if self.detector_type=='yolo':
            self.setup_yolo_detector()
        elif self.detector_type=='mediapipe':
            self.setup_mediapipe_detector()
        # Load the video
        video_path = 'Videos/anomal.mp4'
        cap = cv2.VideoCapture(video_path)

        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        frame_number = 0
        frame_skip = 5  # Process every nth frame

        # Process each frame
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Display every 5th frame
            if frame_number % frame_skip == 0:
                result=self.process_frame(frame)

                # Add text to the frame indicating the frame number
                cv2.putText(frame, f'Driver state: {result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Display the frame
                cv2.imshow('Video', frame)

                # Check for key press to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            frame_number += 1

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
