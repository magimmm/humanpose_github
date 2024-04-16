import cv2
from BodyNeuronNetwork import NeuronNetworkManager
from YoloDetector import YoloDetector
from MediaPipeDetector import MediaPipeDetector
from YoloSkeleton import YoloSkeleton
from MediapipeSkeleton import MediaPipeSkeleton
import mediapipe as mp
from cnn import CNN
class FaceAnalyser:
    def __init__(self):
        self.last_states = []
        self.threshold=0.55
        self.false_positive=0
        self.false_negative=0
        self.true_positive=0
        self.true_negative=0
        self.mp_count=0
        self.opencvcount=0
        self.driver_state_abnormal=False
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

        self.false_positive_body = 0
        self.false_negative_body = 0
        self.true_positive_body = 0
        self.true_negative_body = 0

        self.y_min = - 40
        self.y_max = 10
        self.x_min = - 40
        self.x_max = 0
        self.detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (0, 0))
        self.abnormal_ranges = [
            (5404, 5455, "cough"),
            (5495, 5540, "scratch"),
            (5743, 5845, "scratch"),
            (6125, 6218, "cough"),
            (6585, 6670, "cough"),
            (7020, 7070, "cough"),
            (7902, 8003, "cough"),
            (8403, 8443, "cough"),
            (8520, 8701, "scratch"),
            (8964, 9036, "cough"),
            (9406, 9510, "cough"),
            (9674, 9886, "scratch"),
            (10353, 10466, "cough"),
            (10666, 10766, "cough"),
            (11256, 11360, "cough"),
            (11412, 11515, "cough"),
            (12271, 12327, "cough"),
            (12563, 12616, "cough"),
            (12937, 13123, "yawn"),
            (13399, 13463, "cough"),
            (13684, 13950, "scratch"),
            (14347, 14525, "scratch"),
            (15060, 15097, "cough"),
            (15197, 15292, "cough"),
            (15636, 15700, "cough"),
            (16275, 16340, "cough"),
            (16463, 16547, "scratch"),
            (17105, 17170, "cough"),
            (17290, 17331, "yawn")
        ]

        self.abnormal_ranges_body = [
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


    def setup_yolo_detector(self):
        # TODO yolo rozne modely
        models=['yolov8n-pose.pt','yolov8s-pose.pt','yolov8m-pose.pt','yolov8l-pose.pt','yolov8x-pose.pt']
        self.yolo_detector = YoloDetector(models[0])
        self.yolo_detector.setup()

    def setup_model(self):
        self.cnn = CNN()
        self.cnn.load_model("cnn_model_aug.pth")

    def find_face_mp(self, frame):
        # Convert the frame to RGB for processing with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        results = self.face_detection.process(frame_rgb)

        # Check if any faces were detected
        if results.detections:
            return True
        else:
            return False

    def find_face_opencv(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect frontal faces
        frontal_faces = self.frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Detect profile faces
        profile_faces = self.profile_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Return True if any face (frontal or profile) is detected
        if len(frontal_faces) > 0 or len(profile_faces) > 0:
            return True
        else:
            return False
    def process_frame(self,frame):
        yolo_detected_landmarks = self.yolo_detector.get_landmarks(frame)
        yolo_skeleton = YoloSkeleton()
        yolo_skeleton.setup_from_detector(yolo_detected_landmarks)
        yolo_skeleton.find_face()
        crop_region = frame[yolo_skeleton.y_min + self.y_min:yolo_skeleton.y_max + self.y_max,
                      yolo_skeleton.x_min + self.x_min:yolo_skeleton.x_max + self.x_max]

        # save_path = os.path.join("Photos", "cutouts", "normal2", filename)
        # cv2.imwrite(save_path, crop_region)


        result= self.cnn.predict(crop_region)



        return result

    def run(self):
        print('Run')
        self.setup_yolo_detector()
        self.setup_model()

        print('Video')
        # Load the video
        video_path = 'Videos/anomal.mp4'
        cap = cv2.VideoCapture(video_path)

        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        frame_count=5400
        # frame_skip=0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5400)
        # Process each frame
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # if frame_count%frame_skip==0:
            if 1:
                if self.find_face_mp(frame):
                    self.mp_count+=1
                if self.find_face_opencv(frame):
                    self.opencvcount+=1
    #---------------------------
                # result=self.process_frame(frame)
                # print(result, frame_count)
                #
                # if result==0:
                #     state='abnormal'
                # elif result==1:
                #     state='left'
                # elif result==2:
                #     state='normal'
                # else:
                #     state='right'
                #
                #
                # if result>0:
                #     one_frame_state_abnormal=False
                # else:
                #     one_frame_state_abnormal=True
                #
                # self.last_states.append(one_frame_state_abnormal)
                #
                # # If the list of last states exceeds 5, remove the oldest state
                # if len(self.last_states) > 5:
                #     self.last_states.pop(0)
                #
                # if len(set(self.last_states)) == 1:
                #     # If all states are the same, return this common state
                #     self.driver_state_abnormal=one_frame_state_abnormal
                #
                # self.check_detection(self.driver_state_abnormal, frame_count)
                #
                #
                #
                # cv2.putText(frame, f'Driver state: {state}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                #
                #

#----------------------------------




                # Display the frame
                # cv2.imshow('Video', frame)
                # if cv2.waitKey(5) & 0xFF == ord('q'):
                #     break

                # result=self.process_frame(frame)


                # Add text to the frame indicating the frame number
                # cv2.putText(frame, f'Driver state: {result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                #
                # # Display the frame
                # cv2.imshow('Video', frame)
                #
                # # Check for key press to exit
                # if cv2.waitKey(5) & 0xFF == ord('q'):
                #     break

            frame_count+=1


        print(frame_count-5400)
        print('mp',self.mp_count, self.mp_count/frame_count-5400)
        print('opencv',self.opencvcount, self.opencvcount/frame_count-5400)
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
        print('--body')

        total = self.true_negative_body + self.true_positive_body + self.false_negative_body + self.false_positive_body
        self.sensitivity2 = self.true_positive_body / (self.true_positive_body + self.false_negative_body)
        self.specificity2 = self.true_negative_body / (self.true_negative_body + self.false_positive_body)
        self.accuracy2 = (self.true_positive_body + self.true_negative_body) / total
        print("Specificity:", self.specificity2)
        print("Sensitivity:", self.sensitivity2)
        print("Accuracy:", self.accuracy2)

    def check_detection(self,abnormal,frame_count):
        abnormal_real_state=False
        for start, end, action in self.abnormal_ranges:
            if start <= frame_count <= end:
                abnormal_real_state = True

        abnormal_real_state_body = False
        for start, end, action in self.abnormal_ranges_body:
            if start <= frame_count <= end:
                abnormal_real_state_body = True

        #bol vyhodnoteny ako abnormal
        if abnormal:
            #a aj je abnormal
            if abnormal_real_state:
                self.true_positive += 1
                # print('true positive',frame_count)
            #ale je normal- takze bol zle ako pozitiv
            else:
                # pr\int('false postive ',frame_count)
                # cv2.imwrite('Photos/false_positive/'+str(frame_count)+'.jpg',frame)

                self.false_positive+=1
            if abnormal_real_state_body:
                self.true_positive_body+=1
            else:
                self.false_positive_body+=1
        # bol vyhodnoteny ako normalny
        else:
            #ale v skutocnosti nie je
            if abnormal_real_state:
                self.false_negative+=1
                # print('false negative  ',frame_count)
                # cv2.imwrite('Photos/false_negative/'+str(frame_count)+'.jpg',frame)

            else:
                self.true_negative+=1
                # print('true negative',frame_count)

            if abnormal_real_state_body:
                self.false_negative_body+=1
            else:
                self.true_negative_body+=1

print(1)
fvideo=FaceAnalyser()
fvideo.run()