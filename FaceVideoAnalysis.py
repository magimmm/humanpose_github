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
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.1)
        self.frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

        self.false_positive_body = 0
        self.false_negative_body = 0
        self.true_positive_body = 0
        self.true_negative_body = 0
        self.not_detected=0

        self.y_min = - 70
        self.y_max = 70
        self.x_min = - 70
        self.x_max = 70
        self.detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (0, 0))
        self.ranges = [
            (5900, 6500, "cough"),
            (7102, 8300, "cough"),
            (9990, 10600, "cough"),
            (12400, 12901, "sneeze"),
            (12900, 13300, "yawn"),
            (17200, 17600, "yawn")
        ]

        self.abnormal_ranges = [
            (6122, 6214, "cough"),
            (7895, 7999, "cough"),
            (10349, 10460, "cough"),
            (12552, 12614, "sneeze"),
            (12937, 13118, "yawn"),
            (17289, 17325, "yawn")
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
        self.cnn.load_model("cnn_model_mp.pth")

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


    def process_frame(self,frame):
        yolo_detected_landmarks = self.yolo_detector.get_landmarks(frame)
        yolo_skeleton = YoloSkeleton()
        yolo_skeleton.setup_from_detector(yolo_detected_landmarks)
        yolo_skeleton.find_face()

        crop_region = frame[max(0, yolo_skeleton.y_min + self.y_min):yolo_skeleton.y_max + self.y_max,
                      max(0, yolo_skeleton.x_min) + self.x_min:yolo_skeleton.x_max + self.x_max]

        image_rgb = cv2.cvtColor(crop_region, cv2.COLOR_BGR2RGB)

        # Process the image
        results = self.face_detection.process(image_rgb)

        # Check if at least one face is detected
        if results.detections:
            # Only process the first detected face
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)

            # Crop the detected face
            face_crop = crop_region[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            # print(f"Face detected and saved successfully: {save_path}")
        else:
            self.not_detected+=1
            return None
        # save_path = os.path.join("Photos", "cutouts", "normal2", filename)
        # cv2.imwrite(save_path, crop_region)


        result= self.cnn.predict(face_crop)
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

        # frame_count=5400
        frame_count=5400
        self.total_count=0
        # frame_skip=0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5400)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 5400)
        # Process each frame
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # if frame_count%frame_skip==0:

            result=self.process_frame(frame)
            print(result, frame_count)

            # if result==0:
            #     state='abnormal'
            # elif result==1:
            #     state='left'
            # elif result==2:
            #     state='normal'
            # else:
            #     state='right'

            if result:
                if result>0:
                    one_frame_state_abnormal=False
                else:
                    one_frame_state_abnormal=True

#odkomentuj pre frames check
                # self.last_states.append(one_frame_state_abnormal)
                #
                # # If the list of last states exceeds 5, remove the oldest state
                # if len(self.last_states) > 5:
                #     self.last_states.pop(0)
                #
                # if len(set(self.last_states)) == 1:
                #     # If all states are the same, return this common state
                #     self.driver_state_abnormal=one_frame_state_abnormal
                # self.check_detection(self.driver_state_abnormal, frame_count)



                self.check_detection(one_frame_state_abnormal, frame_count)

            frame_count += 1

                # cv2.putText(frame, f'Driver state: {state}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # # Display the frame
                # cv2.imshow('Video', frame)
                #
                # # Check for key press to exit
                # if cv2.waitKey(5) & 0xFF == ord('q'):
                #     break

        print(self.total_count)
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

    def check_detection_select(self, abnormal, frame_count):
        for start, end, action in self.ranges:
            if start <= frame_count <= end:
                # cv2.imwrite('Photos/testra/'+str(frame_count)+'.jpg',frame)
                self.total_count += 1
                abnormal_real_state = False
                for start, end, action in self.abnormal_ranges:
                    if start <= frame_count <= end:
                        abnormal_real_state = True

                # bol vyhodnoteny ako abnormal
                if abnormal:
                    # a aj je abnormal
                    if abnormal_real_state:
                        self.true_positive += 1
                        # print('true positive',frame_count)
                    # ale je normal- takze bol zle ako pozitiv
                    else:
                        # pr\int('false postive ',frame_count)
                        # cv2.imwrite('Photos/false_positive/'+str(frame_count)+'.jpg',frame)

                        self.false_positive += 1

                # bol vyhodnoteny ako normalny
                else:
                    # ale v skutocnosti nie je
                    if abnormal_real_state:
                        self.false_negative += 1
                        # print('false negative  ',frame_count)
                        # cv2.imwrite('Photos/false_negative/'+str(frame_count)+'.jpg',frame)

                    else:
                        self.true_negative += 1
                        # print('true negative',frame_count)



    def check_detection(self,abnormal,frame_count):

        abnormal_real_state_body = False
        for start, end, action in self.abnormal_ranges_body:
            if start <= frame_count <= end:
                abnormal_real_state_body = True

        #bol vyhodnoteny ako abnormal
        if abnormal:
            #a aj je abnormal

            if abnormal_real_state_body:
                self.true_positive_body+=1
            else:
                self.false_positive_body+=1
        # bol vyhodnoteny ako normalny
        else:
            #ale v skutocnosti nie je

            if abnormal_real_state_body:
                self.false_negative_body+=1
            else:
                self.true_negative_body+=1

print(1)
fvideo=FaceAnalyser()
fvideo.run()