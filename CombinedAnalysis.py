import cv2
from YoloDetector import YoloDetector
from YoloSkeleton import YoloSkeleton
import mediapipe as mp
from cnn import CNN
from BodyNeuronNetwork import NeuronNetworkManager

class StateAnalyser:
    def __init__(self,model_path,using_mp,show,noise_correction,noise):
        self.noise=noise
        self.noise_correction=noise_correction
        self.show=show
        self.using_mp=using_mp
        self.model_path=model_path
        self.last_states = []
        self.driver_head_turning_state='Normal'

        self.last_states_body = []

        self.last_states_face = []
        self.last_states_turning = []


        self.threshold=0.55
        self.false_positive=0
        self.false_negative=0
        self.h=0
        self.true_positive=0
        self.true_negative=0

        self.false_positive_select=0
        self.false_negative_select=0
        self.true_positive_select=0
        self.true_negative_select=0

        self.driver_state_abnormal=False
        self.driver_state_abnormal_face = False
        self.driver_state_abnormal_body = False

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.1)

        self.false_positive_body = 0
        self.false_negative_body = 0
        self.true_positive_body = 0
        self.true_negative_body = 0
        self.not_detected=0

        self.y_min_crop = - 40
        self.y_max_crop = 10
        self.x_min_crop = - 40
        self.x_max_crop = 0
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

    def setup_models(self):
        self.cnn = CNN()
        self.cnn.load_model(self.model_path)

        self.body_nn_manager = NeuronNetworkManager()
        self.body_nn_manager.load_model()

    def process_frame_without_mp(self,frame):
        yolo_detected_landmarks = self.yolo_detector.get_landmarks(frame)
        yolo_skeleton = YoloSkeleton()
        yolo_skeleton.setup_from_detector(yolo_detected_landmarks)
        yolo_skeleton.find_face()
        yolo_skeleton.create_body_nn_feature_vector()

        crop_region = frame[max(0, yolo_skeleton.y_min + self.y_min_crop):min(yolo_skeleton.y_max + self.y_max_crop, 1080),
                      max(0, yolo_skeleton.x_min) + self.x_min_crop:min(yolo_skeleton.x_max + self.x_max_crop, 1920)]

        face_result = self.cnn.predict_probs(crop_region)
        if face_result is not None:
            self.face_prediction,self.face_predict_probabilities = face_result
        else:
            self.face=None
            self.h+=1
        self.body_prediction= self.body_nn_manager.predict_img(yolo_skeleton.features_vector)

    def get_body_abnormal_state(self,value):
        if value>self.threshold:
            return 'abnormal_b'
        else:
            return 'normal_b'
    def get_abormal_state_from_list_prob(self,probabilities,state):
        if probabilities[1] + probabilities[2] + probabilities[3] > probabilities[0]:
            return False
        else:
            return True
    def get_abormal_state_from_one_val(self,state):
        if state:
            return 'Abnormal'

        else:
            return 'Normal'
    def run(self):
        # print('Run')
        self.setup_yolo_detector()
        self.setup_models()

        # print('Video')
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

            if self.handle_frame(frame,frame_count)=='break':
                break

            frame_count+=1

        cap.release()
        cv2.destroyAllWindows()


        self.calculate_sensitivity_specificity(tn=self.true_negative,tp=self.true_positive,fn=self.false_negative,fp=self.false_positive)
    def noise_correction(self):
        self.last_states.append(self.one_frame_total_abnormality)

        # If the list of last states exceeds 5, remove the oldest state
        if len(self.last_states) > self.noise:
            self.last_states.pop(0)

        if len(set(self.last_states)) == 1:
            # If all states are the same, return this common state
            self.driver_state_abnormal = self.one_frame_total_abnormality


    def noise_correction_body(self):
        self.last_states_body.append(self.one_frame_body_state_abnormal)

        # If the list of last states exceeds 5, remove the oldest state
        if len(self.last_states_body) > 3:
            self.last_states_body.pop(0)

        if len(set(self.last_states_body)) == 1:
            # If all states are the same, return this common state
            self.driver_state_abnormal_body = self.one_frame_body_state_abnormal

    def noise_correction_face(self):
        self.last_states_face.append(self.one_frame_face_state_abnormal)

        # If the list of last states exceeds 5, remove the oldest state
        if len(self.last_states_face) > self.noise:
            self.last_states_face.pop(0)

        if len(set(self.last_states_face)) == 1:
            # If all states are the same, return this common state
            self.driver_state_abnormal_face = self.one_frame_face_state_abnormal

    def combine_states(self):
        if (self.driver_state_abnormal_face )or (self.driver_state_abnormal_body):
        # if (self.driver_state_abnormal_face)or self.driver_state_abnormal_body:
            self.one_frame_total_abnormality=True
            self.last_states_turning.clear()

        else:
            self.one_frame_total_abnormality=False
        # face_state=combined_state=body_state=False
        # if self.noise_correction:
        #     self.noise_correction(combined_state,body_state,face_state)

    def set_states_from_predictions(self):
        if self.body_prediction > self.threshold:
            self.one_frame_body_state_abnormal=True
        else:
            self.one_frame_body_state_abnormal=False

        # TODO
        # if self.face_prediction == 0 and self.face_predict_probabilities[0] > 0.9:
        #     self.one_frame_face_state_abnormal = True
        # else:
        #     self.one_frame_face_state_abnormal = False

        if self.face_prediction == 0:
            self.one_frame_face_state_abnormal=True
        else:
            self.one_frame_face_state_abnormal=False

    def get_head_turning(self):
        if self.driver_head_turning_state == 0:
            return '-'
        elif self.driver_head_turning_state == 1:
            return 'Left'
        elif self.driver_head_turning_state == 2:
            return 'Straight'
        else:
            return 'Right'

    def set_head_turning(self):

        self.last_states_turning.append(self.face_prediction)

        # If the list of last states exceeds 5, remove the oldest state
        if len(self.last_states_turning) > self.noise:
            self.last_states_turning.pop(0)

        if len(set(self.last_states_turning)) == 1:
            # If all states are the same, return this common state
            self.driver_head_turning_state=self.face_prediction

    def handle_frame(self,frame,frame_count):
        self.process_frame_without_mp(frame)
        self.set_states_from_predictions()
        # first one by one, then just the final
        # self.noise_correction()
        if self.noise_correction:
            self.noise_correction_body()
            self.noise_correction_face()
        else:
            self.driver_state_abnormal_body=self.one_frame_body_state_abnormal
            self.driver_state_abnormal_face=self.one_frame_face_state_abnormal

        #druha varianta ked najskor skombinujem z one frames a pootom noise na total driver
        self.combine_states()

        self.check_detection(frame_count)

        offset=950
        normal_color=(119, 204, 29)
        abnormal_color=(12,24,166)
        text_color=(174, 176, 148)

        color_face=normal_color
        color_body=normal_color
        color_total=normal_color
        prob_face=self.face_predict_probabilities[1]+self.face_predict_probabilities[2]+self.face_predict_probabilities[3]

        if self.driver_state_abnormal_face:
            color_face=abnormal_color
            prob_face=self.face_predict_probabilities[0]
            head_state='-'
        else:
            self.set_head_turning()
            head_state=self.get_head_turning()
        if self.driver_state_abnormal_body:
            color_body=abnormal_color
        if self.one_frame_total_abnormality:
            color_total = abnormal_color

        width = 600
        height = 500
        # Define the origin (position) of the rectangle
        origin = (offset-50, 50)  # Example: X-coordinate = 50, Y-coordinate = 50
        endpoint = (origin[0] + width, origin[1] + height)


        if self.show:
            # Draw a filled rectangle on the image with the specified opacity
            cv2.rectangle(frame,  origin, endpoint, (0, 0, 0), -1)
            cv2.putText(frame,
                        f'Face state: {self.get_abormal_state_from_one_val(self.driver_state_abnormal_face)}, ({prob_face:.2f})',
                        (offset, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_face, 2)
            cv2.putText(frame,
                        f'Head turned to: {head_state}',
                        (offset, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_face, 2)

            cv2.putText(frame, f'Body state: {self.get_abormal_state_from_one_val(self.driver_state_abnormal_body)}',
                        (offset, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color_body, 2)
            cv2.putText(frame, f'Total state: {self.get_abormal_state_from_one_val(self.one_frame_total_abnormality)}', (offset, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        color_total, 3)

            # cv2.putText(frame,'1',(offset+100, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            # cv2.putText(frame,'2',(offset+150, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            # cv2.putText(frame,'2',(offset+200, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            # cv2.putText(frame,'4',(offset+250, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            # cv2.putText(frame,'5',(offset+300, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            # cv2.putText(frame,'6',(offset+350, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            # cv2.putText(frame,'7',(offset+400, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                return 'break'

        # if self.show:
        #     cv2.putText(frame, f'Face state: {self.get_face_state(self.face_prediction)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                 (255, 255, 255), 2)
        #     cv2.putText(frame, f'Face abnormality {self.get_abormal_state_from_one_val(self.face_prediction)}', (50, 100),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                 (255, 0, 255), 2)
        #     cv2.putText(frame, f'Body state: {self.get_body_abnormal_state(), self.body_prediction}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                 (255, 0, 255), 2)
        #     # cv2.putText(frame, f'Total state: {one_frame_state_abnormal}', (50, 150),
        #     #             cv2.FONT_HERSHEY_SIMPLEX, 1,
        #     #             (255, 0, 255), 2)
        #     cv2.imshow('Video', frame)
        #     if cv2.waitKey(5) & 0xFF == ord('q'):
        #         return 'break'


    def calculate_sensitivity_specificity(self,tp,tn,fp,fn):
        total = tn+tp+fp+fn
        print(tn,tp,fp,fn)
        sensitivity = tp / (tp+fn)
        specificity = tn / (tn+fp)
        accuracy = (tp+tn) / total
        print("Specificity:",specificity)
        print("Sensitivity:", sensitivity)
        print("Accuracy:", accuracy)
        print(self.h)

    def check_detection(self,frame_count):

        abnormal_real_state_body = False
        for start, end, action in self.abnormal_ranges_body:
            if start <= frame_count <= end:
                abnormal_real_state_body = True

        #bol vyhodnoteny ako abnormal
        if  self.one_frame_total_abnormality:
            #a aj je abnormal

            if abnormal_real_state_body:
                self.true_positive+=1
            else:
                self.false_positive+=1
                # if self.one_frame_body_state_abnormal==False:
                # print('F',self.face_predict_probabilities[0])
                # print('fp  ', frame_count)
        # bol vyhodnoteny ako normalny
        else:
            #ale v skutocnosti nie je

            if abnormal_real_state_body:
                self.false_negative+=1
                # print('fn  ', frame_count)
            else:
                self.true_negative+=1

# print(1)
# print('cnn_model_v_8_25_2.pth')
# fvideo=FaceAnalyser("cnn_model_v_8_25_2.pth",using_mp=False,show=False,noise_correction=False,test_seq=False)
# fvideo.run()



print('cnn_model_v_8_16_2.pth aug')
fvideo=StateAnalyser("cnn_model_v_8_16_2_aug.pth",using_mp=False,show=False,noise_correction=False,noise=5)
fvideo.run()

print('cnn_model_v_8_16_2.pth')
fvideo=StateAnalyser("cnn_model_v_8_16_2_aug.pth",using_mp=False,show=False,noise_correction=True,noise=5)
fvideo.run()
# fvideo=StateAnalyser("cnn_model_v_8_16_2.pth",using_mp=False,show=False,noise_correction=True,noise=5)
# fvideo.run()



#
# print('cnn_model_v_8_50_2.pth')
# fvideo=FaceAnalyser("cnn_model_v_8_50_2.pth",using_mp=False,show=False,noise_correction=True,test_seq=False)
# fvideo.run()
#
# print('cnn_model_v_16_50_2.pth')
# fvideo=FaceAnalyser("cnn_model_v_16_50_2.pth",using_mp=False,show=False,noise_correction=True,test_seq=False)
# fvideo.run()