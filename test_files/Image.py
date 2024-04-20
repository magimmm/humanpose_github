User
import os

from PIL.Image import Image
from PIL import Image

from LandmarkTester import LandmarkTester
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import mediapipe as mp
import time
import glob
import torchvision.models as models
import numpy as np

from SkeletonDetector import SkeletonDetector
from YoloDetector import YoloDetector
from YoloSkeleton import YoloSkeleton


class CNN():
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (0, 0))
        self.missed_abnormal=0
        self.misssed_normal=0
        self.missed=0

    def train_model(self):
        print('Training')
        BATCH_SIZE = 32
        EPOCHS = 50
        LR = 0.0001

        transform = transforms.Compose([
            transforms.Resize((80, 80)),  # Resize the images to 80x80
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(degrees=5),  # Random rotation within -10 to +10 degrees
            transforms.RandomAutocontrast(0.7),
            # transforms.RandomResizedCrop(size=(80, 80), scale=(0.8, 1.2)),  # Random resized crop
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Define data transformations for "normal" subfolder without augmentation
        transform_normal = transforms.Compose([
            transforms.Resize((80, 80)),  # Resize the images to 80x80
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Load the original dataset with augmentation only for "abnormal" subfolder
        # Load the original dataset with augmentation
        dataset = ImageFolder(root='Photos/neuron_convolution_binary', transform=transform)

        # Create a DataLoader for the dataset
        trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # Define the neural network model, loss function, and optimizer
        model = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.ReLU(),
            nn.LazyLinear(84),
            nn.ReLU(),
            nn.Linear(84, 4)
        )

        loss_fun = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LR)

        # Training loop
        for epoch in range(EPOCHS):
            print("epoch", epoch + 1)

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fun(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 20 == 19:
                    print(f'minibatch: {i + 1} loss: {running_loss / 20}')
                    running_loss = 0

        # Save the trained model
        filename = "cnn_model_mp_aug.pth"
        torch.save(model.state_dict(), filename)
        print('Finished Training')

    def predict(self,image):
        image_pil = Image.fromarray(image)

        transform = transforms.Compose([
            transforms.Resize((80, 80)),  # Resize the images to 80x80
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Preprocess the image
        image = transform(image_pil).unsqueeze(0)  # Add batch dimension

        # Perform prediction
        # with torch.no_grad():
        #     output = self.model(image)
        #     _, predicted = torch.max(output, 1)
        #
        # return predicted.item()

        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1)

        predicted_class_index = torch.argmax(probabilities)

        return probabilities.squeeze().tolist(),predicted_class_index

    def load_model(self,model_path):
        # Load the model
        model = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.ReLU(),
            nn.LazyLinear(84),
            nn.ReLU(),
            nn.Linear(84, 4)  # Changed the number of output neurons to 4
        )

        # Load the trained weights
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        self.model=model

        # Define the transformation to apply to the image

# train_model()
# cnn=CNN()
# cnn.create_cutouts(normal_path='Photos/all_images/3/normal1', abnormal_path='Photos/all_images/3/abnormal1')
# cnn.create_cutouts2()
# cnn=CNN()
# cnn.train_model()

-------------------
import cv2
from BodyNeuronNetwork import NeuronNetworkManager
from YoloDetector import YoloDetector
from MediaPipeDetector import MediaPipeDetector
from YoloSkeleton import YoloSkeleton
from MediapipeSkeleton import MediaPipeSkeleton
import mediapipe as mp
from cnn import CNN
class FaceAnalyser:
    def __init__(self,model_path,using_mp,show,noise_correction,test_seq):
        self.test_seq=test_seq
        self.noise_correction=noise_correction
        self.show=show
        self.using_mp=using_mp
        self.model_path=model_path
        self.last_states = []
        self.threshold=0.55
        self.false_positive=0
        self.false_negative=0
        self.true_positive=0
        self.true_negative=0

        self.false_positive_select=0
        self.false_negative_select=0
        self.true_positive_select=0
        self.true_negative_select=0

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

        # y_min = - 40
        # y_max = 10
        # x_min = - 40
        # x_max = 0

        self.y_min_crop = - 40
        self.y_max_crop = 10
        self.x_min_crop = - 40
        self.x_max_crop = 0

        self.test_ranges = [
            (5900, 6500, "cough"),
            (7102, 8300, "cough"),
            (9990, 10600, "cough"),
            (12400, 12901, "sneeze"),
            (12900, 13300, "yawn"),
            (17200, 17600, "yawn")
        ]

        self.test_abnormal_ranges = [
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
        self.cnn.load_model(self.model_path)

    def process_frame_without_mp(self,frame):
        yolo_detected_landmarks = self.yolo_detector.get_landmarks(frame)
        yolo_skeleton = YoloSkeleton()
        yolo_skeleton.setup_from_detector(yolo_detected_landmarks)
        yolo_skeleton.find_face()

        crop_region = frame[max(0, yolo_skeleton.y_min + self.y_min_crop):min(yolo_skeleton.y_max + self.y_max_crop, 1080),
                      max(0, yolo_skeleton.x_min) + self.x_min_crop:min(yolo_skeleton.x_max + self.x_max_crop, 1920)]

        result = self.cnn.predict(crop_region)
        return result


    def process_frame(self,frame):
        yolo_detected_landmarks = self.yolo_detector.get_landmarks(frame)
        yolo_skeleton = YoloSkeleton()
        yolo_skeleton.setup_from_detector(yolo_detected_landmarks)
        yolo_skeleton.find_face()

        crop_region = frame[max(0, yolo_skeleton.y_min + self.y_min):min(yolo_skeleton.y_max + self.y_max,1080),
                      max(0, yolo_skeleton.x_min) + self.x_min:min(yolo_skeleton.x_max + self.x_max,1920)]



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
        else:
            self.not_detected+=1
            return None
        for i in face_crop.shape:
            if i==0:
                self.not_detected += 1
                return None
        # save_path = os.path.join("Photos", "cutouts", "normal2", filename)
        # cv2.imwrite(save_path, crop_region)
        # print(face_crop.shape)
        # cv2.imshow('h',face_crop)
        result= self.cnn.predict(face_crop)
        return result

    def get_state(self, result):
        if result==0:
            state='abnormal'
        elif result==1:
            state='left'
        elif result==2:
            state='normal'
        else:
            state='right'
        return state
    def get_abormal_state_from_list_prob(self,probabilities,state):
        if probabilities[1] + probabilities[2] + probabilities[3] > probabilities[0]:
            return False
        else:
            return True
    def get_abormal_state_from_one_val(self,state):
        if state>0:
            return False
        else:
            return True


    def test(self):
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

        frame_count = 5400
        self.total_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5400)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            for start, end, action in self.test_ranges:
                if start <= frame_count <= end:

                    if self.using_mp:
                        result = self.process_frame(frame)
                    else:
                        result = self.process_frame_without_mp(frame)
                    if result is not None:
                        probabilities, result_value = result
                        # Further processing using probabilities and result_value
                        one_frame_state_abnormal = self.get_abormal_state_from_list_prob(probabilities,result_value)
                        # print(result_value, frame_count)

                        self.last_states.append(one_frame_state_abnormal)

                        # If the list of last states exceeds 5, remove the oldest state
                        if len(self.last_states) > 5:
                            self.last_states.pop(0)

                        if len(set(self.last_states)) == 1:
                            # If all states are the same, return this common state
                            self.driver_state_abnormal = one_frame_state_abnormal

                        self.check_detection_select(self.driver_state_abnormal, frame_count)
                        cv2.putText(frame, f'Driver state: {result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)
                        cv2.putText(frame, f'Abnormal: {self.driver_state_abnormal}', (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 255), 2)
                        cv2.imshow('Video', frame)
                        if cv2.waitKey(5) & 0xFF == ord('q'):
                            break


                    # Check for key press to exit

            frame_count+=1

        cap.release()
        cv2.destroyAllWindows()
        self.calculate_sensitivity_specificity(tp=self.true_positive_select,tn=self.true_negative_select,fn=self.false_negative_select,fp=self.false_positive_select)

    def test2(self):
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

        frame_count = 5400
        self.total_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5400)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            result = self.process_frame(frame)
            if result is not None:
                probabilities, result_value = result
                # Further processing using probabilities and result_value
                one_frame_state_abnormal = self.get_state_from_list_prob(probabilities, result_value)
                print(result_value, frame_count)

                self.last_states.append(one_frame_state_abnormal)

                # If the list of last states exceeds 5, remove the oldest state
                if len(self.last_states) > 5:
                    self.last_states.pop(0)

                if len(set(self.last_states)) == 1:
                    # If all states are the same, return this common state
                    self.driver_state_abnormal = one_frame_state_abnormal

                cv2.putText(frame, f'Abnormal: {self.driver_state_abnormal}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 255), 2)
                # Display the frame

                self.check_detection(self.driver_state_abnormal, frame_count)

                self.check_detection_select(self.driver_state_abnormal, frame_count)

            frame_count += 1

            cv2.putText(frame, f'Driver state: {result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Display the frame
            cv2.imshow('Video', frame)

            # Check for key press to exit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        print(self.total_count)
        cap.release()
        cv2.destroyAllWindows()
        self.calculate_sensitivity_specificity()
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
            if self.test_seq:
                for start, end, action in self.test_ranges:
                    if start <= frame_count <= end:
                        if not self.handle_frame(frame, frame_count):
                            break
            else:
                if not self.handle_frame(frame,frame_count):
                    break
            frame_count+=1

        cap.release()
        cv2.destroyAllWindows()

        if self.test_seq:
            self.calculate_sensitivity_specificity(tp=self.true_positive_select, tn=self.true_negative_select,
                                                   fn=self.false_negative_select, fp=self.false_positive_select)
        else:
            self.calculate_sensitivity_specificity(tn=self.true_negative,tp=self.true_positive,fn=self.false_negative,fp=self.false_positive)


    def handle_frame(self,frame,frame_count):
        if self.using_mp:
            result = self.process_frame(frame)
        else:
            result = self.process_frame_without_mp(frame)
        if result is not None:
            one_frame_state_abnormal = self.get_abormal_state_from_one_val(result)
            # print(result_value, frame_count)

            if self.noise_correction:
                self.last_states.append(one_frame_state_abnormal)

                # If the list of last states exceeds 5, remove the oldest state
                if len(self.last_states) > 5:
                    self.last_states.pop(0)

                if len(set(self.last_states)) == 1:
                    # If all states are the same, return this common state
                    self.driver_state_abnormal = one_frame_state_abnormal
                else:
                    one_frame_state_abnormal = self.driver_state_abnormal


            if self.test_seq:
                self.check_detection_select(one_frame_state_abnormal,frame_count)
            else:
                self.check_detection(one_frame_state_abnormal, frame_count)

            if self.show:
                cv2.putText(frame, f'Driver state: {self.get_state(result)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
                cv2.putText(frame, f'Abnormal: {one_frame_state_abnormal}', (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 255), 2)
                cv2.imshow('Video', frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    return 0

    def run2(self):
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

        frame_count = 5400
        # frame_skip=0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5400)
        # Process each frame
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if self.using_mp:
                result = self.process_frame(frame)
            else:
                result = self.process_frame_without_mp(frame)
            if result is not None:
                one_frame_state_abnormal = self.get_abormal_state_from_one_val(result)
                # print(result_value, frame_count)

                if self.noise_correction:
                    self.last_states.append(one_frame_state_abnormal)

                    # If the list of last states exceeds 5, remove the oldest state
                    if len(self.last_states) > 5:
                        self.last_states.pop(0)

                    if len(set(self.last_states)) == 1:
                        # If all states are the same, return this common state
                        self.driver_state_abnormal = one_frame_state_abnormal
                    else:
                        one_frame_state_abnormal = self.driver_state_abnormal

                self.check_detection(one_frame_state_abnormal, frame_count)

                if self.show:
                    cv2.putText(frame, f'Driver state: {self.get_state(result)}', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2)
                    cv2.putText(frame, f'Abnormal: {one_frame_state_abnormal}', (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 255), 2)
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()
        self.calculate_sensitivity_specificity(tn=self.true_negative, tp=self.true_positive, fn=self.false_negative,
                                               fp=self.false_positive)

    def calculate_sensitivity_specificity(self,tp,tn,fp,fn):
        total = tn+tp+fp+fn
        sensitivity = tp / (tp+fn)
        specificity = tn / (tn+fp)
        accuracy = (tp+tn) / total
        print("Specificity:",specificity)
        print("Sensitivity:", sensitivity)
        print("Accuracy:", accuracy)
        print(tn,tp,fp,fn)


    def check_detection_select(self, abnormal, frame_count):
            abnormal_real_state = False
            for start, end, action in self.test_abnormal_ranges:
                if start <= frame_count <= end:
                    abnormal_real_state = True

            # bol vyhodnoteny ako abnormal
            if abnormal:
                # a aj je abnormal
                if abnormal_real_state:
                    self.true_positive_select += 1
                else:
                    print('fp  ',frame_count)
                    self.false_positive_select += 1

            # bol vyhodnoteny ako normalny
            else:
                # ale v skutocnosti nie je
                if abnormal_real_state:
                    self.false_negative_select += 1
                    print('fn  ',frame_count)
                    # cv2.imwrite('Photos/false_negative/'+str(frame_count)+'.jpg',frame)

                else:
                    self.true_negative_select += 1
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
                print('fp  ', frame_count)
        # bol vyhodnoteny ako normalny
        else:
            #ale v skutocnosti nie je

            if abnormal_real_state_body:
                self.false_negative_body+=1
                print('fn  ', frame_count)
            else:
                self.true_negative_body+=1



print(1)
fvideo=FaceAnalyser("cnn_model_v_8_50_2.pth",using_mp=False,show=True,noise_correction=True,test_seq=False)
fvideo.run()




