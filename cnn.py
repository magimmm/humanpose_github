import os
from LandmarkTester import LandmarkTester

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
    def cut_face(self,frame,path,filename):
        height, width, _ = frame.shape

        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)

        if faces is None:
            self.missed += 1

        # if faces[1] is None, no face found


            # not None:
            # for i,face in enumerate(faces):
                # parameters: x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm
                # pass
                # # bouding box
                # box = list(map(int, face[:4]))
                # color = (0, 0, 255)
                # face_img = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                # cv2.imwrite(f'{path}.jpg', face_img)
                #
                # cv2.rectangle(frame, box, color, 5)
                #
                # # confidence
                # confidence = face[-1]
                # confidence = "{:.2f}".format(confidence)
                # position = (box[0], box[1] - 10)
                # cv2.putText(frame, confidence, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3, cv2.LINE_AA)
        # else:


        # face_cutout = frame[y:y + h, x:x + w]
        # cv2.imwrite(output_path, face_cutout)
        # print("Face cutout saved as:", output_path)

        # Display the image with detected faces

        # Display the image with detected faces


    def create_cutouts2(self):

        normal_path='Photos/all_images/3/normal1'
        abnormal_path='Photos/all_images/3/abnormal1'
        normal_images_paths=[]
        abnormal_images_paths=[]
        skeleton_detector = SkeletonDetector()
        self.yolo_detector = YoloDetector('yolov8n-pose.pt')
        self.yolo_detector.setup()
        y_min= - 40
        y_max= 10
        x_min= - 40
        x_max=0

        ni=0
        for filename in os.listdir(normal_path):
            if ni>100000:
                break
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more formats if needed
                # Read the image
                img_path = os.path.join(normal_path, filename)
                yolo_detected_landmarks = self.yolo_detector.get_landmarks(img_path)
                yolo_skeleton = YoloSkeleton(img_path)
                yolo_skeleton.setup_from_detector(yolo_detected_landmarks)
                yolo_skeleton.find_face()
                img = cv2.imread(img_path)
                crop_region = img[yolo_skeleton.y_min +y_min:yolo_skeleton.y_max +y_max,
                              yolo_skeleton.x_min +x_min:yolo_skeleton.x_max+x_max]

                save_path = os.path.join("Photos", "cutouts", "normal2", filename)
                cv2.imwrite(save_path, crop_region)
                ni += 1
                # frame = cv2.imread(img_path)
                # self.cut_face(frame,'Photos/cutouts/normal2/'+str(i),filename)

        ni=0
        for filename in os.listdir(abnormal_path):
            if ni > 100000:
                break
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more formats if needed
                # Read the image
                img_path = os.path.join(abnormal_path, filename)
                yolo_detected_landmarks = self.yolo_detector.get_landmarks(img_path)
                yolo_skeleton = YoloSkeleton(img_path)
                yolo_skeleton.setup_from_detector(yolo_detected_landmarks)
                yolo_skeleton.find_face()
                img = cv2.imread(img_path)
                crop_region = img[yolo_skeleton.y_min + y_min:yolo_skeleton.y_max + y_max,
                              yolo_skeleton.x_min + x_min:yolo_skeleton.x_max + x_max]

                save_path = os.path.join("Photos", "cutouts", "abnormal2", filename)
                cv2.imwrite(save_path, crop_region)
                ni+=1

                # frame = cv2.imread(img_path)
                # self.cut_face(frame,'Photos/cutouts/abnormal2/'+str(i),filename)



    def detect_and_save_face(self, image_path, output_directory):
        # Load the image
        image = cv2.imread(image_path)
        # Convert the image to RGB (MediaPipe requires RGB input)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            # Process the image
            results = face_detection.process(image_rgb)

            # Check if at least one face is detected
            if results.detections:
                # Only process the first detected face
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)

                # Crop the detected face
                face_crop = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

                # Save the cropped face
                filename = os.path.basename(image_path)
                save_path = os.path.join(output_directory, filename)
                cv2.imwrite(save_path, face_crop)
                print(f"Face detected and saved successfully: {save_path}")
            else:
                print(f"No face detected in: {image_path}")

    def create_cutouts(self, normal_path, abnormal_path):
        normal_output_dir = 'Photos/cutouts/normal/'
        abnormal_output_dir = 'Photos/cutouts/abnormal/'

        # Process normal images
        for filename in os.listdir(normal_path):#[:10]:
            image_path = os.path.join(normal_path, filename)
            if os.path.isfile(image_path):
                self.detect_and_save_face(image_path, normal_output_dir)

        # Process abnormal images
        for filename in os.listdir(abnormal_path):#[:10]:
            image_path = os.path.join(abnormal_path, filename)
            if os.path.isfile(image_path):
                self.detect_and_save_face(image_path, abnormal_output_dir)







        # start=time.time()
        # normal_path='Photos/all_images/3/normal1'
        # abnormal_path='Photos/all_images/3/abnormal1'
        # i=0
        # for filename in os.listdir(normal_path):
        #     if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more formats if needed
        #         # Read the image
        #         img_path = os.path.join(normal_path, filename)
        #         i+=1
        #         frame = cv2.imread(img_path)
        #         self.cut_face(frame,'Photos/cutouts/normal/'+str(i),filename)
        #
        # print('normal ',self.missed)
        # self.missed=0
        # i = 0
        # for filename in os.listdir(abnormal_path):
        #     if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more formats if needed
        #         # Read the image
        #         i += 1
        #         img_path = os.path.join(abnormal_path, filename)
        #         frame = cv2.imread(img_path)
        #         self.cut_face(frame,'Photos/cutouts/abnormal/'+str(i),filename)
        #
        # print('abnormal ', self.missed)
        # end=time.time()
        # print(end-start)


    def train_model(self):
            BATCH_SIZE = 8
            EPOCHS = 16
            LR = 0.001

            transform = transforms.Compose([
                transforms.Resize((80, 80)),  # Resize the images to 80x80
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])

            trainset = torchvision.datasets.ImageFolder(root='Photos/neuron_convolution', transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            classes = trainset.classes

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

            loss_fun = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=LR)
            filename = "cnn_model.pth"
            for epoch in range(EPOCHS):
                print("epoch", epoch+1)

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
                        print(f'minibatch: {i+1} loss: {running_loss / 20}')
                        running_loss = 0

            torch.save(model.state_dict(), filename)
            print('Finished Training')

    def load_model(self):
        BATCH_SIZE = 8
        EPOCHS = 16
        LR = 0.001

        transform = transforms.Compose([
            transforms.Resize((80, 80)),  # Resize the images to 80x80
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        trainset = torchvision.datasets.ImageFolder(root='Photos/neuron_convolution', transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        classes = trainset.classes

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

        loss_fun = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LR)
        filename = "cnn_model.pth"
        model.load_state_dict(torch.load(filename))
        model.eval()
        return model, classes, transform

    def predict(self,image):
        transform = transforms.Compose([
            transforms.Resize((80, 80)),  # Resize the images to 80x80
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Preprocess the image
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform prediction
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)

        return predicted.item()
    def load_model_and_predict(self,model_path):
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
cnn=CNN()
# cnn.create_cutouts(normal_path='Photos/all_images/3/normal1', abnormal_path='Photos/all_images/3/abnormal1')
# cnn.create_cutouts2()
#cnn.train_model()