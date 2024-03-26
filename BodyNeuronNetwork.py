import numpy as np
import math
import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from YoloSkeleton import YoloSkeleton
import os
from LandmarkTester import LandmarkTester
from ultralytics import YOLO
import cv2
import mediapipe as mp

class NeuronNetworkManager:
    def __init__(self):
        self.model = None
        self.num_features = 0
    def create_nn_input_from_skeletons(self,skeletons):
        skeletons_distances_list = []
        for skeleton in skeletons:
            skeleton.preprocess_for_comparing()
            # TODO vsetky vzdialenosti
            #skeletons_distances_list.append(skeleton.features_vector_whole_body)
            skeletons_distances_list.append(skeleton.features_vector)
        array_of_arrays = np.array([np.array(sublist) for sublist in skeletons_distances_list])
        self.num_features = len(skeletons[0].features_vector)
        return array_of_arrays

    def load_paths(self,normal_folder,abnormal_folder):
        normal_paths = [os.path.join(normal_folder, filename) for filename in os.listdir(normal_folder) if
                        os.path.isfile(os.path.join(normal_folder, filename))]
        abnormal_paths = [os.path.join(abnormal_folder, filename) for filename in os.listdir(abnormal_folder) if
                          os.path.isfile(os.path.join(abnormal_folder, filename))]

        # Concatenate the two lists and return
        all_paths = normal_paths + abnormal_paths
        return all_paths

    def train_model(self,normal_folder,abnormal_folder):

        images_paths=self.load_paths(normal_folder,abnormal_folder)

        landmarktester=LandmarkTester()
        landmarktester.create_skeletons_from_detections(images_paths)
        #normal, abnormal
        skeletons=landmarktester.detected_skeletons_yolo
        #skeletons=landmarktester.detected_skeletons_mediapipe


        data=self.create_nn_input_from_skeletons(skeletons)
        num_samples = len(skeletons)

        # Create labels (0 for normal, 1 for abnormal)
        labels = np.concatenate([np.zeros(num_samples // 2), np.ones(num_samples // 2)])

        # for d in data:
        #     print(d)
        # print(data)

        # Shuffle data and labels
        p = np.random.permutation(num_samples)
        data = data[p]
        labels = labels[p]


        # Define the neural network model
        model = Sequential([
            Dense(16, activation='selu', input_shape=(self.num_features,)),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        custom_optimizer = SGD(learning_rate=0.001)
        # Compile the model
        model.compile(optimizer=custom_optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(data, labels, epochs=50, batch_size=16)

        model.save("new_model")

    def load_model(self):
        self.model = load_model("new_model")

    def predict_img(self,features_vector):
        input_vector = np.array([features_vector])

        # Reshape the input vector to match the expected shape of the input data for the model
        input_vector = input_vector.reshape(1, -1)  # If your input is 1D, you need to reshape it to 2D

        # Make predictions using the trained model
        predictions = self.model.predict(input_vector)

        # Print the predictions
        print("Predictions:", predictions)
        return predictions[0]

    def augmentation(self):


        # Create an instance of ImageDataGenerator with augmentation parameters
        datagen = ImageDataGenerator(
            brightness_range=(0.2, 0.6),
            preprocessing_function=add_noise  #
        )
        destination_directory = 'Photos/augumented/normal'
        os.makedirs(destination_directory, exist_ok=True)

        # Generate augmented images from a directory containing daytime images
        augmented_generator = datagen.flow_from_directory('Photos/neuron_body/normal_for_aug',
            target_size=(1080, 1920),
            batch_size=8,
            class_mode='binary',
            save_to_dir=destination_directory,
            save_prefix='augmented',
            save_format='jpg'
        )

    # Iterate through all batches and save each augmented image
        for i, (images, _) in enumerate(augmented_generator):
            for j, image in enumerate(images):
                filename = f'augmented_{i * augmented_generator.batch_size + j}.jpg'
                filepath = os.path.join(destination_directory, filename)
                image=adjust_contrast(image, np.random.uniform(0.2, 0.6))
                cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # OpenCV expects BGR format
            if i >= augmented_generator.samples // augmented_generator.batch_size:
                break

        print("Augmented images saved successfully.")


# Function to add noise to an image
def add_noise(image):
    row, col, _ = image.shape
    # Generate random noise
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, 3))

    # Ensure both arrays have the same data type
    image = image.astype(np.float32)
    gauss = gauss.astype(np.float32)

    # Add noise to the image
    noisy_image = cv2.add(image, gauss)
    return noisy_image.astype(np.uint8)  # Convert back to uint8

def adjust_contrast(image, contrast_factor):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the mean pixel value
    mean = np.mean(gray)
    # Adjust contrast
    adjusted = (image - mean) * contrast_factor + mean
    # Clip values to ensure they are within the valid range [0, 255]
    adjusted = np.clip(adjusted, 0, 255)
    # Convert back to uint8
    adjusted = adjusted.astype(np.uint8)
    return adjusted



