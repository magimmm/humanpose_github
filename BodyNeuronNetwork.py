import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import save_model, load_model
import os
from LandmarkTester import LandmarkTester


class NeuronNetworkManager:
    def __init__(self):
        self.model = None
        self.num_features = 0

    def create_nn_input_from_skeletons(self, skeletons):
        """
           Create input data for the neural network from a list of skeletons.

           Args:
               skeletons (list): List of Skeleton objects.

           Returns:
               numpy.ndarray: Input data for the neural network.
        """
        skeletons_distances_list = []
        for skeleton in skeletons:
            skeleton.preprocess_for_comparing()
            skeletons_distances_list.append(skeleton.features_vector)
        array_of_arrays = np.array([np.array(sublist) for sublist in skeletons_distances_list])
        self.num_features = len(skeletons[0].features_vector)
        return array_of_arrays

    def load_paths(self, normal_folder, abnormal_folder):
        """
        Load file paths from normal and abnormal folders.

        Args:
            normal_folder (str): Path to the folder containing normal images.
            abnormal_folder (str): Path to the folder containing abnormal images.

        Returns:
            list: List of file paths.
        """
        normal_paths = [os.path.join(normal_folder, filename) for filename in os.listdir(normal_folder) if
                        os.path.isfile(os.path.join(normal_folder, filename))]
        abnormal_paths = [os.path.join(abnormal_folder, filename) for filename in os.listdir(abnormal_folder) if
                          os.path.isfile(os.path.join(abnormal_folder, filename))]

        # Concatenate the two lists and return
        all_paths = normal_paths + abnormal_paths
        return all_paths

    def train_model(self, normal_folder, abnormal_folder):
        """
        Train a neural network model using skeleton data extracted from images.

        Args:
            normal_folder (str): Path to the folder containing normal images.
            abnormal_folder (str): Path to the folder containing abnormal images.
        """
        images_paths = self.load_paths(normal_folder, abnormal_folder)

        landmarktester = LandmarkTester()
        landmarktester.create_skeletons_from_detections(images_paths)
        skeletons = landmarktester.detected_skeletons_yolo

        data = self.create_nn_input_from_skeletons(skeletons)
        num_samples = len(skeletons)

        # Create labels (0 for normal, 1 for abnormal)
        labels = np.concatenate([np.zeros(num_samples // 2), np.ones(num_samples // 2)])

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

        model.save("new_model2")

    def load_model(self):
        """
        Load a pre-trained neural network model.
        """
        self.model = load_model("new_model")

    def predict_img(self, features_vector):
        """
            Make predictions using the trained neural network model.

            Args:
                features_vector (list): Feature vector for prediction.

            Returns:
                numpy.ndarray: Predicted probabilities.
        """
        input_vector = np.array([features_vector])

        # Reshape the input vector to match the expected shape of the input data for the model
        input_vector = input_vector.reshape(1, -1)  # If your input is 1D, you need to reshape it to 2D

        # Make predictions using the trained model
        predictions = self.model.predict(input_vector)

        # Print the predictions
        print("Predictions:", predictions)
        return predictions[0]
