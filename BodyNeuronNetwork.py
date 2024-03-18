import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import save_model, load_model


from YoloSkeleton import YoloSkeleton

class NeuronNetworkManager:
    def __init__(self):
        self.model = None
        self.num_features = 0
    def create_nn_input_from_skeletons(self,skeletons):
        skeletons_distances_list = []
        skeleton: YoloSkeleton
        for skeleton in skeletons:
            # TODO vsetky vzdialenosti
            #skeletons_distances_list.append(skeleton.features_vector_whole_body)
            skeletons_distances_list.append(skeleton.features_vector)
        array_of_arrays = np.array([np.array(sublist) for sublist in skeletons_distances_list])
        return array_of_arrays

    def train_model(self,normal_skeletons,abnormal_skeletons):
        self.num_features = len(normal_skeletons[0].features_vector)
        # Generate sample data
        num_samples = 100

        abnormal_data=self.create_nn_input_from_skeletons(abnormal_skeletons)
        normal_data=self.create_nn_input_from_skeletons(normal_skeletons)

        # Create labels (0 for normal, 1 for abnormal)
        labels = np.concatenate([np.zeros(num_samples // 2), np.ones(num_samples // 2)])

        # Combine normal and abnormal data
        data = np.concatenate([normal_data, abnormal_data])
        # for d in data:
        #     print(d)
        # print(data)

        # Shuffle data and labels
        p = np.random.permutation(num_samples)
        data = data[p]
        labels = labels[p]

        # Split data into training and testing sets
        train_ratio = 0.8
        num_train = int(train_ratio * num_samples)
        x_train, x_test = data[:num_train], data[num_train:]
        y_train, y_test = labels[:num_train], labels[num_train:]

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
        model.fit(x_train, y_train, epochs=40, batch_size=32, validation_split=0.1)

        # Evaluate the model on test data
        loss, accuracy = model.evaluate(x_test, y_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

        model.save("my_model")

    def load_model(self):
        self.model = load_model("my_model")

    def predict_img(self,features_vector):
        input_vector = np.array([features_vector])

        # Reshape the input vector to match the expected shape of the input data for the model
        input_vector = input_vector.reshape(1, -1)  # If your input is 1D, you need to reshape it to 2D

        # Make predictions using the trained model
        predictions = self.model.predict(input_vector)

        # Print the predictions
        print("Predictions:", predictions)
        return predictions[0]

