import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

from YoloSkeleton import YoloSkeleton

def from_whole_body(skeletons):
    skeletons_distances_list=[]
    for skeleton in skeletons:
        skeletons_distances_list.append(distance_list_from_body_list(skeleton.body))

    array_of_arrays = np.array([np.array(sublist) for sublist in skeletons_distances_list])
    return array_of_arrays
def create_nn_input_from_skeletons(skeletons):
    skeletons_distances_list = []
    skeleton: YoloSkeleton
    for skeleton in skeletons:
        skeleton.calculate_arms_distances()
        skeletons_distances_list.append([skeleton.right_wrist_to_elbow,skeleton.right_elbow_to_shoulder,skeleton.right_wrist_to_shoulder,skeleton.left_wrist_to_elbow,skeleton.left_elbow_to_shoulder,skeleton.left_wrist_to_shoulder])

    array_of_arrays = np.array([np.array(sublist) for sublist in skeletons_distances_list])
    return array_of_arrays
    #return from_whole_body(skeletons)

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
def distance_list_from_body_list(body_keypoints):
    distance_list=[]
    for index,keypoint in enumerate(body_keypoints):
        if index < len(body_keypoints) - 1:
            distance=calculate_distance(keypoint[0],keypoint[1],body_keypoints[index+1][0],body_keypoints[index+1][1])
            distance_list.append(distance)
    return distance_list

def run_neuron_network(normal_skeletons,abnormal_skeletons,model):
    # Generate sample data
    num_samples = 100
    if model == 'mediapipe':
        num_features = 13
    elif model == 'yolo':
        # TODO len ruky aktualne
        num_features = 6

    abnormal_data=create_nn_input_from_skeletons(abnormal_skeletons)
    normal_data=create_nn_input_from_skeletons(normal_skeletons)

    #abnormal =0
    #normal=1
    # Generate normal data
    #normal_data = np.random.normal(loc=0, scale=1, size=(num_samples // 2, num_features))

    # Generate abnormal data
    #abnormal_data = np.random.uniform(low=-2, high=2, size=(num_samples // 2, num_features))

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
        Dense(16, activation='selu', input_shape=(num_features,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    custom_optimizer = SGD(learning_rate=0.001)
    # Compile the model
    model.compile(optimizer=custom_optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=40, batch_size=32, validation_split=0.2)

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
