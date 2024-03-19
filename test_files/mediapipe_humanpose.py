# required libraries
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Loading the image using OpenCV.
img = cv2.imread("../Photos/Personal-Driver-in-Dubai.jpg")

# Getting the image's width and height.
img_width = img.shape[1]
img_height = img.shape[0]

# Creating a figure and a set of axes.
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('off')
ax.imshow(img[...,::-1])
plt.show()

# Initializing the Pose and Drawing modules of MediaPipe.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(static_image_mode=True) as pose:
    """
    This function utilizes the MediaPipe library to detect and draw 'landmarks'
    (reference points) on an image. 'Landmarks' are points of interest
    that represent various body parts detected in the image.

    Args:
        static_image_mode: a boolean to inform if the image is static (True) or sequential (False).
    """

    # Make a copy of the original image.
    annotated_img = img.copy()

    # Processes the image.
    results = pose.process(img)

    # Set the circle radius for drawing the 'landmarks'.
    # The radius is scaled as a percentage of the image's height.
    circle_radius = int(.007 * img_height)

    # Specifies the drawing style for the 'landmarks'.
    point_spec = mp_drawing.DrawingSpec(color=(220, 100, 0), thickness=-1, circle_radius=circle_radius)

    # Draws the 'landmarks' on the image.
    mp_drawing.draw_landmarks(annotated_img,
                              landmark_list=results.pose_landmarks,
                              landmark_drawing_spec=point_spec)


# Make a copy of the original image.
annotated_img = img.copy()

# Specifies the drawing style for landmark connections.
line_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

# Draws both the landmark points and connections.
mp_drawing.draw_landmarks(
annotated_img,
landmark_list=results.pose_landmarks,
connections=mp_pose.POSE_CONNECTIONS,
landmark_drawing_spec=point_spec,
connection_drawing_spec=line_spec
)

cv2.imshow('anp',annotated_img)
cv2.waitKey(0)
# Select the coordinates of the points of interest
# .
l_hip_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * img_width)
l_hip_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * img_height)

l_knee_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * img_width)
l_knee_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * img_height)

l_ankle_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * img_width)
l_ankle_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * img_height)

l_heel_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x * img_width)
l_heel_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y * img_height)

l_foot_index_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * img_width)
l_foot_index_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * img_height)

# Print the coordinates on the screen.
print('Left knee coordinates: (', l_knee_x,',',l_knee_y,')' )
print('Left ankle coordinates: (', l_ankle_x,',',l_ankle_y,')' )
print('Left heel coordinates: (', l_heel_x,',',l_heel_y,')' )
print('Left foot index coordinates: (', l_foot_index_x,',',l_foot_index_y,')' )

# Displaying a graph with the selected points.
fig, ax = plt.subplots()
ax.imshow(img[:, :, ::-1])
ax.plot([l_hip_x,l_knee_x, l_ankle_x, l_heel_x, l_foot_index_x], [l_hip_y,l_knee_y, l_ankle_y, l_heel_y, l_foot_index_y], 'ro')
plt.show()

