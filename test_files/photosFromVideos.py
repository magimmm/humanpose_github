import cv2
import os

# Open the video file
cap = cv2.VideoCapture('../Videos/normal.mp4')

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Create directory to store extracted frames
# Set interval for extracting frames (in seconds)


# Initialize frame counter
count = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 26111)

# Loop through video frames
while True:
    # Read a frame from the video file
    ret, frame = cap.read()

    # If frame reading was unsuccessful, exit the loop
    if not ret:
        break

    # Increment frame counter
    count += 1

    # Save the extracted frame as an image file
    path='../Photos/fr/'+str(count)+'.jpg'
    cv2.imwrite(path, frame)


# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()
