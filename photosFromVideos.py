import cv2
import os

# Open the video file
cap = cv2.VideoCapture('Videos/anormal.mp4')

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Create directory to store extracted frames
if not os.path.exists('frames'):
    os.makedirs('frames')

# Set interval for extracting frames (in seconds)
interval = 0.2

# Initialize frame counter
count = 0

# Loop through video frames
while True:
    # Read a frame from the video file
    ret, frame = cap.read()

    # If frame reading was unsuccessful, exit the loop
    if not ret:
        break

    # Increment frame counter
    count += 1

    # Extract frame at specified interval
    if count % (interval * cap.get(cv2.CAP_PROP_FPS)) == 0:
        # Save the extracted frame as an image file
        cv2.imwrite(f'Photos/abnormal/abnormal{count}.jpg', frame)


# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()
