import cv2
import os

import cv2
import os

# Function to save frames to the specified directory
def save_frames(video_path, abnormal_ranges):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return

    # Create directories for abnormal and normal frames if they don't exist
    abnormal_dir = "Photos/all_images/3/abnormal1"
    normal_dir = "Photos/all_images/3/normal1"
    os.makedirs(abnormal_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    frame_count = 0

    # Read and save frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count>5034:
            return

        # Check if the frame falls within any abnormal range
        abnormal = False
        for start, end, action in abnormal_ranges:
            if start <= frame_count <= end:
                abnormal = True
                break

        # If abnormal, save to abnormal directory with action name
        if abnormal:
            for start, end, action in abnormal_ranges:
                if start <= frame_count <= end:
                    cv2.imwrite(f"{abnormal_dir}/3abnormal_{frame_count}_{action}.jpg", frame)
                    break

        # If not abnormal, check if the frame falls within any normal range
        else:
            cv2.imwrite(f"{normal_dir}/3normal_{frame_count}.jpg", frame)

    cap.release()

# Define abnormal and normal ranges (start_frame, end_frame, action_name)
abnormal_ranges = [
    (435, 490, "cough"),
    (715, 788, "cough"),
    (1082, 1152, "yawn"),
    (1413, 1500, "cough"),
    (2040, 2086, "cough"),
    (2385, 2420, "cough"),
    (2605, 2645, "cough"),
    (2910, 2955, "cough"),
    (3005, 3140, "scratch_cough"),
    (3388, 3510, "cough"),
    (3905, 5032, "phone"),
    # Add more abnormal ranges as needed
]



# Call the function to save frames

video_path = "Videos/anomal_3.mp4"

save_frames(video_path, abnormal_ranges)
