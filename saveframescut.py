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
    abnormal_dir = "Photos/all_images/2/abnormal1"
    normal_dir = "Photos/all_images/2/normal1"
    os.makedirs(abnormal_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    frame_count = 0

    # Read and save frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

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
                    cv2.imwrite(f"{abnormal_dir}/2abnormal_{frame_count}_{action}.jpg", frame)
                    break

        # If not abnormal, check if the frame falls within any normal range
        else:
            cv2.imwrite(f"{normal_dir}/2normal_{frame_count}.jpg", frame)

    cap.release()

# Define abnormal and normal ranges (start_frame, end_frame, action_name)
abnormal_ranges = [
    (248, 387, "cough"),
    (616, 807, "cough"),
    (942, 1058, "cough"),
    (1134, 1162, "cough"),
    (2125, 2228, "cough"),
    (2731, 2857, "cough"),
    (3233, 3415, "cough"),
    (3610, 3682, "cough"),
    (3747, 3787, "hand_wiping"),
    (4340, 4410, "cough"),
    (4441, 4487, "cough"),
    (4911, 5057, "cough"),
    (5106, 5130, "hand_wiping"),
    (5242, 5290, "nose_wiping"),
    (5872, 6091, "cough"),
    (6185, 6210, "hand_wiping"),
    (6608, 6718, "cough"),
    (6948, 7018, "cough"),
    (7806, 7963, "cough"),
    (8807, 8915, "cough"),
    (8960, 8985, "hand_wiping"),
    (9636, 9672, "cough"),
    (9980, 10005, "cough"),
    (10033, 10115, "hand_wiping"),
    (12014, 12201, "cough"),
    (12253, 12286, "cough"),
    (12313, 12360, "hand_wiping"),
    (12625, 12677, "cough"),
    (12751, 12786, "cough"),
    # Add more abnormal ranges as needed
]



# Call the function to save frames

video_path = "Videos/anomal_3.mp4"

save_frames(video_path, abnormal_ranges)
