from ultralytics import YOLO
import cv2
# Load a model
model = YOLO('yolov8n-pose.pt')

# Predict with the model
results = model(source="Screenshot 2024-02-09 162507.png", show=True, save=True)
#print(results)

img= cv2.imread('Screenshot 2024-02-09 162507.png')

print(results[0].keypoints.xy)
print('nnnnnnnnnnnn')
landmarks = [landmark.cpu().numpy() for landmark in results[0].keypoints.xy]
landmarks=landmarks[0]
for landmark in landmarks:
    # Check if the landmark is not null
    if landmark[0] != 'null' and landmark[1] != 'null':
        # Convert landmark coordinates to integers
        landmark_x = int(landmark[0])
        landmark_y = int(landmark[1])
        print(landmark_x,landmark_y)
        # Draw a circle at the landmark position
        cv2.circle(img, (landmark_x, landmark_y), 4, (255, 120, 0), 1, 1)
cv2.imshow('m',img)
cv2.waitKey(0)

