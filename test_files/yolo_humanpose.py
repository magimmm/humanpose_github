from ultralytics import YOLO
import cv2
# Load a model
model = YOLO('../yolov8n-face.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model.predict('../Photos/Personal-Driver-in-Dubai.jpg')  # return a list of Results objects

print(results)


# model = YOLO('../yolov8n-pose.pt')
#
# # Predict with the model
# results = model(source='../Photos/neuron_body/train/abnormal/2abnormal_6960_cough.jpg', show=True, save=True)
# #print(results)

# img= cv2.imread('../Photos/neuron_body/train/abnormal/2abnormal_6960_cough.jpg')
#
# print(results[0].keypoints.xy)
# print('nnnnnnnnnnnn')
# l=results[0].keypoints
# box=results[0].boxes[0]
# landmarks = [landmark.cpu().numpy() for landmark in results[0].keypoints.xy]
# landmarks=landmarks[0]
# top_left_x=int(box.xyxy.tolist()[0][0])
# top_left_y=int(box.xyxy.tolist()[0][1])
# bottom_right_y=int(box.xyxy.tolist()[0][2])
# bottom_right_x=int(box.xyxy.tolist()[0][3])
#
# cv2.rectangle(img,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(50,200,129),2)
# for landmark in landmarks:
#     # Check if the landmark is not null
#
#     if landmark[0] != 0 and landmark[1] != 0:
#         # Convert landmark coordinates to integers
#         landmark_x = int(landmark[0])
#         landmark_y = int(landmark[1])
#         print(landmark_x,landmark_y)
#         # Draw a circle at the landmark position
#         cv2.circle(img, (landmark_x, landmark_y), 4, (255, 120, 0), 1, 1)
# cv2.imshow('m',img)
# cv2.waitKey(0)
#
# #results = model(['bus.jpg', 'zidane.jpg'])  # list of 2 Results objects