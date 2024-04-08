from ultralytics import YOLO

models = ['yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt']
model = YOLO('yolov8n-pose.pt')
for path in images_paths:
    results = model(source=path, show=False, save=False, verbose=False, max_det=1)
    # detected_landmarks = [landmark.cpu().numpy() for landmark in results[0].keypoints.xy][0]
