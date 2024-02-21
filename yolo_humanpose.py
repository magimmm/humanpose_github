from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')

# Predict with the model
results = model(source='teste.jpg',show=True,save=True)
print(results)

results[0].keypoints.xy.xyn
