from ultralytics import YOLO
import time

class YoloDetector():
    def __init__(self,model):
        self.model = None
        self.model_type=model

    def setup(self):
        # Load a model
        self.model = YOLO(self.model_type)

    def get_landmarks(self,img):
        #ziska landmarky priamo z modelu
        left_hip_index=12
        # Predict with the model
        start_time=time.time()
        results = self.model(source=img, show=False, save=False,verbose=True)
        landmarks = [landmark.cpu().numpy() for landmark in results[0].keypoints.xy]
        detected_landmarks = landmarks[0]
        end_time=time.time()
        time_detection =end_time-start_time
        all_landmarks = []
        # prejde vsetky landmarky a ak nejaky chyba, da mu nejaku specificku hodnotu
        for i,landmark in enumerate(detected_landmarks):
            if landmark[0] == 0 and landmark[1]==0:
                # TODO nejaka hodnota pre nedetekovane landmarky
                all_landmarks.append([0, 0])
                print(i,'missing')
            else:
                landmark_x = int(landmark[0])
                landmark_y = int(landmark[1])
                all_landmarks.append([landmark_x, landmark_y])
            # Check if the current landmark is the one you want to stop at
            if i==left_hip_index:
                break  # Stop the loop if the landmark is reached
        return all_landmarks,time_detection

