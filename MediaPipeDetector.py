import cv2
import mediapipe as mp

class MediaPipeDetector():
    def __init__(self):
        self.deteced_landmarks = None

    def setup_detector(self):
        self.mp_pose = mp.solutions.pose
        self.pose_image = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5,model_complexity=0)

        # Initialize mediapipe drawing class - to draw the landmarks points.
        self.mp_drawing = mp.solutions.drawing_utils


    def get_landmarks(self,img_path):
        all_landmarks = []

        img=cv2.imread(img_path)
        # TODO zmen na false
        annotated_image, result = self.detectPose(img,False)
        if not result.pose_landmarks or not result.pose_landmarks.landmark:
            print(img_path)
        else:
            deteced_landmarks = result.pose_landmarks.landmark
            img_width = img.shape[1]
            img_height = img.shape[0]
            index_left_hip = 24

            # Initialize lists to store the x and y coordinates of all landmarks
            # prejde vsetky landmarky a ak nejaky chyba, da mu nejaku specificku hodnotu
            for (i,landmark) in enumerate(deteced_landmarks):
                #print(landmark.presence)
                #print(landmark.visibility)
                # TODO upravit podmienku lebo vsade je presence 0 a visibility je od 0-1
                if float(landmark.visibility)>0.6:
                    landmark_x = int(landmark.x * img_width)
                    landmark_y = int(landmark.y * img_height)
                    all_landmarks.append([landmark_x, landmark_y])
                else:

                    #print(img_path,i,'missing', landmark.visibility)
                    # TODO nejaka hodnota pre nedetekovane landmarky
                    all_landmarks.append([0,0])
                # Check if the current landmark is the one you want to stop at
                if i == index_left_hip:
                    break  # Stop the loop if the landmark is reached

                # TODO vymazat
                # Display the image with pose landmarks
            # cv2.namedWindow('Pose Landmarks', cv2.WINDOW_NORMAL)
            # cv2.imshow('Pose Landmarks', annotated_image)
            # cv2.waitKey(0)

        return all_landmarks

    def detectPose(self, image_input, draw):
        original_image = image_input.copy()

        image_in_RGB = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

        resultant = self.pose_image.process(image_in_RGB)

        if resultant.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(image=original_image, landmark_list=resultant.pose_landmarks,
                                      connections=self.mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                   thickness=3, circle_radius=3),
                                      connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(49, 125, 237),
                                                                                     thickness=2, circle_radius=2))

        return original_image, resultant



