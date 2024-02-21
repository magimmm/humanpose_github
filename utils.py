# Extract keypoint

print(results)
print('-------------------')
result_keypoint = results[0].keypoints.xyn.cpu().numpy()[0]
print(result_keypoint)

for result in results:
    # Each 'result' corresponds to a detected person
    keypoints = result['keypoints']
    for keypoint in keypoints:
        # 'keypoint' is a dictionary with 'name', 'x', 'y', and 'confidence'
        x, y, confidence = keypoint['x'], keypoint['y'], keypoint['confidence']
        # Use the ke

        import json
        import os

        import cv2
        from MediapipeSkeleton import MediaPipeSkeleton
        from YoloSkeleton import YoloSkeleton


        def get_image_files_in_folder(folder_path):
            image_files = []
            image_filenames = []
            for file_name in os.listdir(folder_path):

                # Check if the file is an image (you can add more extensions if needed)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_files.append(os.path.join(folder_path, file_name))
                    image_filenames.append(file_name)
            return image_files, image_filenames


        def photo_landmarks(filename):
            # Load JSON data from a file
            with open('annotation/person_keypoints_default.json', 'r') as file:
                data = json.load(file)

            images = data['images']
            for i in images:
                if i['file_name'] == filename:
                    id_img = i['id']

            annotations = data['annotations']
            for a in annotations:
                if id_img == a['id']:
                    return a['keypoints']

            return "Something went wrong"


        def media_pipe_model_from_keypoints(keypoints):
            skeleton = MediaPipeSkeleton(keypoints)


        def create_skeletons_from_annotations(images_normal_annotated,
                                              images_filenames, model='mediapipe'
                                              ):
            skeletons_images = []
            for (i, filename) in enumerate(images_filenames):
                annotated_landmarks = photo_landmarks(filename)
                # body zistene z anotovaneho suboru zoberie a vytvori z nich instanciu triedy
                if model == 'mediapipe':
                    skeleton = MediaPipeSkeleton(annotated_landmarks, images_normal_annotated[i])
                elif model == 'yolo':
                    skeleton = YoloSkeleton(annotated_landmarks, images_normal_annotated[i])

                skeletons_images.append(skeleton)
            return skeletons_images


        def view_annotated_landmarks(skeletons):
            for (i, filename) in enumerate(images_filenames):
                annotated_landmarks = photo_landmarks(filename)
                # body zistene z anotovaneho suboru zoberie a vytvori z nich instanciu triedy
                skeleton = MediaPipeSkeleton(annotated_landmarks, images_normal_annotated[i])
                img = cv2.imread(images_normal_annotated[i])
                for j in skeleton.all_landmarks:
                    print(j)
                    cv2.circle(img, (int(j[0]), int(j[1])), 4, (255, 120, 255), 1, 1)
                cv2.imshow('jm', img)
                cv2.waitKey(0)


        def view_annotated_landmarks(skeletons):
            for skeleton in skeletons:
                img = cv2.imread(skeleton.path)
                for j in skeleton.all_landmarks:
                    print(j)
                    cv2.circle(img, (int(j[0]), int(j[1])), 4, (0, 120, 255), 1, 1)
                cv2.imshow('anotated_landmarks', img)
                cv2.waitKey(0)


        path_normal_images = 'Photos/normal_select'
        images_normal_annotated, images_filenames = get_image_files_in_folder(path_normal_images)
        skeletons = create_skeletons_from_annotations(images_normal_annotated, images_filenames, model='yolo')
        view_annotated_landmarks(skeletons)
