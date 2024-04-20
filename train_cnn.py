from cnn import CNN

# cnn.create_cutouts(normal_path='Photos/all_images/3/normal1', abnormal_path='Photos/all_images/3/abnormal1')
# cnn.create_cutouts2()
cnn=CNN()
cnn.train_model_4_class(folder_path='Photos/neuron_convolution_original',model_name="cnn_model_v_8_20_2_aug.pth",augument=True,batch=8,epoch=20,lr=0.001)
#                                                                                v/mediapipe, batches, epochs, nulls