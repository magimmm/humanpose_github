from cnn import CNN

cnn=CNN()
cnn.train_model_4_class(folder_path='Photos/neuron_convolution_original',model_name="cnn_model_v_8_20_2_aug.pth",augument=True,batch=8,epoch=20,lr=0.001)
#                                                                                v/mediapipe, batches, epochs, nulls