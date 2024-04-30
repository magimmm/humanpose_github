from FaceVideoAnalysis import FaceAnalyser

print('cnn_model_v_8_16_2.pth without aug')
fvideo=FaceAnalyser('cnn_model_v_8_16_2.pth',using_mp=False,show=False,noise_correction=False,test_seq=False)
fvideo.run()

# print('cnn_model_v_8_16_aug.pth')
# fvideo=FaceAnalyser('cnn_model_v_8_16_2_aug.pth',using_mp=False,show=False,noise_correction=False,test_seq=False)
# fvideo.run()
