from VideoAnalysis import Analyser

#in case of mp - edit get landamrks so it wont read the img bc it is sent already loaded, not just path...
#videoanalyser= Analyser('mediapipe')

videoanalyser= Analyser('yolo')
videoanalyser.setup_model()
videoanalyser.run()
#------------------------
#false positivve... na videu totalnu uspesnost
