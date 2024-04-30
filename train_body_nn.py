from BodyNeuronNetwork import NeuronNetworkManager

body_nn_manager = NeuronNetworkManager()
body_nn_manager.train_model('Photos/neuron_body/train/normal', 'Photos/neuron_body/train/abnormal')
