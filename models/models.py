from .mobilevit import MobileViT_S
from torchvision import models
import torch

NUMBER_OF_CLASSES = 23 

def get_model(name='MobileNet2'):
	"""
	Return the selected model for training
	"""
	if name == "MobileNet2":
		return get_mobilenet_v2()

	elif name == "MobileNet3":
		return get_mobilenet_v3()

	elif name == "EfficientNet-B0":
		return get_efficientnet_b0()

	elif name == "MobileViT":
		return get_mobilevit()

	else:
		raise ValueError("Please select a model from the following: ['MobileNet2', 'MobileNet3', 'EfficientNet-B0', "
						 "'MobileViT'].")


def get_mobilenet_v2():
	"""
	Load MobileNet V2 pretrained model from PyTorch
	"""
	model = models.mobilenet_v2(pretrained=True)

	# Freeze the first layers as mentioned in the paper (B. Training Details): https://arxiv.org/pdf/1803.00949.pdf
	freeze_first_layer = True

	for param in model.features.parameters():
	  param.requires_grad = True

	for param in model.classifier.parameters():
	  param.requires_grad = True

	# Freeze the first layer of the model
	"""if freeze_first_layer:
		for param in model.features[0].parameters():
			param.requires_grad = False
	"""
	model.classifier[1].out_features = NUMBER_OF_CLASSES

	return model

def get_mobilenet_v3():
	"""
	Load MobileNet V3 pretrained model from PyTorch
	"""
	model = models.mobilenet_v3_large(pretrained=True)

	# Freeze the first layers as mentioned in the paper (B. Training Details): https://arxiv.org/pdf/1803.00949.pdf
	freeze_first_layer = True

	for param in model.features.parameters():
	  param.requires_grad = True

	for param in model.classifier.parameters():
	  param.requires_grad = True

	# Freeze the first layer of the model
	"""if freeze_first_layer:
		for param in model.features[0].parameters():
			param.requires_grad = False
	"""
	model.classifier[3].out_features = NUMBER_OF_CLASSES

	return model

def get_efficientnet_b0():
	"""
	Load EfficientNet-B0 pretrained model from PyTorch
	"""
	model = models.efficientnet_b0(pretrained=True)

	# Freeze the first layers as mentioned in the paper (B. Training Details): https://arxiv.org/pdf/1803.00949.pdf
	freeze_first_layer = True

	for param in model.features.parameters():
	  param.requires_grad = True

	for param in model.classifier.parameters():
	  param.requires_grad = True

	# Freeze the first layer of the model
	"""if freeze_first_layer:
		for param in model.features[0].parameters():
			param.requires_grad = False
	"""
	model.classifier[1].out_features = NUMBER_OF_CLASSES

	return model

def load_mobilevit_weights(model_path):
	# Create an instance of the MobileViT model
	net = MobileViT_S()

	# Load the PyTorch state_dict
	state_dict = torch.load(model_path)['state_dict']

	# Since there is a problem in the names of layers, we will change the keys to meet the MobileViT model architecture
	for key in list(state_dict.keys()):
		state_dict[key.replace('module.', '')] = state_dict.pop(key)

	# Once the keys are fixed, we can modify the parameters of MobileViT
	net.load_state_dict(state_dict)

	return net

def get_mobilevit():
	"""
	Load MobileViT pretrained model from Github : 
		https://github.com/wilile26811249/MobileViT/blob/main/main.py
	"""
	model = load_mobilevit_weights("models/pretrained/MobileViT_S_model_best.pth.tar")

	# Freeze the first layers as mentioned in the paper (B. Training Details): https://arxiv.org/pdf/1803.00949.pdf
	freeze_first_layer = True

	# Change the number of output neurons
	model.fc.out_features = NUMBER_OF_CLASSES

	# Train all layers
	for param in model.stem.parameters():
	  param.requires_grad = True 

	for param in model.stage1.parameters():
	  param.requires_grad = True 

	for param in model.stage2.parameters():
	  param.requires_grad = True 

	for param in model.stage3.parameters():
	  param.requires_grad = True 

	for param in model.stage4.parameters():
	  param.requires_grad = True

	# Set the classification layers to trainable
	for param in model.fc.parameters():
	  param.requires_grad = True

	# Freeze the first layer of the model
	"""if freeze_first_layer:
		for param in model.stem[0].parameters():
			param.requires_grad = False
	"""
	return model