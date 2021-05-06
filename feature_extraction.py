# Tutorial based on: https://pytorch.org/tutorials/beginner/saving_loading_models.html

from models.mti_net import MTINet 
from models.models import SingleTaskModel, MultiTaskModel 
import torch

# Load best model
model = torch.load("/home/data2/yd/results_yd/mtlpt/NYUD/hrnet_w18/single_task/depth/best_model.pth.tar")
print("State dict : ", model.state_dict())

# features = model.last_layer 
# print(features)

#print("model load of best model: ", model) 

# # Load last checkpoint 
# checkpoint = torch.load("/home/data2/yd/results_yd/mtlpt/NYUD/hrnet_w18/single_task/depth/checkpoint.pth.tar", map_location='cpu')
# optimizer.load_state_dict(checkpoint['optimizer'])
# model.load_state_dict(checkpoint['model'])
# # start_epoch = checkpoint['epoch']
# # best_result = checkpoint['best_result']
# print("optimizer: ", optimizer) 
# print("model: ", model)

