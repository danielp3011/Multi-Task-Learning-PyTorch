# Tutorial based on: https://pytorch.org/tutorials/beginner/saving_loading_models.html

from models.mti_net import MTINet 
from models.models import SingleTaskModel, MultiTaskModel 
import torch

from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_val_dataset, get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion
from utils.logger import Logger
from train.train_utils import train_vanilla
from evaluation.evaluate_utils import eval_model, validate_results, save_model_predictions,\
                                    eval_all_results
from termcolor import colored
import torch.nn as nn


# Load architecture of model
p = create_config("configs/env.yml", "configs/nyud/hrnet18/mti_net.yml", "Dsn_Dde=Dss")   
model = get_model(p) 
model = torch.nn.DataParallel(model)
model = model.cuda()  # device=device)
model.load_state_dict(torch.load(p['best_model']))

print(model)
for name, layer in model.named_modules():
    if isinstance(layer, nn.ReLU):
        print(name, layer)




# new_model = model 

# new_model = nn.Sequential(*list(model.children())[:-5])
# print(new_model)
# class feature_extractor(nn.Module): 
#     def __init__(self, output_layer=None): 
#         super().__init__() 
#         self.p = create_config("configs/env.yml", "configs/nyud/hrnet18/mti_net.yml", "Dsn_Dde=Dss")   
#         self.model = get_model(p)  
#         self.model = torch.nn.DataParallel(model) 
#         model = model.cuda() 
#         model.load_state_dict(torch.load(p["best_model"])) 

#         self.output_layer = output_layer 
#         self.layer = list(model.named_parameters()) 
#         self.layer_count = 0 
#         for l in self.layers

# count = 0 
# output_layer = "module.scale_3.decoders.depth.bias"
# for layer in model.named_parameters(): 
#     name, _ = layer
#     if name == output_layer:
#         break  
#     else:
#         count += 1 

# print(len(list(model.named_parameters())))

# for i in range(1, len(list(model.named_parameters()))): 
#     new_model = model._modules["module"].pop(list(model.named_parameters[-i]))
# new = model 
# print(new["module.scale_3.decoders.depth.bias"]) 
# print(list(model.named_parameters())[0])
# # Transforms 
# train_transforms, val_transforms = get_transformations(p)
# train_dataset = get_train_dataset(p, train_transforms)
# val_dataset = get_val_dataset(p, val_transforms)
# true_val_dataset = get_val_dataset(p, None) # True validation dataset without reshape 
# train_dataloader = get_train_dataloader(p, train_dataset)
# val_dataloader = get_val_dataloader(p, val_dataset)

# Load all task decoders without last layer
#semseg_decoder = torch.nn.Sequential(*list(model.module.heads.semseg.last_layer.children())[:-1])
# depth_decoder = torch.nn.Sequential(*list(model.module.heads.depth.last_layer.children())[:-1])
# normals_decoder = torch.nn.Sequential(*list(model.module.heads.normals.last_layer.children())[:-1])

# Replace model in decoder head with decoder, that doesn´t has last head
#model.module.heads.semseg.last_layer = semseg_decoder
# model.module.heads.depth.last_layer = depth_decoder
# model.module.heads.normals.last_layer = normals_decoder 

# Feature Extraction after Initial task predictions 
# new_model = torch.nn.Sequential(*list(model.module.children())[:-5])  # model for feature extraction


# print(new_model)

# with open('new_model2.txt', 'w') as f:
#     print('Filename:', new_model, file=f)  # Python 3.x 

# with open('model.txt2', 'w') as f:
#     print('Filename:', model, file=f)  # Python 3.x 

# for c in model.module.children():
#     print(c)
#     print("##########################################################")
    

# print("modules")
# for c in model.module.modules():
#     print(c) 
#     print("##########################################################")
#     break

# for name, layer in new_model.named_modules():
#     print(name)

# class FeatureExtractor(nn.Module): 
#     def __init__(self, model):
#         super(FeatureExtractor, self).__init__() 



# for layer in list(model):
#     if isinstance()

# print(list(model))
