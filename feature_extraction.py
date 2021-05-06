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

# Load best model
p = create_config("configs/env.yml", "configs/nyud/hrnet18/mti_net.yml", "all")   
model = get_model(p) 
model = torch.nn.DataParallel(model)
model = model.cuda()  # device=device)
model.load_state_dict(torch.load(p['best_model'])) 
print(model)


#print("Model: ", model) 
#model.to(torch.cuda)
#model.load_state_dict(torch.load("/home/data2/yd/results_yd/mtlpt/NYUD/hrnet_w18/mti_net/all/best_model.pth.tar")) 
#print("Model: ", model)
#features = list(model.last_layer)
#print(features)

