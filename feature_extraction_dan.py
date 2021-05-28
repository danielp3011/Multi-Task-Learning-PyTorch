
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


# class FeatureExtractor(nn.Module):
#     def __init__(self, submodule, extracted_layers):
#         self.submodule = submodule

#     def forward(self, x):
#         outputs = []
#         for name, module in self.submodule._modules.items():
#             x = module(x)
#             if name in self.extracted_layers:
#                 outputs += [x]
#         return outputs + [x]


# fe = FeatureExtractor(self, model, ["Conv2d"])
# fe()

##----

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.extracted_layers= extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            print(name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

# myresnet=resnet18(pretrained=False)
# num_ftrs=model.fc.in_features
# myresnet.fc=nn.Linear(num_ftrs,10)
# exact_list=["conv1","layer1","avgpool"]
# myexactor=FeatureExtractor(myresnet,exact_list)
# a=torch.randn(1,3,32,32)
# a=Variable(a)
# x=myexactor(a)

print(model.module.fc.in_features)
