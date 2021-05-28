#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import sys
from yd_utils.reporting_email import send_email
from yd_utils.yd_config import server_name
############################## 
target_mail_address_list = ["y.vorpahl@hotmail.de", "daniel.pietschmann@outlook.de"]
##############################

import argparse
import cv2
import os
import numpy as np
import sys
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



# Parsers
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--save_name', help='folder_name, where results are saved (only for multi-task mode)', type=str) 
args = parser.parse_args()

def main():

    ################
    feature_extraction = True
    ################

    #try:
    # Retrieve config file 
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp, args.save_name)
    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    model = torch.nn.DataParallel(model)
    model = model.cuda()  # device=device)

    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda()  # device=device)
    print(criterion)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    train_transforms, val_transforms = get_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms)
    val_dataset = get_val_dataset(p, val_transforms)
    true_val_dataset = get_val_dataset(p, None) # True validation dataset without reshape 
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    print('Train transformations:')
    print(train_transforms)
    print('Val transformations:')
    print(val_transforms)
    
    # Evaluate best model at the end
    print(colored('Evaluating best model at the end', 'blue'))
    model.load_state_dict(torch.load(p['best_model'])) 
    
    if "feature_extract_final":
        # Load all task decoders without last layer & replace model in decoder head with decoder, that doesn´t has last head
        if p["setup"] == "single_task": 
            model.module.decoder.last_layer = torch.nn.Sequential(*list(model.module.decoder.last_layer.children())[:-1])    
            print("SINGLE TASK")
        else:
            for task_name in p.TASKS.NAMES: 
                model.module.heads[task_name].last_layer = torch.nn.Sequential(*list(model.module.heads[task_name].last_layer.children())[:-1])    
                print("MULTI-TASK") 
    
    elif "feature_extract_scale":
        scales = [0,1,2,3]
        for sc in scales:
            if p["setup"] == "single_task": 
                print("SINGLE TASK - no extraction at different scales!")
                sys.exit()
            else:
                new_model = torch.nn.Sequential(*list(model.module.modules())[:-5])
                print("MULTI-TASK") 

        
    # Load all task decoders without last layer 
    #human_parts = torch.nn.Sequential(*list(model.module.heads.human_parts.last_layer.children())[:-1])
    #semseg_decoder = torch.nn.Sequential(*list(model.module.heads.semseg.last_layer.children())[:-1])
    # depth_decoder = torch.nn.Sequential(*list(model.module.heads.depth.last_layer.children())[:-1])
    # normals_decoder = torch.nn.Sequential(*list(model.module.heads.normals.last_layer.children())[:-1])
    #sal_decoder = torch.nn.Sequential(*list(model.module.heads.sal.last_layer.children())[:-1])


    # Replace model in decoder head with decoder, that doesn´t has last head
    #model.module.heads.semseg.last_layer = semseg_decoder
    # model.module.heads.depth.last_layer = depth_decoder
    # model.module.heads.normals.last_layer = normals_decoder 
    #model.module.heads.human_parts.last_layer = human_parts
    #model.module.heads.sal.last_layer = sal_decoder

    # print("MODELS: ", model)

    #print("Model state dict all: ", model.state_dict().items()) 
    save_model_predictions(p, val_dataloader, model, feature_extraction, args.save_name)
    # eval_stats = eval_all_results(p)
    send_email(target_mail_address_list, server_name=server_name, exception_message="Success!", successfully=True)

    # except Exception:
    #     print(str(sys.exc_info()))
    #     send_email(target_mail_address_list, server_name=server_name, exception_message=str(sys.exc_info()), successfully=False)

if __name__ == "__main__":
    main() 
