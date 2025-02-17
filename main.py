from utils.get_arg import get_args
import warnings
import torch
import os
import logging
import json
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    warnings.simplefilter("ignore")

    args = get_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('log/main_{}_{}_train_ratio_{}_val_ratio_{}.log'.format(args.config[7:-4],args.data,args.train_ratio,args.val_ratio))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.cuda.set_device(int(args.gpu))
    device = torch.device(f'cuda:{args.gpu}')
    set_seed(args.seed)
        
    logger.info('Main test start')
    logger.info(args)
    selected_feature_file_path = f"./selected_features/{args.data}.json"

    ## Feature Selection ##
    if os.path.exists(selected_feature_file_path):
        logger.info('Find proper feature selection file')
        with open(selected_feature_file_path, 'r') as file:
            selected_feature = json.load(file)['selected_feat']
    else:
        logger.info('Proper feature selection start')
        if args.task in ['anomaly', 'class']:
            from feature_selection import feature_select
        elif args.task in ['affinity']:
            from feature_selection_TGB import feature_select
        else:
            raise NotImplementedError
        
        selected_feature, _ = feature_select(args, logger)
    
    logger.info('Selected feature: {}'.format(selected_feature))
    args.selected_feature = selected_feature
    
    ## model run ##
    if args.task in ['anomaly', 'class']:
        from train_node_property_prediction import objective
    elif args.task in ['affinity']:
        from train_node_property_prediction_TGB import objective
    
    result = objective(args, logger)
    logger.info('Main test end')