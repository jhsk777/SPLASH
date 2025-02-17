import argparse
import os
import hashlib
import optuna
from functools import partial
import torch
import time
import random
import dgl
import numpy as np
import pandas as pd
import pdb
from modules.modules import *
from utils.sampler import *
from utils.utils import *
from modules.layers import Global_TimeEncode, Global_TimeEncode
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import pickle
import logging
from tgb.nodeproppred.evaluate import Evaluator
from node2vec import Node2Vec
from scipy.stats import wasserstein_distance

from scipy.special import kl_div
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import json



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#processed_edge_id = 0
def feature_select(args, logger):
    TASK = args.task
    HIDDEN_DIM = args.hidden_dim
    ldf = pd.read_csv('DATA/{}/labels.csv'.format(args.data))
    if args.data in ['WIKI', 'REDDIT', 'MOOC']:
        args.task = 'anomaly'
    elif args.data in ['Email-EU', 'GDELT_sample', 'synthetic-70', 'synthetic-90']:
        args.task = 'class'
    else:
        raise ValueError('Task not supported')

    if args.split_num == 5:
        train_val_ratio = [0.1, 0.3, 0.5, 0.7, 0.9]
    elif args.split_num == 10:
        train_val_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    elif args.split_num == 4:
        train_val_ratio = [0.2, 0.4, 0.6, 0.8]
    total_mean_rand_val_loss = []
    total_mean_pos_val_loss = []
    total_mean_str_val_loss = []


    for ratio in train_val_ratio:
        print('train_val_ratio: {}'.format(ratio))
        ldf = pd.read_csv('DATA/{}/labels.csv'.format(args.data))
        role = ldf['ext_roll'].values
        labels = ldf['label'].values.astype(np.int64)
        class_num = labels.max() + 1

        train_edge_end = int(len(role) * args.val_ratio * ratio)
        val_edge_end = int(len(role) * args.val_ratio)
        test_start = val_edge_end

        train_end_time = ldf.time[train_edge_end]
        val_end_time = ldf.time[val_edge_end]

        train_ldf = ldf[ldf.time < train_end_time]
        val_ldf = ldf[(ldf.time >= train_end_time) & (ldf.time < val_end_time)]
        test_ldf = ldf[ldf.time >= val_end_time]
    

        node_feats, edge_feats = load_feat(args.data)
        g, df = load_graph(args.data)
        if args.task == 'tgbn':
            df['edge_weight'] = edge_feats
        train_df = df[df.time < train_end_time]

        node_count = int(max(df.src.max(), df.dst.max())) + 1
        train_node_max = int(max(train_df.src.max(), train_df.dst.max()))
        train_node_index = np.unique(np.concatenate([train_df.src.values, train_df.dst.values]))
        total_node_index = np.unique(np.concatenate([df.src.values, df.dst.values]))
        new_node_index = np.setdiff1d(total_node_index, train_node_index) 

        train_time = train_end_time - df.time.min()



        node_feats = torch.zeros(node_count, HIDDEN_DIM)
        train_adj = torch.zeros((train_node_max+1, train_node_max+1))
        train_G = nx.Graph()
        for i, rows in train_df.iterrows():
            if train_G.has_edge(rows.src.astype(np.int32), rows.dst.astype(np.int32)):
                train_G[rows.src.astype(np.int32)][rows.dst.astype(np.int32)]['weight'] += 1
            else:
                train_G.add_edge(rows.src.astype(np.int32), rows.dst.astype(np.int32), weight=1)
        node2vec = Node2Vec(train_G, dimensions=HIDDEN_DIM, walk_length=10, num_walks=80, workers=1, weight_key='weight', p=10)
        node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
        for node in train_G.nodes():
            node_feats[node] = torch.tensor(node2vec_model.wv[str(node)])        
        rand_feats = torch.randn(node_count, HIDDEN_DIM).cuda()
        rand_feats[new_node_index] = 0
            

        if node_feats is not None:
            fixed_node_feats = node_feats.clone()
            fixed_rand_feats = rand_feats.clone()
        sample_param, memory_param, gnn_param, train_param = parse_config(args.config)


        if args.set_degree:
            if type(args.n_degree) == int:
                sample_param['neighbor'] = [args.n_degree]
            #1l10deg
            elif args.n_degree[0] == '1':
                sample_param['neighbor'] = [int(args.n_degree[2:-3])]
                sample_param['layer'] = 1
                gnn_param['layer'] = 1
            elif args.n_degree[0] == '2':
                sample_param['neighbor'] = [int(args.n_degree[2:-3]), int(args.n_degree[2:-3])]
                sample_param['layer'] = 2
                gnn_param['layer'] = 2


        gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
        gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

        # To get node degree, we need to one more sampling process
        sample_param['layer'] += 1
        sample_param['neighbor'].append(1)
        degree_encoding = Global_TimeEncode(gnn_param['dim_out']).cuda()

        combine_first = False
        if 'combine_neighs' in train_param and train_param['combine_neighs']:
            combine_first = True
        ID_model_mean = ID_Model_full_mean(gnn_dim_node).cuda()
        criterion = torch.nn.CrossEntropyLoss()

        if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
            if node_feats is not None:
                node_feats = node_feats.cuda()
            if edge_feats is not None:
                edge_feats = edge_feats.cuda()

        sampler = None
        if not ('no_sample' in sample_param and sample_param['no_sample']):
            sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                    sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                    sample_param['strategy']=='recent', sample_param['prop_time'],
                                    sample_param['history'], float(sample_param['duration']))
            


        global feature_processed_edge_id
        feature_processed_edge_id = len(train_df)

        #global unseen_nodes_degree
        unseen_nodes_degree = torch.zeros(node_count).cuda()
        
        # positional node feature mixing process
        def forward_feature_to(time, node_feats=node_feats, rand_feats=rand_feats, train_node_index=train_node_index, unseen_nodes_degree=unseen_nodes_degree):
            global feature_processed_edge_id
            if feature_processed_edge_id >= len(df):
                return node_feats, rand_feats, unseen_nodes_degree
            while df.time[feature_processed_edge_id] < time:
                rows = df[feature_processed_edge_id:min(feature_processed_edge_id + args.batch_size, len(df))]
                feature_processed_edge_id += args.batch_size
                src_unseen_ind = (~np.in1d(rows.src.values, train_node_index)).astype(np.int32)
                dst_unseen_ind = (~np.in1d(rows.dst.values, train_node_index)).astype(np.int32)
                unseen_node_id = np.union1d(rows.src.values[src_unseen_ind==1], rows.dst.values[dst_unseen_ind==1])

                new_unseen_nodes_degree = torch.zeros(node_count).cuda()
                new_unseen_nodes_degree[rows.src.values[src_unseen_ind==1]] += 1
                new_unseen_nodes_degree[rows.dst.values[dst_unseen_ind==1]] += 1
                node_feats[unseen_node_id] = node_feats[unseen_node_id]*(unseen_nodes_degree[unseen_node_id]/(unseen_nodes_degree[unseen_node_id]+new_unseen_nodes_degree[unseen_node_id])).unsqueeze(-1)
                node_feats[rows.src.values[src_unseen_ind==1]] += node_feats[rows.dst.values[src_unseen_ind==1]]/(new_unseen_nodes_degree[rows.src.values[src_unseen_ind==1]]+unseen_nodes_degree[rows.src.values[src_unseen_ind==1]]).unsqueeze(-1)
                node_feats[rows.dst.values[dst_unseen_ind==1]] += node_feats[rows.src.values[dst_unseen_ind==1]]/(new_unseen_nodes_degree[rows.dst.values[dst_unseen_ind==1]]+unseen_nodes_degree[rows.dst.values[dst_unseen_ind==1]]).unsqueeze(-1)
                rand_feats[unseen_node_id] = rand_feats[unseen_node_id]*(unseen_nodes_degree[unseen_node_id]/(unseen_nodes_degree[unseen_node_id]+new_unseen_nodes_degree[unseen_node_id])).unsqueeze(-1)
                rand_feats[rows.src.values[src_unseen_ind==1]] += rand_feats[rows.dst.values[src_unseen_ind==1]]/(new_unseen_nodes_degree[rows.src.values[src_unseen_ind==1]]+unseen_nodes_degree[rows.src.values[src_unseen_ind==1]]).unsqueeze(-1)
                rand_feats[rows.dst.values[dst_unseen_ind==1]] += rand_feats[rows.src.values[dst_unseen_ind==1]]/(new_unseen_nodes_degree[rows.dst.values[dst_unseen_ind==1]]+unseen_nodes_degree[rows.dst.values[dst_unseen_ind==1]]).unsqueeze(-1)
                unseen_nodes_degree[rows.src.values[src_unseen_ind==1]] += 1
                unseen_nodes_degree[rows.dst.values[dst_unseen_ind==1]] += 1

                if feature_processed_edge_id >= len(df):
                    break
            return node_feats, rand_feats, unseen_nodes_degree


   
        def get_node_ID_emb(root_nodes, ts, rand_feats=rand_feats): # 
            if sampler is not None:
                sampler.sample(root_nodes, ts)
                
                ret = sampler.get_ret()
                
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            for k in range(sample_param['history']):
                mfgs[1][k].srcdata['degree'][mfgs[1][k].num_dst_nodes():] = mfgs[0][k].srcdata['degree'][:mfgs[0][k].num_dst_nodes()-mfgs[1][k].num_dst_nodes()]
                if 'h' in mfgs[0][k].srcdata:
                    mfgs[1][k].srcdata['h'] = mfgs[0][k].srcdata['h'][:mfgs[0][k].num_dst_nodes()]
            mfgs = mfgs[1:]
            
            for k in range(sample_param['history']):
                # channel -> [random-based augmentation, transductive-based augmentation, structural-based augmentation]
                mfgs[0][k].srcdata['h'] = torch.cat([rand_feats[mfgs[0][0].srcdata['ID'].long()], mfgs[0][k].srcdata['h'], degree_encoding(mfgs[0][k].srcdata['degree'])], dim=1)
            ID_rep_mean = ID_model_mean(mfgs[0][0])
            return ID_rep_mean


        train_emb_mean = []
        val_emb_mean = []
        train_val_label = []
        train_label = []
        val_label = []
        
        
        feature_processed_edge_id = len(train_df)
        if node_feats is not None:
            node_feats = fixed_node_feats.clone().cuda()
            rand_feats = fixed_rand_feats.clone().cuda()
        unseen_nodes_degree = torch.zeros(node_count).cuda()
        if sampler is not None:
            sampler.reset()
        
        #### add training set before the split time ####
        for _, rows in tqdm(train_ldf.groupby(train_ldf.index // args.batch_size)):
            ID_emb_mean = get_node_ID_emb(rows.node.values.astype(np.int32), rows.time.values.astype(np.float32))
            for k in range(len(rows.label.values)):
                train_emb_mean.append(ID_emb_mean[k])
                train_label.append(rows.label.values.astype(np.int64)[k])
                train_val_label.append(rows.label.values.astype(np.int64)[k])
                
        #### add validation set after the split time ####
        for _, rows in val_ldf.groupby(val_ldf.index // args.batch_size):
            node_feats, rand_feats, unseen_nodes_degree = forward_feature_to(rows.time.values[0], node_feats, rand_feats, unseen_nodes_degree=unseen_nodes_degree)
            ID_emb_mean = get_node_ID_emb(rows.node.values.astype(np.int32), rows.time.values.astype(np.float32))
            for k in range(len(rows.label.values)):
                val_emb_mean.append(ID_emb_mean[k])
                val_label.append(rows.label.values.astype(np.int64)[k])
                train_val_label.append(rows.label.values.astype(np.int64)[k])



        train_emb_mean = torch.stack(train_emb_mean, dim=0).cuda()
        val_emb_mean = torch.stack(val_emb_mean, dim=0).cuda()
        train_label = torch.tensor(train_label).cuda()
        val_label = torch.tensor(val_label).cuda()


        train_emb_mean_rand = torch.cat([train_emb_mean[:, :gnn_dim_node], train_emb_mean[:, gnn_dim_node*3:gnn_dim_node*4]], dim=1)
        train_emb_mean_pos = torch.cat([train_emb_mean[:, gnn_dim_node:gnn_dim_node*2], train_emb_mean[:, gnn_dim_node*4:gnn_dim_node*5]], dim=1)
        train_emb_mean_str = torch.cat([train_emb_mean[:, gnn_dim_node*2:gnn_dim_node*3], train_emb_mean[:, gnn_dim_node*5:gnn_dim_node*6]], dim=1)
        
        val_emb_mean_rand = torch.cat([val_emb_mean[:, :gnn_dim_node], val_emb_mean[:, gnn_dim_node*3:gnn_dim_node*4]], dim=1)
        val_emb_mean_pos = torch.cat([val_emb_mean[:, gnn_dim_node:gnn_dim_node*2], val_emb_mean[:, gnn_dim_node*4:gnn_dim_node*5]], dim=1)
        val_emb_mean_str = torch.cat([val_emb_mean[:, gnn_dim_node*2:gnn_dim_node*3], val_emb_mean[:, gnn_dim_node*5:gnn_dim_node*6]], dim=1)


        single_mean_rand_model = single_Linear(gnn_dim_node*2, class_num).cuda()
        single_mean_pos_model = single_Linear(gnn_dim_node*2, class_num).cuda()
        single_mean_str_model = single_Linear(gnn_dim_node*2, class_num).cuda()

        
        single_mean_rand_optimizer = torch.optim.Adam(single_mean_rand_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        single_mean_pos_optimizer = torch.optim.Adam(single_mean_pos_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        single_mean_str_optimizer = torch.optim.Adam(single_mean_str_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        iter = 100
        batch_size = args.batch_size

        for i in range(iter):

            for j in range(0, len(train_emb_mean_rand)//batch_size):
                single_mean_rand_model.train()
                single_mean_pos_model.train()
                single_mean_str_model.train()
                single_mean_rand_optimizer.zero_grad()
                single_mean_pos_optimizer.zero_grad()
                single_mean_str_optimizer.zero_grad()
                rand_mean_predict = single_mean_rand_model(train_emb_mean_rand[j*batch_size:(j+1)*batch_size])
                pos_mean_predict = single_mean_pos_model(train_emb_mean_pos[j*batch_size:(j+1)*batch_size])
                str_mean_predict = single_mean_str_model(train_emb_mean_str[j*batch_size:(j+1)*batch_size])
                rand_mean_loss = criterion(rand_mean_predict, train_label[j*batch_size:(j+1)*batch_size])
                pos_mean_loss = criterion(pos_mean_predict, train_label[j*batch_size:(j+1)*batch_size])
                str_mean_loss = criterion(str_mean_predict, train_label[j*batch_size:(j+1)*batch_size])

                rand_mean_loss.backward()
                pos_mean_loss.backward()
                str_mean_loss.backward()
                single_mean_rand_optimizer.step()
                single_mean_pos_optimizer.step()
                single_mean_str_optimizer.step()
            if i % 10 == 0:
                single_mean_rand_model.eval()
                single_mean_pos_model.eval()
                single_mean_str_model.eval()

                val_rand_mean_pred = single_mean_rand_model(val_emb_mean_rand)
                val_pos_mean_pred = single_mean_pos_model(val_emb_mean_pos)
                val_str_mean_pred = single_mean_str_model(val_emb_mean_str)

                val_rand_mean_loss = criterion(val_rand_mean_pred, val_label)
                val_pos_mean_loss = criterion(val_pos_mean_pred, val_label)
                val_str_mean_loss = criterion(val_str_mean_pred, val_label)

                    
                print('val rand mean loss: {:.4f}, val pos mean loss: {:.4f}, val str mean loss: {:.4f}'.format(val_rand_mean_loss, val_pos_mean_loss, val_str_mean_loss))
        
        total_mean_rand_val_loss.append(val_rand_mean_loss.cpu().detach())
        total_mean_pos_val_loss.append(val_pos_mean_loss.cpu().detach())
        total_mean_str_val_loss.append(val_str_mean_loss.cpu().detach())
    
    total_mean_rand_val_loss = np.array(total_mean_rand_val_loss)
    total_mean_pos_val_loss = np.array(total_mean_pos_val_loss)
    total_mean_str_val_loss = np.array(total_mean_str_val_loss)


    print('mean rand val loss: {:.4f}, mean pos val loss: {:.4f}, mean str val loss: {:.4f}'.format(total_mean_rand_val_loss.mean(), total_mean_pos_val_loss.mean(), total_mean_str_val_loss.mean()))
    final_loss_results = [total_mean_rand_val_loss.mean(), total_mean_pos_val_loss.mean(), total_mean_str_val_loss.mean()]
    logger.info('mean rand val loss: {:.4f}, mean pos val loss: {:.4f}, mean str val loss: {:.4f}'.format(total_mean_rand_val_loss.mean(), total_mean_pos_val_loss.mean(), total_mean_str_val_loss.mean()))
    top1_ind = np.argmin(final_loss_results)
    selected_feat = ''
    if top1_ind == 0:
        selected_feat = 'random_mix'
        logger.info('select random_mix')
    elif top1_ind == 1:
        selected_feat = 'node2vec_mix'
        logger.info('select transductive_mix')
    elif top1_ind == 2:
        selected_feat = 'structural'
        logger.info('select degree_encoding')

    save_model_folder = f"./selected_features/"
    os.makedirs(save_model_folder, exist_ok=True)
    selected_feature_config = {
        'selected_feat': selected_feat
    }
    with open(f"{save_model_folder}/{args.data}.json", 'w') as f:
        json.dump(selected_feature_config, f)

        
    return selected_feat, final_loss_results



if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='dataset name')
    parser.add_argument('--config', type=str, default='', help='path to config file')
    parser.add_argument('--batch_size', type=int, default=600)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--n_degree', default=100)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--split_num', type=int, default=5)
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
    parser.add_argument('--set_degree', default=False, action='store_true', help='set maximum degree')
    parser.add_argument('--task', type=str, default='class',choices=['class', 'anomaly', 'tgbn', 'AP'], help='name of stored model to load')

    parser.add_argument('--train_ratio', type=float, default=0.1, help='train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation ratio')


    args=parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True
    fh = logging.FileHandler('log/feature_selection_{}train_ratio{}.log'.format(args.data, args.train_ratio))
    #fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    #ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)   
    
    logger.info(args)
    logger.info('Data Processing')
    total_run_results = []
    for i in range(args.n_runs):
        result, final_loss_results = feature_select(args, logger)
        total_run_results.append(final_loss_results)
    total_run_results = np.array(total_run_results)
    logger.info(f'total run mean results:{total_run_results.mean(axis=0)}')
    logger.info(f'total run std results:{total_run_results.std(axis=0)}')



