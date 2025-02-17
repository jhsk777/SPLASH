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
from DATA.tgb.nodeproppred.evaluate import Evaluator
from node2vec import Node2Vec
import timeit

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#processed_edge_id = 0
def objective(args, logger):

    logger.info('Data Processing')
    device = torch.device(f'cuda:{args.gpu}')
    if args.data in ['WIKI', 'REDDIT', 'MOOC']:
        args.task = 'anomaly'
    elif args.data in ['Email-EU', 'GDELT_sample', 'synthetic-70', 'synthetic-90']:
        args.task = 'class'
    else:
        raise ValueError('Task not supported')

    TASK = args.task

    HIDDEN_DIM = args.hidden_dim
    ldf = pd.read_csv('DATA/{}/labels.csv'.format(args.data))
    auc_mean = []
    training_mean = []
    val_mean = []

    best_split_list = []
    for i in range(args.n_runs):
        ldf = pd.read_csv('DATA/{}/labels.csv'.format(args.data))
        role = ldf['ext_roll'].values
        labels = ldf['label'].values.astype(np.int64)
        class_num = labels.max() + 1
        train_edge_end = int(len(role) * args.train_ratio)
        val_edge_end = int(len(role) * args.val_ratio)
        test_start = val_edge_end

        train_end_time = ldf.time[train_edge_end]
        val_end_time = ldf.time[val_edge_end]

        train_ldf = ldf[ldf.time < train_end_time]
        val_ldf = ldf[(ldf.time >= train_end_time) & (ldf.time < val_end_time)]
        test_ldf = ldf[ldf.time >= val_end_time]
        print('train:', len(train_ldf), 'val:', len(val_ldf), 'test:', len(test_ldf))  
        node_feats, edge_feats = load_feat(args.data)
        

        g, df = load_graph(args.data)
        train_df = df[df.time < train_end_time]

        node_count = int(max(df.src.max(), df.dst.max())) + 1
        train_node_max = int(max(train_df.src.max(), train_df.dst.max()))
        train_node_index = np.unique(np.concatenate([train_df.src.values, train_df.dst.values]))
        total_node_index = np.unique(np.concatenate([df.src.values, df.dst.values]))
        new_node_index = np.setdiff1d(total_node_index, train_node_index) 
        train_time = train_end_time - df.time.min()

        ### feature augmentation according to the selected process 
        if args.selected_feature == 'empty':
            pass
        if args.selected_feature == 'random':
            node_feats = torch.randn(node_count, HIDDEN_DIM)
        elif args.selected_feature == 'zero':
            node_feats = torch.zeros(node_count, HIDDEN_DIM)
        elif args.selected_feature == 'random_mix':
            node_feats = torch.randn(node_count, HIDDEN_DIM)
            node_feats[new_node_index] = 0

        elif args.selected_feature == 'node2vec_mix':
            node_feats = torch.zeros(node_count, HIDDEN_DIM)
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
        if node_feats is not None:
            fixed_node_feats = node_feats.clone()

        sample_param, memory_param, gnn_param, train_param = parse_config(args.config)


        if args.set_degree:
            if type(args.n_degree) == int:
                sample_param['neighbor'] = [args.n_degree]
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
        if (args.selected_feature == 'structural') and gnn_param['arch'] != 'identity':
            gnn_dim_node = gnn_param['dim_out']
            sample_param['layer'] += 1
            sample_param['neighbor'].append(1)
        elif (args.selected_feature == 'structural') and ('no_sample' in sample_param):
            sample_param = {'layer':1, 'neighbor':[1], 'strategy':'recent', 'prop_time':False, 'history':1, 'duration':0, 'num_thread':1}
        combine_first = False
        if 'combine_neighs' in train_param and train_param['combine_neighs']:
            combine_first = True
        gnn_param['skip_alpha'] = args.skip_alpha
        model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first, device=device).cuda()
        total_params = sum(p.numel() for p in model.parameters())
        print("Total parameters:", total_params)       
        mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
        classifier = NodeClassificationModel(gnn_param['dim_out'], gnn_param['dim_out'], class_num).cuda()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(set(model.parameters())|set(classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        if args.selected_feature == 'structural':
            degree_encoding = Global_TimeEncode(gnn_param['dim_out']).cuda()
        
        
        if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
            if node_feats is not None:
                node_feats = node_feats.cuda()
            if edge_feats is not None:
                edge_feats = edge_feats.cuda()
            if mailbox is not None:
                mailbox.move_to_gpu(device=device)

        sampler = None
        if not ('no_sample' in sample_param and sample_param['no_sample']):
            sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                    sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                    sample_param['strategy']=='recent', sample_param['prop_time'],
                                    sample_param['history'], float(sample_param['duration']))
            

       
        neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

        global feature_processed_edge_id
        feature_processed_edge_id = len(train_df)

        unseen_nodes_degree = torch.zeros(node_count).cuda()
        
        def forward_feature_to(time, node_feats=node_feats, train_node_index=train_node_index, unseen_nodes_degree=unseen_nodes_degree):
            global feature_processed_edge_id
            #global negative_node_set
            if feature_processed_edge_id >= len(df):
                return node_feats, unseen_nodes_degree
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
                unseen_nodes_degree[rows.src.values[src_unseen_ind==1]] += 1
                unseen_nodes_degree[rows.dst.values[dst_unseen_ind==1]] += 1

                if feature_processed_edge_id >= len(df):
                    break
            return node_feats, unseen_nodes_degree


        global processed_edge_id
        processed_edge_id = 0
        def forward_model_to(time):
            global processed_edge_id
            if processed_edge_id >= len(df):
                return
            while df.time[processed_edge_id] < time:
                rows = df[processed_edge_id:min(processed_edge_id + args.batch_size, len(df))]
                if processed_edge_id < train_edge_end:
                    model.train()
                else:
                    model.eval()
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
                ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
                if sampler is not None:
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        pos_root_end = root_nodes.shape[0] * 2 // 3
                        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    else:
                        sampler.sample(root_nodes, ts)
                    ret = sampler.get_ret()  
                if gnn_param['arch'] != 'identity':
                    mfgs = to_dgl_blocks(ret, sample_param['history'], device=device)
                    
                else:
                    mfgs = node_to_dgl_blocks(root_nodes, ts, device=device)
                mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first, device=device)
                if (args.selected_feature == 'structural') and gnn_param['arch'] == 'identity':
                    f_mfgs = to_dgl_blocks(ret, sample_param['history'], device=device)
                    f_mfgs = prepare_input(f_mfgs, node_feats, edge_feats, combine_first=combine_first, device=device)
                    mfgs[0][0].srcdata['h'] = degree_encoding(f_mfgs[-1][0].srcdata['degree'][:f_mfgs[-1][0].num_dst_nodes()])
                elif args.selected_feature == 'structural':
                    mfgs[1][0].srcdata['degree'][mfgs[1][0].num_dst_nodes():] = mfgs[0][0].srcdata['degree'][mfgs[1][0].num_dst_nodes():mfgs[0][0].num_dst_nodes()]
                    mfgs = mfgs[1:]
                    mfgs[0][0].srcdata['h'] = degree_encoding(mfgs[0][0].srcdata['degree'])
                if mailbox is not None:
                    mailbox.prep_input_mails(mfgs[0])
                    
                with torch.no_grad():
                    pred_pos, pred_neg = model(mfgs)
                    
                    if mailbox is not None:
                        eid = rows['Unnamed: 0'].values
                        mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                        block = None
                        if memory_param['deliver_to'] == 'neighbors':
                            block = to_dgl_blocks(ret, sample_param['history'], reverse=True, device=device)[0][0]
                        mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                        mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
                processed_edge_id += args.batch_size
                if processed_edge_id >= len(df):
                    return

        def get_node_emb(root_nodes, ts):
            if memory_param['type'] != 'none':
                forward_model_to(ts[0])
            if sampler is not None:
                sampler.sample(root_nodes, ts)
                
                ret = sampler.get_ret()
                
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'], device=device)
                
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts, device=device)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first, device=device)
            if (args.selected_feature == 'structural') and (gnn_param['arch'] == 'identity'):
                f_mfgs = to_dgl_blocks(ret, sample_param['history'], device=device)
                f_mfgs = prepare_input(f_mfgs, node_feats, edge_feats, combine_first=combine_first, device=device)
                mfgs[0][0].srcdata['h'] = degree_encoding(f_mfgs[-1][0].srcdata['degree'][:f_mfgs[-1][0].num_dst_nodes()])
            elif args.selected_feature == 'structural':
                for k in range(sample_param['history']):
                    mfgs[1][k].srcdata['degree'][mfgs[1][k].num_dst_nodes():] = mfgs[0][k].srcdata['degree'][:mfgs[0][k].num_dst_nodes()-mfgs[1][k].num_dst_nodes()]
                mfgs = mfgs[1:]
                for k in range(sample_param['history']):
                    mfgs[0][k].srcdata['h'] = degree_encoding(mfgs[0][k].srcdata['degree'])
            
            if mailbox is not None:
                mailbox.move_to_gpu()
                mailbox.prep_input_mails(mfgs[0])
            ret = model.get_emb(mfgs)

            return ret


        ### joint train start
        best_val_auc = 0
        best_test_auc = 0
        train_auc_list, val_auc_list, test_auc_list = [], [], []
        train_loss_list, val_loss_list, test_loss_list = [], [], []
        early_stop_count = 0
        training_time_list = []
        test_time_list = []
        
        for e in range(args.epoch):
            start_time = timeit.default_timer()
            emb_list = []
            processed_edge_id = 0
            feature_processed_edge_id = len(train_df)
            if node_feats is not None:
                node_feats = fixed_node_feats.clone().cuda()
            unseen_nodes_degree = torch.zeros(node_count).cuda()
            
            if sampler is not None:
                sampler.reset()
            if mailbox is not None:
                mailbox.reset()
                model.memory_updater.last_updated_nid = None            
            model.train()
            classifier.train()
            train_ground_label = train_ldf.label.values.astype(np.float32)
            pred_label = []
            total_loss = 0
            total_train_ele_num = len(train_ldf)
            min_index = train_ldf.index.min()
            for _, rows in tqdm(train_ldf.groupby(train_ldf.index // args.batch_size)):
                current_label = torch.from_numpy(rows.label.values.astype(np.int64)).cuda()
                optimizer.zero_grad()
                current_emb = get_node_emb(rows.node.values.astype(np.int32), rows.time.values.astype(np.float32))

                emb_list.append(current_emb.detach().cpu())
                pred = classifier(current_emb)
                loss = criterion(pred, current_label.long())
                total_loss += loss.item() / total_train_ele_num
                loss.backward()
                optimizer.step()
                if TASK == 'anomaly':
                    pred_label.append((pred.detach()[:, 1] - pred.detach()[:, 0]).cpu().reshape(-1))
                elif TASK == 'class':
                    pred_label.append(torch.argmax(pred.softmax(dim=1), dim=1).detach().cpu().reshape(-1))
   
            if TASK == 'anomaly':
                train_auc = roc_auc_score(train_ground_label, torch.cat(pred_label, dim=0))
                print('Epoch: {}\tTrain auc: {:.4f}\tTraining loss: {:.4f}'.format(e, train_auc, total_loss))
            elif TASK == 'class':
                train_auc = f1_score(train_ground_label, torch.cat(pred_label, dim=0).cpu(), average="micro")
                print('Epoch: {}\tTrain f1: {:.4f}\tTraining loss: {:.4f}'.format(e, train_auc, total_loss))          
            train_auc_list.append(train_auc)
            train_loss_list.append(total_loss)
            
            training_time = timeit.default_timer() - start_time
            training_time_list.append(training_time)
            print("Training takes--- %s seconds ---" % (training_time))
            

            model.eval()
            classifier.eval()
            criterion = torch.nn.CrossEntropyLoss()
            total_val_loss = 0
            total_val_ele_num = len(val_ldf)    
            with torch.no_grad():            
                val_ground_label = val_ldf.label.values.astype(np.float32)
                pred_label = []
                min_index = val_ldf.index.min()
                for _, rows in val_ldf.groupby(val_ldf.index // args.batch_size):
                    if args.selected_feature in ['random_mix', 'node2vec_mix']:
                        node_feats, unseen_nodes_degree = forward_feature_to(rows.time.values[0], node_feats, unseen_nodes_degree=unseen_nodes_degree)
                    current_label = torch.from_numpy(rows.label.values.astype(np.int64)).cuda()
                    current_emb = get_node_emb(rows.node.values.astype(np.int32), rows.time.values.astype(np.float32))
                    pred = classifier(current_emb)
                    loss = criterion(pred, current_label.long())  
                    total_val_loss += loss.item() / total_val_ele_num                 
                    if TASK == 'anomaly':
                        pred_label.append((pred.detach()[:, 1] - pred.detach()[:, 0]).cpu().reshape(-1))
                    elif TASK == 'class':
                        pred_label.append(torch.argmax(pred.softmax(dim=1), dim=1).detach().cpu().reshape(-1))

            
            if TASK =='anomaly':
                val_auc = roc_auc_score(val_ground_label, torch.cat(pred_label, dim=0))
                val_auc_list.append(val_auc)
                print('Epoch: {}\tVal auc: {:.4f}'.format(e, val_auc))
            elif TASK == 'class':
                val_auc = f1_score(val_ground_label, torch.cat(pred_label, dim=0).cpu(), average="micro")
                val_auc_list.append(val_auc)
                print('Epoch: {}\tVal f1: {:.4f}'.format(e, val_auc))

            val_loss_list.append(total_val_loss)
            #### test start
            start_time = timeit.default_timer()
            with torch.no_grad():
                total_test_loss = 0
                total_test_ele_num = len(test_ldf)
                test_ground_label = test_ldf.label.values.astype(np.float32)
                pred_label = []
                for _, rows in test_ldf.groupby(test_ldf.index // args.batch_size):
                    if args.selected_feature in ['random_mix', 'node2vec_mix']:
                        node_feats, unseen_nodes_degree = forward_feature_to(rows.time.values[0], node_feats, unseen_nodes_degree=unseen_nodes_degree)
                    current_label = torch.from_numpy(rows.label.values.astype(np.int64)).cuda()
                    current_emb = get_node_emb(rows.node.values.astype(np.int32), rows.time.values.astype(np.float32))
                    pred = classifier(current_emb)
                    loss = criterion(pred, current_label.long())
                    total_test_loss += loss.item() / total_test_ele_num
                    if TASK == 'anomaly':
                        pred_label.append((pred.detach()[:, 1] - pred.detach()[:, 0]).cpu().reshape(-1))
                    elif TASK == 'class':
                        pred_label.append(torch.argmax(pred.softmax(dim=1), dim=1).detach().cpu().reshape(-1))
            if TASK == 'anomaly':
                test_auc = roc_auc_score(test_ground_label, torch.cat(pred_label, dim=0))
                test_auc_list.append(test_auc)
                print('Epoch: {}\tTest auc: {:.4f}'.format(e, test_auc))
            elif TASK == 'class':
                test_auc = f1_score(test_ground_label, torch.cat(pred_label, dim=0).cpu(), average="micro")
                test_auc_list.append(test_auc)
                print('Epoch: {}\tTest f1: {:.4f}'.format(e, test_auc))
            test_loss_list.append(total_test_loss)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= args.early_stop:
                    break
            
            test_time = timeit.default_timer() - start_time
            print("Testing takes--- %s seconds ---" % (test_time))
            test_time_list.append(test_time)
        auc_mean.append(best_test_auc)
        training_mean.append(train_auc)
        val_mean.append(best_val_auc)
        print('training time:', np.array(training_time_list).mean(), np.array(training_time_list).std())
        print('test time:', np.array(test_time_list).mean(), np.array(test_time_list).std())

    best_mean = np.array(auc_mean)
    best_val_mean = np.array(val_mean)
    logger.info('Final Validation auc: {:.4f}, std: {:.4f}'.format(np.mean(best_val_mean), np.std(best_val_mean)))
    logger.info('Final Testing auc: {:.4f}, std: {:.4f}'.format(np.mean(best_mean), np.std(best_mean)))
    training_mean = np.array(training_mean)

    return np.mean(best_mean)



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
    parser.add_argument('--skip_alpha', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
    parser.add_argument('--task', type=str, default='class',choices=['class', 'anomaly'], help='name of stored model to load')

    parser.add_argument('--train_ratio', type=float, default=0.1, help='train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation ratio')

    parser.add_argument('--set_degree', default=False, action='store_true', help='set maximum degree')
    parser.add_argument('--selected_feature', type=str, default='empty', choices=['empty', 'zero', 'random', 'random_mix', 'node2vec_mix', 'structural'])

    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True
    if not os.path.exists(f'..log/{args.train_ratio}'):
        os.makedirs(f'..log/{args.train_ratio}')
    fh = logging.FileHandler('..log/joint_{}_{}_train_ratio.log'.format(args.data, args.config[7:-4],args.train_ratio))
    #fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    #ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)   
    
    logger.info(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.set_device(int(args.gpu))
    set_seed(args.seed)
    result = objective(args, logger)


