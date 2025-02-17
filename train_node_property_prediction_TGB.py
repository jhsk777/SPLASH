import argparse
import os
import hashlib
from functools import partial
import torch
import timeit
import random
import dgl
import numpy as np
import pandas as pd
import pdb
from modules.modules import *
from utils.sampler import *
from utils.utils import *
from tqdm import tqdm
import logging
from DATA.tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from DATA.tgb.nodeproppred.evaluate import Evaluator
from torch_geometric.loader import TemporalDataLoader
import sys


def objective(args, logger):
	## data processing part
	dataset = PyGNodePropPredDataset(name=args.data, root="datasets")
	device = torch.device(f'cuda:{args.gpu}')
	data = dataset.get_TemporalData()
	max_node_id = max(data.src.max(), data.dst.max())
	train_data, val_data, test_data = data.train_val_test_split(val_ratio=args.val_ratio-args.train_ratio, test_ratio=1.0-args.val_ratio)
	HIDDEN_DIM = args.dim


	eval_metric = dataset.eval_metric
	num_classes = dataset.num_classes
	node_count = int(max(train_data.src.max(), train_data.dst.max(), val_data.src.max(), val_data.dst.max(), test_data.src.max(), test_data.dst.max())) + 1
	train_node_max = int(max(train_data.src.max(), train_data.dst.max()))
	train_node_index = np.unique(np.concatenate([train_data.src, train_data.dst]))
	total_node_index = np.unique(np.concatenate([train_data.src, train_data.dst, val_data.src, val_data.dst, test_data.src, test_data.dst]))
	new_node_index = np.setdiff1d(total_node_index, train_node_index) 
	train_end_time = train_data.t.max()
	final_result_list = []
	final_val_results_list = []
	best_split_list = []
	for _ in range(args.n_runs):
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
			src_np = train_data.src.numpy()
			dst_np = train_data.dst.numpy()
			edge_weight_np = train_data.msg.squeeze().numpy()
			for i in range(len(train_data.src)):
				if train_G.has_edge(src_np[i].astype(np.int32), dst_np[i].astype(np.int32)):
					train_G[src_np[i].astype(np.int32)][dst_np[i].astype(np.int32)]['weight'] += edge_weight_np[i]
				else:
					train_G.add_edge(src_np[i].astype(np.int32), dst_np[i].astype(np.int32), weight=edge_weight_np[i])
			node2vec = Node2Vec(train_G, dimensions=HIDDEN_DIM, walk_length=10, num_walks=80, workers=1, weight_key='weight', p=10)
			node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
			for node in train_G.nodes():
				node_feats[node] = torch.tensor(node2vec_model.wv[str(node)])          
		else:
			node_feats, edge_feats = None, dataset.edge_feat
		edge_feats = dataset.edge_feat
		edge_weight = torch.cat([train_data.msg, val_data.msg, test_data.msg], dim=0)
		if node_feats is not None:
			fixed_node_feats = node_feats.clone()
		g, df = load_graph(args.data)
		sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

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

		
		train_loader = TemporalDataLoader(train_data, batch_size=train_param['batch_size'], shuffle=False)
		val_loader = TemporalDataLoader(val_data, batch_size=train_param['batch_size'])
		test_loader = TemporalDataLoader(test_data, batch_size=train_param['batch_size'])

		if args.data == 'tgbn-trade-filter' or args.data == 'tgbn-trade_sort':
			evaluator = Evaluator(name='tgbn-trade')
		else:
			evaluator = Evaluator(name=args.data)

		gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
		gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

		if args.selected_feature == 'structural' and gnn_param['arch'] != 'identity':
			gnn_dim_node = gnn_param['dim_out']
			sample_param['layer'] += 1
			sample_param['neighbor'].append(1)
		elif args.selected_feature == 'structural' and ('no_sample' in sample_param):
			sample_param = {'layer':1, 'neighbor':[1], 'strategy':'recent', 'prop_time':False, 'history':1, 'duration':0, 'num_thread':1}
		combine_first = False
		if 'combine_neighs' in train_param and train_param['combine_neighs']:
			combine_first = True
		gnn_param['skip_alpha'] = args.skip_alpha
		model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first, device=device).cuda()
		# print model paramter number
		logger.info('Model Parameter Number: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
		mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None	
		node_pred = NodeClassificationModel(gnn_param['dim_out'], args.dim, num_classes).cuda()


		#model.load_state_dict(torch.load(args.model))

		optimizer = torch.optim.Adam(set(model.parameters()) | set(node_pred.parameters()), lr=args.lr, weight_decay=args.weight_decay)
		if args.selected_feature == 'structural':
			degree_encoding = Global_TimeEncode(gnn_param['dim_out']).cuda()
		criterion = torch.nn.CrossEntropyLoss()

		if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
			if node_feats is not None:
				node_feats = node_feats.cuda()
			if edge_feats is not None:
				edge_feats = edge_feats.cuda()
			if edge_weight is not None:
				edge_weight = edge_weight.cuda()
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
		feature_processed_edge_id = len(train_data.src)
		unseen_nodes_degree = torch.zeros(node_count).cuda()

		def forward_feature_to(time, node_feats=node_feats, train_node_index=train_node_index, unseen_nodes_degree=unseen_nodes_degree):
			global feature_processed_edge_id
			if feature_processed_edge_id >= len(df):
				return node_feats
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

		#model.load_state_dict(torch.load(args.model))
		global processed_edge_id
		processed_edge_id = 0
		
		def forward_model_to(time):
			global processed_edge_id
			if processed_edge_id >= len(df):
				return
			while df.time[processed_edge_id] < time:
				rows = df[processed_edge_id:min(processed_edge_id + train_param['batch_size'], len(df))]
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

					
				mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first, device=device, edge_weight=edge_weight)
				if args.selected_feature == 'structural' and gnn_param['arch'] == 'identity':
					f_mfgs = to_dgl_blocks(ret, sample_param['history'], device=device)
					f_mfgs = prepare_input(f_mfgs, node_feats, edge_feats, combine_first=combine_first, device=device, edge_weight=edge_weight)
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
				processed_edge_id += train_param['batch_size']
				if processed_edge_id >= len(df):
					return

		def get_node_emb(root_nodes, ts, query_t):
			if memory_param['type'] != 'none':
				forward_model_to(query_t)
			#forward_model_to(ts[0])
			if sampler is not None:
				sampler.sample(root_nodes, ts)
				
				ret = sampler.get_ret()
				
			if gnn_param['arch'] != 'identity':
				mfgs = to_dgl_blocks(ret, sample_param['history'], device=device)
				
			else:
				mfgs = node_to_dgl_blocks(root_nodes, ts, device=device)
				
			mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first, device=device, edge_weight=edge_weight)
			if args.selected_feature == 'structural' and (gnn_param['arch'] == 'identity'):
				f_mfgs = to_dgl_blocks(ret, sample_param['history'], device=device)
				f_mfgs = prepare_input(f_mfgs, node_feats, edge_feats, combine_first=combine_first, device=device, edge_weight=edge_weight)
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

		def train():
			model.train()
			node_pred.train()
			if sampler is not None:
				sampler.reset()
			if mailbox is not None:
				mailbox.reset()
				model.memory_updater.last_updated_nid = None
			

			total_loss = 0
			
			label_t = dataset.get_label_time()
			num_label_ts = 0
			total_score = 0
			

			for batch in train_loader:
				
				optimizer.zero_grad()
				src, dst, t = batch.src, batch.dst, batch.t
				
				query_t = batch.t[-1]
				if query_t > label_t:
					#pdb.set_trace()
					# find the node labels from the past day
					label_tuple = dataset.get_node_label(query_t)
					label_ts, label_srcs, labels = (
						label_tuple[0],
						label_tuple[1],
						label_tuple[2],
					)
					#pdb.set_trace()
					label_t = dataset.get_label_time()
					n_id, n_ts = label_srcs.numpy(), label_ts.numpy()
					qeury_t_repeat = np.repeat(query_t.numpy(), len(n_id))
					output_emb = get_node_emb(n_id.astype(np.int32), n_ts.astype(np.float32), label_ts[0])
					output_pred = node_pred(output_emb)
					#pdb.set_trace()
					loss = criterion(output_pred, labels.cuda())
					np_pred = output_pred.softmax(dim=1).cpu().detach().numpy()
					np_true = labels.cpu().detach().numpy()
					input_dict = {
						"y_true": np_true,
						"y_pred": np_pred,
						"eval_metric": [eval_metric],
					}
					result_dict = evaluator.eval(input_dict)
					score = result_dict[eval_metric]
					total_score += score
					num_label_ts += 1
					loss.backward()
					optimizer.step()
					total_loss += float(loss)
					
			metric_dict = {
				"ce": total_loss / num_label_ts,
			}
			metric_dict[eval_metric] = total_score / num_label_ts
			return metric_dict


		@torch.no_grad()
		def test(loader, node_feats, unseen_nodes_degree):
			model.eval()
			node_pred.eval()
			total_score = 0
			label_t = dataset.get_label_time()
			# if args.data == 'tgbn-trade':
			# 	label_t *= 3600 * 24 * 365
			num_label_ts = 0

			for batch in loader:
				src, dst, t = batch.src, batch.dst, batch.t
				query_t = batch.t[-1]
				if query_t > label_t:
					label_tuple = dataset.get_node_label(query_t)
					if label_tuple is None:
						break
					label_ts, label_srcs, labels = (
						label_tuple[0],
						label_tuple[1],
						label_tuple[2],
					)
					label_t = dataset.get_label_time()
					if args.selected_feature in ['random_mix', 'node2vec_mix']:
						node_feats, unseen_nodes_degree = forward_feature_to(label_ts[0], node_feats, unseen_nodes_degree=unseen_nodes_degree)
					# if args.data == 'tgbn-trade':
					# 	label_t *= 3600 * 24 * 365
					n_id, n_ts = label_srcs.numpy(), label_ts.numpy()
					output_emb = get_node_emb(n_id.astype(np.int32), n_ts.astype(np.float32), label_ts[0])
					output_pred = node_pred(output_emb)				
					np_pred = output_pred.softmax(dim=1).cpu().detach().numpy()
					np_true = labels.cpu().detach().numpy()
					input_dict = {
						"y_true": np_true,
						"y_pred": np_pred,
						"eval_metric": [eval_metric],
					}
					result_dict = evaluator.eval(input_dict)
					score = result_dict[eval_metric]
					total_score += score
					num_label_ts += 1
					interval_split = 8
					new_split_list = []
					for k in range(interval_split):
						min_index = k * len(np_true)//interval_split
						max_index = max((k+1) * len(np_true)//interval_split, len(np_true))
						split_input_dict = {"y_true":np_true[min_index:max_index], "y_pred":np_pred[min_index:max_index], "eval_metric":[eval_metric]}
						split_result_dict = evaluator.eval(split_input_dict)
						new_split_list.append(split_result_dict[eval_metric])
			metric_dict = {}
			metric_dict[eval_metric] = total_score / num_label_ts
			return metric_dict, node_feats, unseen_nodes_degree, new_split_list


		

		train_curve = []
		val_curve = []
		test_curve = []
		max_val_score = 0  #find the best test score based on validation score
		best_test_idx = 0
		early_stop_count = 0
		test_time_list = []
		for epoch in range(1, args.epoch + 1):
			#train_loader = TemporalDataLoader(train_data, batch_size=train_param['batch_size'], shuffle=True)
			processed_edge_id = 0
			feature_processed_edge_id = len(train_data.src)
			if node_feats is not None:
				node_feats = fixed_node_feats.clone().cuda()
			unseen_nodes_degree = torch.zeros(node_count).cuda()
			start_time = timeit.default_timer()
			train_dict = train()
			print("------------------------------------")
			print(f"training Epoch: {epoch:02d}")
			print(train_dict)
			train_curve.append(train_dict[eval_metric])
			print("Training takes--- %s seconds ---" % (timeit.default_timer() - start_time))
			
			start_time = timeit.default_timer()
			val_dict, node_feats, unseen_nodes_degree, _ = test(val_loader, node_feats, unseen_nodes_degree)
			print(val_dict)
			val_curve.append(val_dict[eval_metric])

			print("Validation takes--- %s seconds ---" % (timeit.default_timer() - start_time))

			start_time = timeit.default_timer()
			
			test_dict, _, _, _ = test(test_loader, node_feats, unseen_nodes_degree)
			print(test_dict)
			test_curve.append(test_dict[eval_metric])
			test_time_list.append(timeit.default_timer() - start_time)
			print("Test takes--- %s seconds ---" % (timeit.default_timer() - start_time))
			print("------------------------------------")
			dataset.reset_label_time()
			if (val_dict[eval_metric] > max_val_score):
				max_val_score = val_dict[eval_metric]
				best_test_idx = epoch - 1
				early_stop_count = 0
			else:
				early_stop_count += 1
				if early_stop_count >= args.early_stop:
					break


		# code for plotting
		# plot_curve(train_curve, "train_curve")
		# plot_curve(val_curve, "val_curve")
		# plot_curve(test_curve, "test_curve")

		max_test_score = test_curve[best_test_idx]
		print("------------------------------------")
		print("------------------------------------")
		logger.info(f"best val score:{max_val_score}")
		logger.info(f"best validation epoch   : {best_test_idx + 1}")
		logger.info(f"best test score:{max_test_score}")
		logger.info(f'test time: mean:{np.mean(test_time_list)}, std:{np.std(test_time_list)}')
		final_result_list.append(max_test_score)
		final_val_results_list.append(max_val_score)
	final_result_np = np.array(final_result_list)
	final_val_results_np = np.array(final_val_results_list)
	print("------------------------------------")
	logger.info(f'final val results:{final_val_results_np.mean(), final_val_results_np.std()}')
	logger.info(f'final results:{final_result_np.mean(), final_result_np.std()}')




if __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('--data', type=str, help='dataset name')
	parser.add_argument('--config', type=str, default='', help='path to config file')
	parser.add_argument('--batch_size', type=int, default=600)
	parser.add_argument('--n_runs', type=int, default=1)
	parser.add_argument('--epoch', type=int, default=50)
	parser.add_argument('--dim', type=int, default=100)
	parser.add_argument('--n_degree', default=100)
	parser.add_argument('--early_stop', type=int, default=10)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
	parser.add_argument('--model', type=str, default='', help='name of stored model to load')
	parser.add_argument('--posneg', default=False, action='store_true', help='for positive negative detection, whether to sample negative nodes')
	parser.add_argument('--use_validation', default=False, action='store_true', help='Use validation set for early stopping')
	parser.add_argument('--train_ratio', type=float, default=0.1, help='total train ratio')
	parser.add_argument('--val_ratio', type=float, default=0.2, help='total train ratio')
	parser.add_argument('--weight_decay', type=float, default=0.0001)	
	parser.add_argument('--skip_alpha', type=float, default=1)
	parser.add_argument('--set_degree', default=False, action='store_true', help='set maximum degree')
	
	parser.add_argument('--selected_feature', type=str, default='empty', choices=['empty', 'zero', 'random', 'random_mix', 'node2vec_mix', 'structural'])
	parser.add_argument('--use_degree', default=False, action='store_true', help='memory update fast')

	


	args=parser.parse_args()
	print(str(args.model)[7:-4])
	os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
	torch.cuda.set_device(int(args.gpu))
	
    
	# set logger part
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	fh = logging.FileHandler('log/{}_{}_train_ratio_{}_val_ratio_{}.log'.format(args.config[7:-4],args.data,args.train_ratio,args.val_ratio))
	fh.setLevel(logging.DEBUG)
	ch = logging.StreamHandler()
	ch.setLevel(logging.WARN)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	logger.addHandler(fh)
	logger.addHandler(ch)   
		
	logger.info(args)
	logger.info('Data Processing')
	result = objective(args, logger)