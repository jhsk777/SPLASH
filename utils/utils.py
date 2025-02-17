import torch
import os
import yaml
import dgl
import time
import pandas as pd
import numpy as np
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt
import gc
import networkx as nx
from node2vec import Node2Vec
from sklearn.manifold import TSNE

def load_feat(d):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    return node_feats, edge_feats

def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('DATA/{}/ext_full.npz'.format(d))
    return g, df

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def to_dgl_blocks(ret, hist, reverse=False, cuda=True, device='cuda:0'):
    mfgs = list()
    for r in ret:
        if not reverse:
            #pdb.set_trace()
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
            b.srcdata['degree'] = torch.cat([torch.from_numpy(r.degrees()), torch.zeros(b.num_dst_nodes())], dim=0)
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
            b.srcdata['degree'] = torch.cat([torch.from_numpy(r.degrees()), torch.zeros(b.num_dst_nodes())], dim=0)
        b.edata['ID'] = torch.from_numpy(r.eid())
        if cuda:
            mfgs.append(b.to(device))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs

def node_to_dgl_blocks(root_nodes, ts, cuda=True, device='cuda:0'):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [b.to(device)])
    else:
        mfgs.insert(0, [b])
    return mfgs


def mfgs_to_cuda(mfgs, device='cuda:0'):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to(device)
            for key in mfg[i].srcdata:
                mfg[i].srcdata[key] = mfg[i].srcdata[key].cuda()
            for key in mfg[i].dstdata:
                mfg[i].dstdata[key] = mfg[i].dstdata[key].cuda()
            for key in mfg[i].edata:
                mfg[i].edata[key] = mfg[i].edata[key].cuda()
    return mfgs



def mfgs_to_cpu(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cpu')
            for key in mfg[i].srcdata:
                mfg[i].srcdata[key] = mfg[i].srcdata[key].detach()#.to('cpu')
                mfg[i].srcdata[key] = mfg[i].srcdata[key].to('cpu')
            for key in mfg[i].dstdata:
                mfg[i].dstdata[key] = mfg[i].dstdata[key].detach()
                mfg[i].dstdata[key] = mfg[i].dstdata[key].to('cpu')
            for key in mfg[i].edata:
                mfg[i].edata[key] = mfg[i].edata[key].detach()#.to('cpu')
                mfg[i].edata[key] = mfg[i].edata[key].to('cpu')
    return mfgs

def mfgs_clone(mfgs):
    new_mfgs = []
    for mfg in mfgs:
        new_mfg = []
        for i in range(len(mfg)):
            new_mfg.append(mfg[i].clone())
        new_mfgs.append(new_mfg)
    return new_mfgs

def mfgs_erase(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            for key in mfg[i].srcdata:
                mfg[i].srcdata.pop(key)
            for key in mfg[i].dstdata:
                mfg[i].dstdata.pop(key)
            for key in mfg[i].edata:
                mfg[i].edata.pop(key)
    return mfgs


def prepare_input(mfgs, node_feats, edge_feats, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None, edge_weight=None, device='cuda:0'):
    if combine_first:
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].float()
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                uts = unts[:, 0]
                unid = unts[:, 1]
                # import pdb; pdb.set_trace()
                b = dgl.create_block((idx + num_dst, mfgs[0][i].edges()[1]), num_src_nodes=unts.shape[0] + num_dst, num_dst_nodes=num_dst, device=torch.device(device))
                b.srcdata['ts'] = torch.cat([mfgs[0][i].srcdata['ts'][:num_dst], uts], dim=0)
                b.srcdata['ID'] = torch.cat([mfgs[0][i].srcdata['ID'][:num_dst], unid], dim=0)
                b.edata['dt'] = mfgs[0][i].edata['dt']
                b.edata['ID'] = mfgs[0][i].edata['ID']
                mfgs[0][i] = b
    t_idx = 0
    t_cuda = 0
    i = 0
    if node_feats is not None:
        for b in mfgs[0]:
            if pinned:
                if nids is not None:
                    idx = nids[i]
                else:
                    idx = b.srcdata['ID'].cpu().long()
                torch.index_select(node_feats, 0, idx, out=nfeat_buffs[i][:idx.shape[0]])
                b.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].to(device, non_blocking=True)
                i += 1
            else:
                srch = node_feats[b.srcdata['ID'].long()].float().contiguous()
                b.srcdata['h'] = srch.to(device)
    i = 0
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                #pdb.set_trace()
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:
                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
                        b.edata['f'] = efeat_buffs[i][:idx.shape[0]].to(device, non_blocking=True)
                        i += 1
                    else:
                        srch = edge_feats[b.edata['ID'].long()].float()
                        b.edata['f'] = srch.to(device)
    i=0
    if edge_weight is not None:
        for mfg in mfgs:
            for b in mfg:
                #pdb.set_trace()
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:
                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(edge_weight, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
                        b.edata['f'] = efeat_buffs[i][:idx.shape[0]].to(device, non_blocking=True)
                        i += 1
                    else:
                        srch = edge_weight[b.edata['ID'].long()].float()
                        b.edata['w'] = srch.to(device)
    return mfgs

def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].long())
    if 'ID' in mfgs[0][0].edata:
        if edge_feats is not None:
            for mfg in mfgs:
                for b in mfg:
                    eids.append(b.edata['ID'].long())
    else:
        eids = None
    return nids, eids

def get_pinned_buffers(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if 'neighbor' in sample_param:
        for i in sample_param['neighbor']:
            limit *= i + 1
            if edge_feats is not None:
                for _ in range(sample_param['history']):
                    pinned_efeat_buffs.insert(0, torch.zeros((limit, edge_feats.shape[1]), pin_memory=True))
    if node_feats is not None:
        for _ in range(sample_param['history']):
            pinned_nfeat_buffs.insert(0, torch.zeros((limit, node_feats.shape[1]), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs


def cosine_similarity(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def plot_curve(y: np.ndarray, outname: str) -> None:
    """
    plot the training curve given y
    Parameters:
        y: np.ndarray, the training curve
        outname: str, the output name
    """
    plt.plot(y, color="#fc4e2a")
    plt.savefig(outname + ".pdf")
    plt.close()



def cosine_similarity(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())



def normalize_adjacency_matrix(adj):
    """Normalize the adjacency matrix using symmetric normalization."""
    rowsum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

def eigen_decomposition(adj, k):
    """Perform eigen decomposition and return the top k eigen vectors."""
    # Normalize adjacency matrix
    adj_normalized = normalize_adjacency_matrix(adj)
    
    # Perform eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(adj_normalized)
    
    # Select the top k eigen vectors (eigvals are sorted in ascending order)
    top_k_eigvecs = eigvecs[:, -k:]
    
    return top_k_eigvecs


def torch_to_networkx(adj_matrix):
    G = nx.Graph()
    num_nodes = adj_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            G.add_edge(i, j, weight=adj_matrix[i][j])
    return G

def plot_tsne(feat, labels, x_size=15, y_size=15, title='t-SNE of Random Features', output='contrast'):
    tsne = TSNE(n_components=2, random_state=42, n_iter=300, learning_rate=100, init='random')
    tsne_results = tsne.fit_transform(feat)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, s=10, cmap='tab10')
    plt.title(title, fontsize=20)
    plt.xlim(-x_size, x_size)
    plt.ylim(-y_size, y_size)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.savefig('contrastive_plot/' + output + ".png")
    #plt.colorbar(scatter, ticks=range(num_classes), label='Classes')
    plt.show()


def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            node = parts[0]
            embedding = list(map(float, parts[1:]))
            embeddings[node] = embedding
    return embeddings