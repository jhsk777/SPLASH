import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import math
import numpy as np
import pdb


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim)) 

        for _ in range(num_layers - 1):
            layers.append(nn.ReLU()) 
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(nn.ReLU()) 

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
    
    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.mlp(x)


class NodeClassificationModel(torch.nn.Module):

    def __init__(self, dim_in, dim_hid, num_class, dropout=0.2):
        super(NodeClassificationModel, self).__init__()
        self.ln_layer = torch.nn.LayerNorm(dim_in)
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.hidden_fc1 = torch.nn.Linear(dim_hid, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_emb(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        return x



class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """
    def __init__(self, dims, expansion_factor, dropout=0, use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer
        
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if self.use_single_layer==False:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.use_single_layer==False:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TimeEncode(torch.nn.Module):

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output
    

class FixedTimeEncode(torch.nn.Module):

    def __init__(self, dim, alpha = 10):
        super(FixedTimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / alpha ** np.linspace(0, alpha-1, dim, dtype=np.float32))).reshape(dim, -1)).requires_grad_(False)
        self.w.bias = torch.nn.Parameter(torch.zeros(dim)).requires_grad_(False)
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output
    
class Global_TimeEncode(torch.nn.Module):

    def __init__(self, dim):
        super(Global_TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim//2, bias=False)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 4, dim//2, dtype=np.float32))).reshape(dim//2, -1)).requires_grad_(False)
        #self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 2, dim//2, dtype=np.float32))).reshape(dim//2, -1)).requires_grad_(False)
        self.w.bias = torch.nn.Parameter(torch.zeros(dim//2)).requires_grad_(False)
        
    @torch.no_grad()
    def forward(self, t):
        #pdb.set_trace()
        cos_output = torch.cos(self.w(t.reshape((-1, 1))))
        sin_output = torch.sin(self.w(t.reshape((-1, 1))))
        output = torch.cat([cos_output, sin_output], dim=-1)
        return output


class EdgePredictor(torch.nn.Module):

    def __init__(self, dim_in):
        super(EdgePredictor, self).__init__()
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)
        self.out_fc = torch.nn.Linear(dim_in, 1)

    def forward(self, h, neg_samples=1):
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)
    
    def get_positive_edge_score(self, h):
        num_edge = h.shape[-1] // 2
        h_src = self.src_fc(h[:,:,:num_edge])
        h_pos_dst = self.dst_fc(h[:,:,num_edge:])
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        return self.out_fc(h_pos_edge)


class single_Linear(torch.nn.Module):
    def __init__(self, dim_node_feat, class_num):
        super(single_Linear, self).__init__()
        self.linear_layer = torch.nn.Linear(dim_node_feat, class_num, bias=False)
        #self.linear_layer = MLP(dim_node_feat, dim_node_feat, class_num)
    
    def forward(self, x):
        return self.linear_layer(x)

class ID_Model_full_mean(torch.nn.Module): # MLP_simple Layer

    def __init__(self, hidden_dim, device='cuda:0'):
        super(ID_Model_full_mean, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, b):
        neighbor_feat = b.srcdata['h'][b.num_dst_nodes():]

        b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), neighbor_feat.shape[1]), device=torch.device(self.device)), neighbor_feat], dim=0)
        b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.mean('m', 'h1'))
        target_input = torch.cat([b.dstdata['h1'][:,:self.hidden_dim], 
                                  b.dstdata['h1'][:,self.hidden_dim:self.hidden_dim*2],
                                  b.dstdata['h1'][:,self.hidden_dim*2:self.hidden_dim*3],
                                  b.srcdata['h'][:b.num_dst_nodes(), :self.hidden_dim],
                                  b.srcdata['h'][:b.num_dst_nodes(), self.hidden_dim:self.hidden_dim*2],
                                  b.srcdata['h'][:b.num_dst_nodes(), self.hidden_dim*2:]], dim=1)
        rst = target_input
        return rst


class SLIM(torch.nn.Module): # Proposed simple MLP-based model

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, dropout, dim_out, skip_alpha=1, device='cuda:0'):
        super(SLIM, self).__init__()
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.skip_alpha = skip_alpha
        self.dropout = torch.nn.Dropout(dropout)
        if dim_time > 0:
            self.time_enc = FixedTimeEncode(dim_time)
        self.time_node_edge_proj = MLP(dim_node_feat + dim_edge_feat + dim_time, dim_out, dim_out)
        self.combine_func = MLP(dim_node_feat + dim_out, dim_out, dim_out)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.layer_norm1 = torch.nn.LayerNorm(dim_out)
        self.layer_norm2 = torch.nn.LayerNorm(dim_out)
        self.device = device
    
    def reset_parameters(self):
        self.time_node_edge_proj.reset_parameters()
        self.combine_func.reset_parameters()
        self.layer_norm.reset_parameters()
        self.layer_norm1.reset_parameters()
        self.layer_norm2.reset_parameters()

    def forward(self, b):
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        #pdb.set_trace()
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device(self.device))
        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
        if 'w' in b.edata:
            edge_weight = b.edata['w']
        else:
            edge_weight = torch.ones(b.edata['dt'].shape[0], 1, device=torch.device(self.device))

        att = edge_weight
        b.srcdata['h'] = self.dropout(b.srcdata['h'])
        if self.dim_edge_feat == 0:
            time_node_edge_V = self.time_node_edge_proj(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
        else:
            time_node_edge_V = self.time_node_edge_proj(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
        time_node_edge_V = time_node_edge_V * att
        b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), time_node_edge_V.shape[1]), device=torch.device(self.device)), time_node_edge_V], dim=0)
        
        b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.mean('m', 'h1'))
        b.update_all(dgl.function.copy_u('v', 'm1'), dgl.function.sum('m1', 'h2'))
        target_input = torch.cat([b.dstdata['h1'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
        target_rst7 = self.combine_func(target_input)
        rst = target_rst7
        return self.layer_norm(rst) + self.skip_alpha * self.layer_norm2(b.dstdata['h2'])



class TransfomerAttentionLayer(torch.nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False, device='cuda:0'):
        super(TransfomerAttentionLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.combined = combined
        if dim_time > 0:
            self.time_enc = FixedTimeEncode(dim_time)
        if combined:
            if dim_node_feat > 0:
                self.w_q_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_k_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_v_n = torch.nn.Linear(dim_node_feat, dim_out)
            if dim_edge_feat > 0:
                self.w_k_e = torch.nn.Linear(dim_edge_feat, dim_out)
                self.w_v_e = torch.nn.Linear(dim_edge_feat, dim_out)
            if dim_time > 0:
                self.w_q_t = torch.nn.Linear(dim_time, dim_out)
                self.w_k_t = torch.nn.Linear(dim_time, dim_out)
                self.w_v_t = torch.nn.Linear(dim_time, dim_out)
        else:
            if dim_node_feat + dim_time > 0:
                self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
            self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
            self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)
        self.device=device
        #self.w_out = torch.nn.Linear(dim_node_feat, dim_out)
        
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device(self.device))
        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device(self.device)))
        if self.combined:
            Q = torch.zeros((b.num_edges(), self.dim_out), device=torch.device(self.device))
            K = torch.zeros((b.num_edges(), self.dim_out), device=torch.device(self.device))
            V = torch.zeros((b.num_edges(), self.dim_out), device=torch.device(self.device))
            if self.dim_node_feat > 0:
                Q += self.w_q_n(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K += self.w_k_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
                V += self.w_v_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
            if self.dim_edge_feat > 0:
                K += self.w_k_e(b.edata['f'])
                V += self.w_v_e(b.edata['f'])
            if self.dim_time > 0:
                Q += self.w_q_t(zero_time_feat)[b.edges()[1]]
                K += self.w_k_t(time_feat)
                V += self.w_v_t(time_feat)
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.edata['v'] = V
            b.update_all(dgl.function.copy_edge('v', 'm'), dgl.function.sum('m', 'h'))
        else:
            if self.dim_time == 0 and self.dim_node_feat == 0:
                Q = torch.ones((b.num_edges(), self.dim_out), device=torch.device(self.device))
                K = self.w_k(b.edata['f'])
                V = self.w_v(b.edata['f'])
            elif self.dim_time == 0 and self.dim_edge_feat == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(b.srcdata['h'][b.num_dst_nodes():])
                V = self.w_v(b.srcdata['h'][b.num_dst_nodes():])
            elif self.dim_time == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
            elif self.dim_node_feat == 0:
                Q = self.w_q(zero_time_feat)[b.edges()[1]]
                K = self.w_k(torch.cat([b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.edata['f'], time_feat], dim=1))
            elif self.dim_edge_feat == 0:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
            else:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            #pdb.set_trace()
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device(self.device)), V], dim=0)
            b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        if self.dim_node_feat != 0:
            rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
            #rst = b.srcdata['h'][:b.num_dst_nodes()]
        else:
            rst = b.dstdata['h']
        rst = self.w_out(rst)
        #pdb.set_trace()
        rst = torch.nn.functional.relu(self.dropout(rst))
        return self.layer_norm(rst)

class IdentityNormLayer(torch.nn.Module):

    def __init__(self, dim_out):
        super(IdentityNormLayer, self).__init__()
        self.norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        return self.norm(b.srcdata['h'])

class JODIETimeEmbedding(torch.nn.Module):

    def __init__(self, dim_out):
        super(JODIETimeEmbedding, self).__init__()
        self.dim_out = dim_out

        class NormalLinear(torch.nn.Linear):
        # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.time_emb = NormalLinear(1, dim_out)
    
    def forward(self, h, mem_ts, ts):
        time_diff = (ts - mem_ts) / (ts + 1)
        rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
        return rst
            