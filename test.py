import os
import pandas as pd
import torch.nn as nn
from evaluate import evaluate
from embedder import embedder
from utils.process import GCN, update_S, drop_feature, Linearlayer, get_multi_agg, global_heter, node_average_heter, CCASSL
import numpy as np
from tqdm import tqdm
import random as random
import torch
from typing import Any, Optional, Tuple
import torch.nn.functional as F
import networkx as nx
import torch
from sklearn.metrics import *
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.nn.conv import GCNConv
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)


class MPGRL(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.w_list = nn.ModuleList([nn.Linear(self.args.out_channels*2, self.args.out_channels, bias=True) for _ in range(self.args.num_view)]).to(self.args.device) # 2个线性层 (128, 128)
        self.y_list = nn.ModuleList([nn.Linear(self.args.out_channels, 1) for _ in range(self.args.num_view)]).to(self.args.device) # 2个线性层 (128, 1)
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)

        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    # def combine_att(self, h_list):
    #     h_combine_list = []
    #     for i, h in enumerate(h_list): # enumerate(h_list) 将 h_list 中的元素同时返回索引和值。
    #         h = self.w_list[i](h).to(self.args.device) # 线性层 128, 128
    #         h = self.y_list[i](h).to(self.args.device) # 线性层 128, 1 输出 [N = 3025,1]
    #         h_combine_list.append(h)
    #     score = torch.cat(h_combine_list, -1) # 按最后一个维度拼接 [3025,2]
    #     score = self.att_act1(score)
    #     score = self.att_act2(score) # softmax
    #     score = torch.unsqueeze(score, -1) # 增加一个维度 [3025,2,1]
    #     h = torch.stack(h_list, dim=1) # 按照第一个维度进行堆叠 [3025,2,128]
        
    #     h = score * h
    #     h = torch.sum(h, dim=1) # [3025,128]
    #     return h.detach()
    

    def training(self):

        # -------------------------固定随机种子------------------------- #
        seed = self.args.seed
        np.random.seed(seed) # 设置NumPy的随机数生成器的种子,NumPy生成的随机数序列将是确定的
        torch.manual_seed(seed) # PyTorch在CPU上生成的随机数（如初始化权重等）将是确定的
        torch.cuda.manual_seed_all(seed) # 为所有GPU设置种子
        random.seed(seed) # 使用卷积操作时的确定性。
        torch.backends.cudnn.deterministic = True # 确保每次运行使用相同的卷积算法
        torch.backends.cudnn.benchmark = False

        # -------------------------特征矩阵列表------------------------- #
        # 都是归一化过的
        features = [feature.to(self.args.device) for feature in self.features] # acm: 2 3025 1870 
        adj_list = [adj.to(self.args.device) for adj in self.adj_list] # acm: 2 3025 3025
        
        for i in range(self.args.num_view):
            features[i] = drop_feature(features[i], self.args.feature_drop)

        self.labels = self.labels.to(self.args.device)

        # ---------------------torch_geometric.data--------------------- #
        pyg_list = []

        for feature, adj in zip(features, adj_list):
            # 将稠密的邻接矩阵转换为稀疏格式
            edge_index = adj.nonzero(as_tuple=False).t()
            # 创建 Data 对象
            data = Data(x=feature, edge_index=edge_index)
            pyg_list.append(data)

        # 随机游走（多次不同长度随机游走的拼接）获取的结构特征x_neighbor， 节点自身特征x
        for i in range(self.args.num_view):
            pyg_list[i].x_neighbor, pyg_list[i].x = get_multi_agg(pyg_list[i].x, pyg_list[i])
            pyg_list[i].y = self.labels
            print(pyg_list[i])

        pyg_list = [pyg.to(self.args.device) for pyg in pyg_list]
        
        # 计算异配度
        # for i in range(self.args.num_view):
        #     print('global_heter:'+str(global_heter(pyg_list[i])))
        #     print('node_average_heter:'+str(node_average_heter(pyg_list[i])))

        # ---------------------model--------------------- #
        models = nn.ModuleList()
        for i in range(self.args.num_view):
            models.append(MVGE(pyg_list[i].x.shape[1], pyg_list[i].x_neighbor.shape[1], self.args.out_channels).to(self.args.device))       
        

        # ---------------------定义优化器--------------------- #
        if self.args.num_view == 2:
            optimizer = torch.optim.Adam([
                {'params': models[0].parameters(), 'lr': self.args.lr},
                {'params': models[1].parameters(), 'lr': self.args.lr}
            ])
        else:
            optimizer = torch.optim.Adam([
                {'params': models[0].parameters(), 'lr': self.args.lr},
                {'params': models[1].parameters(), 'lr': self.args.lr},    
                {'params': models[2].parameters(), 'lr': self.args.lr}           
            ])

        best = 1e9
        max_Macro_F1 = 0
        max_Micro_F1 = 0

        # --------------------- training --------------------- #
        print("Started training ori-MVGE ...")
        I_target = torch.tensor(np.eye(self.args.out_channels * 2)).to(self.args.device)
        
        print(self.args.num_iters)
        for epoch in tqdm(range(self.args.num_iters)):     
            for i in range(self.args.num_view):
                models[i].train()       
            loss = 0
            optimizer.zero_grad()

            # ----------------------- 单图定制loss ----------------------- 
                                       
            # for i in range(self.args.num_view):
            #     losses.append(models[i].loss(pyg_list[i].x, pyg_list[i].x_neighbor, pyg_list[i].edge_index, pyg_list[i].edge_index, self.args.mvge_loss_weight[0], self.args.mvge_loss_weight[1]))
            # for i in range(self.args.num_view):
            #     losses[i].backward()

            for i in range(self.args.num_view):
                loss =+ models[i].loss(pyg_list[i].x, pyg_list[i].x_neighbor, pyg_list[i].edge_index, pyg_list[i].edge_index, self.args.mvge_loss_weight[0], self.args.mvge_loss_weight[1])

            # loss.backward()    
            # optimizer.step() 

            # ----------------------- 一致性loss ----------------------- 
            z_emb = []            

            for i in range(self.args.num_view):
                z,z1,z2 = models[i].embed(pyg_list[i].x, pyg_list[i].x_neighbor,pyg_list[i].edge_index)
                z_emb.append(z)

            # # CoCo一致性提取方法
            # # loss_C, loss_simi = CCASSL(z_emb, pyg_list[0].x.shape[0], I_target, self.args.num_view)
            # # loss_total = loss + (1 - loss_simi + loss_C) 
            # # loss_total.backward()     
                
            # # MGDCR一致性提取方法
            loss_inter = 0
            for i in range(self.args.num_view):
                if i == 1 and self.args.num_view == 2:
                    break
                inter_c = (z_emb[i]).T @ (z_emb[(i + 1) % self.args.num_view])
                on_diag_inter = torch.diagonal(inter_c).add_(-1).pow_(2).sum() # sum(k_ii - 1)
                off_diag_inter = off_diagonal(inter_c).pow_(2).sum() # k_ij
                loss_inter += (on_diag_inter + 0.002 * off_diag_inter)# w' * L_intre
            loss_total = loss + loss_inter
            loss_total.backward()   
            optimizer.step()

            # 看每轮结果
            for i in range(self.args.num_view):
                models[i].eval()

            embedding = []
            for i in range(self.args.num_view):            
                z,z1,z2  = models[i].embed(pyg_list[i].x, pyg_list[i].x_neighbor,pyg_list[i].edge_index)  
                embedding.append(z.detach())            

            embeddings = sum(embedding) / self.args.num_view # 平均池化
            # embeddings = self.combine_att(embedding) # 注意力聚合
            
            macro_f1s, micro_f1s = evaluate(embeddings, self.idx_train, self.idx_val, self.idx_test, self.labels,task=self.args.custom_key,epoch = self.args.test_epo,lr = self.args.test_lr,iterater=self.args.iterater) #,seed=seed
            
            if np.mean(macro_f1s) > max_Macro_F1:
                max_itr_1 = epoch
                max_Macro_F1 = np.mean(macro_f1s)
            if np.mean(micro_f1s) > max_Micro_F1:
                max_itr_2 = epoch
                max_Micro_F1 = np.mean(micro_f1s)
            
            # print('====> Iteration: {} Loss = {:.4f} macro_f1s = {} micro_f1s = {}'.format(epoch, loss_total, macro_f1s, micro_f1s))
            print('====> Iteration: {} Loss = {:.4f} macro_f1s = {} micro_f1s = {}'.format(epoch, loss, np.mean(macro_f1s), np.mean(micro_f1s)))

        print(max_itr_1, max_itr_2, np.mean(macro_f1s), np.mean(micro_f1s))

        # --------------------- evaluating --------------------- #
        print("Evaluating...")                
        print("Started Evaluating ori-MVGE + mc ...")
        for i in range(self.args.num_view):
            models[i].eval()

        embedding = []
        for i in range(self.args.num_view):            
            z,z1,z2  = models[i].embed(pyg_list[i].x, pyg_list[i].x_neighbor,pyg_list[i].edge_index)  
            embedding.append(z.detach())        

        embeddings = sum(embedding) / self.args.num_view # 平均池化
        # embeddings = self.combine_att(embedding) # 注意力聚合
        
        macro_f1s, micro_f1s = evaluate(embeddings, self.idx_train, self.idx_val, self.idx_test, self.labels,task=self.args.custom_key,epoch = self.args.test_epo,lr = self.args.test_lr,iterater=self.args.iterater) #,seed=seed
        return macro_f1s, micro_f1s

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradientReversalLayer.apply(x, coeff)

EPS = 1e-15
MAX_LOGSTD = 10   
class GAE(torch.nn.Module):

    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder


    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)


    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)


    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):


        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss


    def test(self, z, pos_edge_index, neg_edge_index):

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        
        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

class LinearEncoder(nn.Module):
    def __init__(self, in_channels_self,in_channels_agg, out_channels):
        super(LinearEncoder, self).__init__()
        self.linear_in_self = nn.Linear(in_channels_self, out_channels*2)
        self.linear_out_self = nn.Linear(in_channels_self+out_channels*2,out_channels)

        self.gcn1 = GCNConv(in_channels_self,out_channels*2)
        self.gcn2 = GCNConv(out_channels*2,out_channels)
        self.gcn3 = GCNConv(out_channels,out_channels)
        
        # self.linear_out = nn.Linear(in_channels_agg+out_channels*2,out_channels)
        self.linear_out = nn.Linear(in_channels_agg+out_channels, out_channels)

    def forward(self,x_self,x_neighbor,edge_index):
        #nn.ReLU()

        #  MLP(x_self 拼接 MLP(x_self))
        l1 = F.relu(self.linear_in_self(x_self))
        l1 = torch.cat((x_self,l1),1)
        l1 = self.linear_out_self(l1)
        
        #  MLP(x_neighbor 拼接 GCN(x_neighbor) 拼接 GCN(GCN(x_neighbor)))         
        g1 = self.gcn1(x_self,edge_index)
        g2 = self.gcn2(g1,edge_index)
        # g3 = self.gcn3(g2,edge_index)
        
        #concat拼接
        # x2 = torch.cat((x_neighbor,g1),1)
        # x2 = self.linear_out(x_neighbor) # 只用随机游走提取结构信息
        x2 = torch.cat((x_neighbor,g2),1)
        x2 = self.linear_out(x2)
        return l1,x2

class MVGE(GAE):
    # MVGE1(data.x.shape[1],data.x_neighbor.shape[1],out_channels).to(device)
    def __init__(self, in_channels_self,in_channels_agg,out_channels):
        super(MVGE, self).__init__(encoder=LinearEncoder(in_channels_self,
                                                            in_channels_agg,
                                                          out_channels),
                                       decoder=InnerProductDecoder())
        
        self.decoder_self = nn.Linear(out_channels,in_channels_self)
        self.decoder_neighbor = nn.Linear(out_channels,in_channels_agg)

    # ego||agg 重构 adj
    def forward(self, x_self, x_neighbor, pos_edge_index):
        z_self = self.encode(x_self, x_neighbor, pos_edge_index)
        adj_pred = self.decoder.forward_all(z_self)
        return adj_pred

    # ego||agg, ego, agg 
    def embed(self,x_self,x_neighbor,edge_index):
        z1,z2=self.encode(x_self,x_neighbor,edge_index)
        z = torch.cat((z1,z2),1)
        return z,z1,z2
    
    def loss(self, x_self, x_neighbor, pos_edge_index, all_edge_index,a,b):
        # loss = model.loss(data.x,data.x_neighbor, data.edge_index, data.edge_index,a,b)

        z1,z2 = self.encode(x_self,x_neighbor,pos_edge_index)
        z_self = torch.cat((z1,z2),1) # ego||agg


        # ---------------------- ego||agg 重构 adj 损失 ----------------------
        pos_loss = -torch.log(
            self.decoder(z_self, pos_edge_index, sigmoid=True) + 1e-15).mean()

        all_edge_index_tmp, _ = remove_self_loops(all_edge_index) # 去除现有的自环
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp) # 添加所有的自环
       
        neg_edge_index = negative_sampling(all_edge_index_tmp, z_self.size(0), pos_edge_index.size(1)) # 生成负样本边索引        
        neg_loss = -torch.log(1 - self.decoder(z_self, neg_edge_index, sigmoid=True) + 1e-15).mean()
        
        # ---------------------- ego 重构 kl损失---------------------- 
        z_self = self.decoder_self(z1)
        loss1 = F.kl_div(F.log_softmax(z_self,dim=1),x_self,reduction='mean')
        # ---------------------- agg 重构 kl损失---------------------- 
        z_neighbor = self.decoder_neighbor(z2)
        loss2 = F.kl_div(F.log_softmax(z_neighbor,dim=1),x_neighbor,reduction='mean')
                
        # return loss1 + loss2 + (neg_loss+pos_loss)*0.01        
        return b*(a*loss1 + (1-a)*loss2) + (1-b)*(neg_loss+pos_loss)*0.01


    def single_test(self, x_self,x_neighbor, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z,z1,z2 = self.embed(x_self,x_neighbor,train_pos_edge_index)

        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score

    
    def trans_single_node(self,z):
        return torch.mean(torch.stack([z[0:z.shape[0]-2:3],z[1:z.shape[0]-1:3],z[2:z.shape[0]:3]]),0)
    

def off_diagonal(x): # 非对角线元素视图
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()






