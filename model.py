import math
import torch
from dgl.nn import GraphConv, GATConv, SAGEConv, SGConv, TAGConv
from dgl.nn.pytorch.glob import SumPooling
from torch.nn import ModuleList
from torch.nn.functional import one_hot
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
np.set_printoptions(threshold=np.inf)


class MNet(nn.Module):
    
    def __init__(self, args):
        super(MNet, self).__init__()
        self.dim_atom = args.dim_atom
        self.dim_graph = args.dim_graph
        self.KeyM = nn.Linear(2*(self.dim_atom+self.dim_graph), 10)
        self.MemM = nn.Parameter(torch.FloatTensor(10, self.dim_atom+self.dim_graph))
        nn.init.xavier_uniform_(self.KeyM.weight)
        nn.init.xavier_uniform_(self.MemM)

    def forward(self, emb1, emb2):
        att_vec = F.softmax(self.KeyM(torch.cat([emb1.detach(), emb2.detach()], dim=1)), dim=1)
        relation_vec = torch.mm(att_vec, self.MemM)
        return relation_vec


class GNN(torch.nn.Module):
    def __init__(self, args, feature_len, all_mfea, all_rfea, neighbor_info):
        super(GNN, self).__init__()
        
        self.gnn = args.gnn
        self.n_layer = args.layer
        self.feature_len = feature_len
        self.dim_atom = args.dim_atom
        self.dim_graph = args.dim_graph
        self.gnn_layers = ModuleList([])
        if self.gnn in ['gcn', 'gat', 'sage', 'tag', 'sgc']:
            for i in range(self.n_layer):
                if self.gnn == 'gcn':
                    self.gnn_layers.append(GraphConv(in_feats=feature_len if i == 0 else self.dim_atom,
                                                     out_feats=self.dim_atom,
                                                     activation=None if i == self.n_layer - 1 else torch.relu))
                elif self.gnn == 'gat':
                    num_heads = 4  
                    self.gnn_layers.append(GATConv(in_feats=feature_len if i == 0 else self.dim_atom,
                                                   out_feats=self.dim_atom // num_heads,
                                                   activation=None if i == self.n_layer - 1 else torch.relu,
                                                   num_heads=num_heads))
                elif self.gnn == 'sage':
                    agg = 'pool'
                    self.gnn_layers.append(SAGEConv(in_feats=feature_len if i == 0 else self.dim_atom,
                                                    out_feats=self.dim_atom,
                                                    activation=None if i == self.n_layer - 1 else torch.relu,
                                                    aggregator_type=agg))
                elif self.gnn == 'tag':
                    hops = 2
                    self.gnn_layers.append(TAGConv(in_feats=feature_len if i == 0 else self.dim_atom,
                                                   out_feats=self.dim_atom,
                                                   activation=None if i == self.n_layer - 1 else torch.relu,
                                                   k=hops))
        elif self.gnn == 'sgc':
            self.gnn_layers.append(SGConv(in_feats=feature_len, out_feats=self.dim_atom, k=self.n_layer))
        else:
            raise ValueError('unknown GNN model')
        self.pooling_layer = SumPooling()
        self.factor = None

        self.all_mfea = all_mfea
        self.all_rfea = all_rfea
        self.neighbor_matrix = neighbor_info[0]
        self.relation_matrix = neighbor_info[1]
        self.num_neighbor = args.num_neighbor
        self.fea_emb = nn.Parameter(torch.FloatTensor(40, self.dim_graph))
        self.rel_emb = nn.Parameter(torch.FloatTensor(4, self.dim_graph)) 
        nn.init.xavier_uniform_(self.fea_emb)
        nn.init.xavier_uniform_(self.rel_emb)
        
        self.RG_agg = args.RG_agg
        if self.RG_agg == 'att':
            self.att_linear  = nn.Linear(self.dim_graph, 1)
            nn.init.xavier_uniform_(self.att_linear.weight)
            self.tanH = nn.Tanh()
            self.sigmoid = nn.Sigmoid()
        elif self.RG_agg == 'att_gate':
            self.att_linear  = nn.Linear(self.dim_graph, 1)
            self.gate_linear = nn.Linear(self.dim_graph, self.dim_graph)
            nn.init.xavier_uniform_(self.att_linear.weight)
            nn.init.xavier_uniform_(self.gate_linear.weight)
            self.tanH = nn.Tanh()
            self.sigmoid = nn.Sigmoid()
        self.graph_agg = args.graph_agg
        if self.graph_agg == 'linear':
            self.combine  = nn.Linear(self.dim_atom+self.dim_graph, self.dim_atom+self.dim_graph)
            nn.init.xavier_uniform_(self.combine.weight)

        self.crosscl = nn.Linear(self.dim_graph, self.dim_atom)
        nn.init.xavier_uniform_(self.crosscl.weight)

    def forward(self, graph, id_list, id_list_target):
        feature = graph.ndata['feature'] 
        h = one_hot(feature, num_classes=self.feature_len)  
        h = torch.sum(h, dim=1, dtype=torch.float) 
        for layer in self.gnn_layers:
            h = layer(graph, h)
            if self.gnn == 'gat':
                h = torch.reshape(h, [h.size()[0], -1])
        if self.factor is None:
            self.factor = math.sqrt(self.dim_atom) / float(torch.mean(torch.linalg.norm(h, dim=1)))
        h *= self.factor
        mol_atom_embedding = self.pooling_layer(graph, h)
        
        source_list = []
        for i in range(len(id_list)):
            id_th_source = id_list[i]  
            source_raw_feature = self.all_mfea[id_th_source.long()].float()  
            source_emb_feature = torch.mm(source_raw_feature, self.fea_emb) 
            source_list.append(source_emb_feature) 
        source_reaction_embedding = (1/len(source_list)) * sum(source_list)

        target_list = []
        for i in range(len(id_list)):
            id_th_target = id_list_target[i]  
            target_raw_feature = self.all_mfea[id_th_target.long()].float()  
            target_emb_feature = torch.mm(target_raw_feature, self.fea_emb) 
            target_list.append(target_emb_feature) 
        target_reaction_embedding = (1/len(target_list)) * sum(target_list)
            
        agg_list = []
        for i in range(len(id_list)):
            id_th = id_list[i] 
            id_th_target = id_list_target[i]  

            nei_th = torch.from_numpy(self.neighbor_matrix[id_th.numpy()]) 
            rel_th = torch.from_numpy(self.relation_matrix[id_th.numpy()])  
            
            item_raw_feature = self.all_mfea[id_th.long()].float()  
            item_emb_feature = torch.mm(item_raw_feature, self.fea_emb)  

            nei_raw_feature = self.all_mfea[nei_th.long()].float()
            nei_raw_feature = nei_raw_feature.reshape(-1, 40)
            nei_emb_feature = torch.mm(nei_raw_feature, self.fea_emb)
            nei_emb_feature = nei_emb_feature.reshape(-1, self.num_neighbor, self.dim_graph)  
            
            rel_raw_feature = self.all_rfea[rel_th.long()].float()
            rel_raw_feature = rel_raw_feature.reshape(-1, 4)
            rel_emb_feature = torch.mm(rel_raw_feature, self.rel_emb)
            rel_emb_feature = rel_emb_feature.reshape(-1, self.num_neighbor, self.dim_graph)  
            
            if self.RG_agg == 'att':
                self_embedding = item_emb_feature.unsqueeze(1).repeat(1, self.num_neighbor, 1)  
                target_embedding = (source_reaction_embedding + target_reaction_embedding).unsqueeze(1).repeat(1, self.num_neighbor, 1)  
                score_nei = (self_embedding * nei_emb_feature * target_embedding).view(-1, self.dim_graph)
                score_att = self.att_linear(score_nei).view(-1, self.num_neighbor, 1)
                score_att = F.softmax(score_att.float(), 1)
                
                nei_emb_feature = torch.sum(score_att * nei_emb_feature, 1)

            elif self.RG_agg == 'att_gate':
                self_embedding = item_emb_feature.unsqueeze(1).repeat(1, self.num_neighbor, 1)  
                target_embedding = (source_reaction_embedding + target_reaction_embedding).unsqueeze(1).repeat(1, self.num_neighbor, 1)  
                score_nei = (self_embedding * nei_emb_feature * target_embedding).view(-1, self.dim_graph)
                score_att = self.att_linear(score_nei).view(-1, self.num_neighbor, 1)
                score_att = F.softmax(score_att.float(), 1)
                
                neighbor_gate = self.sigmoid(self.gate_linear(score_nei)).view(-1, self.num_neighbor, self.dim_graph)
                nei_emb_feature = torch.sum(score_att * (neighbor_gate * nei_emb_feature), 1)

            else: 
                nei_emb_feature = torch.sum(nei_emb_feature, dim=1)   
            
            agg_list.append(item_emb_feature + nei_emb_feature)
        
        mol_reaction_embedding = (1/len(agg_list)) * sum(agg_list)
        
        mol_x_embedding = self.crosscl(mol_reaction_embedding)
        mol_a_embedding = mol_atom_embedding
        
        if self.graph_agg == 'sum':
            mol_embedding = mol_atom_embedding + mol_reaction_embedding
        elif self.graph_agg == 'cat':
            mol_embedding = torch.cat([mol_atom_embedding, mol_reaction_embedding], dim=1)
        elif self.graph_agg == 'linear':
            mol_embedding = self.combine(torch.cat([mol_atom_embedding, mol_reaction_embedding], dim=1))
        else:
            raise ValueError('unknown graph embedding aggregation')
        
        return mol_embedding, mol_a_embedding, mol_x_embedding



