import os
import dgl
import torch
import pickle
import pysmiles
from collections import defaultdict

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt


from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import FragmentCatalog
from collections import defaultdict

import numpy as np
import random

np.set_printoptions(threshold=np.inf)


attribute_names = ['element', 'charge', 'aromatic', 'hcount']


class SmilesDataset(dgl.data.DGLDataset):
    def __init__(self, args, mode, feature_encoder=None, raw_graphs=None, id_list=None):
        self.args = args
        self.mode = mode
        self.feature_encoder = feature_encoder
        self.raw_graphs = raw_graphs
        self.id_list = id_list
        self.reactant_graphs = []
        self.product_graphs = []
        super().__init__(name='Smiles_' + mode)

    def process(self):   
        for i, (raw_reactant_graph, raw_product_graph) in enumerate(self.raw_graphs):
            reactant_graph = networkx_to_dgl(raw_reactant_graph, self.feature_encoder)
            product_graph = networkx_to_dgl(raw_product_graph, self.feature_encoder)
            self.reactant_graphs.append(reactant_graph)
            self.product_graphs.append(product_graph)

    def __getitem__(self, i):    
        return self.reactant_graphs[i], self.product_graphs[i], self.id_list[i]

    def __len__(self):   
        return len(self.reactant_graphs)


def networkx_to_dgl(raw_graph, feature_encoder):
    src = [s for (s, _) in raw_graph.edges]
    dst = [t for (_, t) in raw_graph.edges]

    graph = dgl.graph((src, dst), num_nodes=len(raw_graph.nodes))
    node_features = []
    for i in range(len(raw_graph.nodes)):
        raw_feature = raw_graph.nodes[i]
        numerical_feature = []
        for j in attribute_names:
            if raw_feature[j] in feature_encoder[j]:
                numerical_feature.append(feature_encoder[j][raw_feature[j]])
            else:
                numerical_feature.append(feature_encoder[j]['unknown'])
        node_features.append(numerical_feature)
    node_features = torch.tensor(node_features)
    graph.ndata['feature'] = node_features
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)

    return graph


def read_data(dataset, mode, molecular_num_dict, max_rea_num, max_pro_num):  

    max_rea_num = 2
    max_pro_num = 2

    path = '../data/' + dataset + '/' + mode + '.txt'

    all_values = defaultdict(set)
    graphs = []
    all_id_list = []

    with open(path) as f:
        for line in f.readlines():
            idx, reactant_smiles, product_smiles = line.strip().split('\t')

            if '[se]' in reactant_smiles:
                reactant_smiles = reactant_smiles.replace('[se]', '[Se]')
            if '[se]' in product_smiles:
                product_smiles = product_smiles.replace('[se]', '[Se]')

            reactant_graph = pysmiles.read_smiles(reactant_smiles, zero_order_bonds=False)
            product_graph = pysmiles.read_smiles(product_smiles, zero_order_bonds=False)
            
            rea_id_list = []
            pro_id_list = []
            for molecular in reactant_smiles.split('.'):
                rea_id_list.append(molecular_num_dict[molecular][0])
            for molecular in product_smiles.split('.'):
                pro_id_list.append(molecular_num_dict[molecular][0])
            temp_list_rea = []
            if len(rea_id_list) < max_rea_num:
                rea_id_list.append(random.choice(rea_id_list))
            else:
                for i in range(max_rea_num):
                    temp_list_rea.append(random.choice(rea_id_list))
                rea_id_list = temp_list_rea
            temp_list_pro = []
            if len(pro_id_list) < max_pro_num:
                pro_id_list.append(random.choice(pro_id_list))
            else:
                for i in range(max_pro_num):
                    temp_list_pro.append(random.choice(pro_id_list))
                pro_id_list = temp_list_pro

            all_id_list.append([rea_id_list, pro_id_list])

            if mode == 'train':
                for graph in [reactant_graph, product_graph]:
                    for attr in attribute_names:
                        for _, value in graph.nodes(data=attr):
                            all_values[attr].add(value)

            graphs.append([reactant_graph, product_graph])
    if mode == 'train':
        return all_values, graphs, all_id_list
    else:
        return graphs, all_id_list



def get_feature_encoder(all_values):  
    feature_encoder = {}
    idx = 0
    for key, values in all_values.items():
        feature_encoder[key] = {}
        for value in values:
            feature_encoder[key][value] = idx
            idx += 1
        feature_encoder[key]['unknown'] = idx
        idx += 1
    return feature_encoder


def preprocess(dataset, molecular_num_dict, max_rea_num, max_pro_num):  
    all_values, train_graphs, train_id_list = read_data(dataset, 'train', molecular_num_dict, max_rea_num, max_pro_num) 
    valid_graphs, valid_id_list = read_data(dataset, 'valid', molecular_num_dict, max_rea_num, max_pro_num)
    test_graphs, test_id_list = read_data(dataset, 'test', molecular_num_dict, max_rea_num, max_pro_num)
    feature_encoder = get_feature_encoder(all_values) 

    return feature_encoder, train_graphs, valid_graphs, test_graphs, train_id_list, valid_id_list, test_id_list


def load_data(args):  
    init_mfea = np.eye(40)
    init_rfea = np.eye(4)
    molecular_num_dict, max_rea_num, max_pro_num, all_fea_np = get_id_feature_dic('../data/' + args.dataset + '/all.txt', args.dataset, init_mfea)
    neighbor_info = load_graph('../data/' + args.dataset + '/adj.txt', args.num_neighbor, len(molecular_num_dict))
    feature_encoder, train_graphs, valid_graphs, test_graphs, train_id_list, valid_id_list, test_id_list = preprocess(args.dataset, molecular_num_dict, max_rea_num, max_pro_num)  
    train_dataset = SmilesDataset(args, 'train', feature_encoder, train_graphs, train_id_list) 
    valid_dataset = SmilesDataset(args, 'valid', feature_encoder, valid_graphs, valid_id_list)
    test_dataset = SmilesDataset(args, 'test', feature_encoder, test_graphs, test_id_list)
    return feature_encoder, train_dataset, valid_dataset, test_dataset, all_fea_np, init_rfea, neighbor_info


def load_graph(file, neighbor_num, num_node):
    graph_np = np.loadtxt(file, dtype=np.int32)
    graph_dict = dict()
    for link in graph_np:
        source = link[0]
        target = link[1]
        relati = link[2]
        if source not in graph_dict:
            graph_dict[source] = [0, [], []]
        graph_dict[source][0] += 1
        graph_dict[source][1].append(target)
        graph_dict[source][2].append(relati)
    neighbor_matrix = np.zeros([num_node, neighbor_num], dtype=np.int32)
    relation_matrix = np.zeros([num_node, neighbor_num], dtype=np.int32)
    for node_id in range(num_node):
        if node_id in graph_dict:
            neighbor_info = graph_dict[node_id]
        else:
            neighbor_info = [1, [int(node_id)], [int(3)]]
        num_neighbor = neighbor_info[0]
        pos_neighbor = neighbor_info[1]
        pos_relation = neighbor_info[2]
        if num_neighbor >= neighbor_num:
            sampled_indices = np.random.choice(list(range(num_neighbor)), size=neighbor_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(num_neighbor)), size=neighbor_num, replace=True)
        neighbor_matrix[node_id] = np.array([pos_neighbor[i] for i in sampled_indices])
        relation_matrix[node_id] = np.array([pos_relation[i] for i in sampled_indices])
    return neighbor_matrix, relation_matrix


def get_id_feature_dic(path, dataset, init_mfea):
    molecular_num_dict = dict()
    id_num = 0
    all_fea_list = []
    max_rea_num = 0
    max_pro_num = 0
    with open(path) as f:
        for line in f.readlines():
            idx, reactant_smiles, product_smiles = line.strip().split('\t')

            if '[se]' in reactant_smiles:
                reactant_smiles = reactant_smiles.replace('[se]', '[Se]')
            if '[se]' in product_smiles:
                product_smiles = product_smiles.replace('[se]', '[Se]')
            
            product_moleculars = []  
            for molecular in product_smiles.split('.'):
                product_moleculars.append(molecular)

            reactant_moleculars = []  
            for molecular in reactant_smiles.split('.'):
                reactant_moleculars.append(molecular)
            
            if len(reactant_smiles.split('.')) > max_rea_num:
                max_rea_num = len(reactant_smiles.split('.'))
            if len(product_smiles.split('.')) > max_pro_num:
                max_pro_num = len(product_smiles.split('.'))

            for molecular in reactant_moleculars:
                if molecular not in molecular_num_dict:
                    molecular_num_dict[molecular] = []
                    molecular_num_dict[molecular].append(id_num)
                    if len(get_func(molecular, dataset)) > 1:
                        func_list = get_func(molecular, dataset)
                        all_fea_list.append(np.sum(init_mfea[func_list], axis=0))
                        molecular_num_dict[molecular].append(func_list)
                    else:
                        all_fea_list.append(init_mfea[39])
                        molecular_num_dict[molecular].append([39])
                    id_num = id_num + 1

            for molecular in product_moleculars:
                if molecular not in molecular_num_dict:
                    molecular_num_dict[molecular] = []
                    molecular_num_dict[molecular].append(id_num)
                    if len(get_func(molecular, dataset)) > 1:
                        func_list = get_func(molecular, dataset)
                        all_fea_list.append(np.sum(init_mfea[func_list], axis=0))
                        molecular_num_dict[molecular].append(func_list)
                    else:
                        all_fea_list.append(init_mfea[39])
                        molecular_num_dict[molecular].append([39])
                    id_num = id_num + 1
    all_fea_np = np.array(all_fea_list)
    return molecular_num_dict, max_rea_num, max_pro_num, all_fea_np


def get_func(str_smile, dataset):

    fName = os.path.join('../data/' + dataset + '/FunctionalGroups.txt')   
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)   
    m_smile = Chem.MolFromSmiles(str_smile)  
    fcat = FragmentCatalog.FragCatalog(fparams) 
    fcgen = FragmentCatalog.FragCatGenerator()   
    fcgen.AddFragsFromMol(m_smile, fcat) 
    fragment_num = fcat.GetNumEntries()  

    func_list = [] 
    for i in range(fragment_num):
        func_id_list = list(fcat.GetEntryFuncGroupIds(i)) 
        if len(func_id_list)>0:
            for temp_func in func_id_list:
                if temp_func not in func_list:
                    func_list.append(temp_func)
    func_list.sort()
    return func_list


