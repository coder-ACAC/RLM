import os
import torch
import pickle
import data_processing
import numpy as np
from model import GNN, MNet
from copy import deepcopy
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
np.set_printoptions(threshold=np.inf)


def train(args, data):
    feature_encoder, train_data, valid_data, test_data, all_fea_np, init_rfea, neighbor_info = data
    feature_len = sum([len(feature_encoder[key]) for key in data_processing.attribute_names])
    all_fea_th = torch.from_numpy(all_fea_np).long()
    all_rel_th = torch.from_numpy(init_rfea).long()
    model = GNN(args, feature_len, all_fea_th, all_rel_th, neighbor_info)
    MemoryNet = MNet(args)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr}, {'params': MemoryNet.parameters(), 'lr': args.lr}], weight_decay=args.wei_delay)
    train_dataloader = GraphDataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if torch.cuda.is_available():
        model = model.cuda(args.gpu)
    best_model_params = None
    best_val_mrr = 0

    for i in range(args.epoch):
        model.train()
        epoch_loss = 0
        for reactant_graphs, product_graphs, id_list in train_dataloader:
            reactant_embeddings, reactant_a_embeddings, reactant_r_embeddings = model(reactant_graphs, id_list[0], id_list[1])
            product_embeddings,  product_a_embeddings,  product_r_embeddings = model(product_graphs, id_list[1], id_list[0])
            
            loss_e = equivalence_cl_loss(reactant_embeddings, product_embeddings, MemoryNet, args)
            loss_c1 = cross_cl_loss(reactant_a_embeddings, reactant_r_embeddings, args)
            loss_c2 = cross_cl_loss(product_a_embeddings, product_r_embeddings, args)
            loss = loss_e + args.factor * (loss_c1 + loss_c2)

            batch_loss = loss.item()
            epoch_loss = epoch_loss + batch_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        val_mrr = evaluate(model, MemoryNet, 'valid', valid_data, args)
        _ = evaluate(model, MemoryNet, 'test', test_data, args)
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_model_params = deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    best_mrr = evaluate(model, MemoryNet, 'test', test_data, args)
    print("best_mrr:", best_mrr)


def cross_cl_loss(a_embeddings, r_embeddings, args):
    a_embeddings = F.normalize(a_embeddings, dim=1)
    r_embeddings = F.normalize(r_embeddings, dim=1)

    pos_cl_ratings = torch.sum(torch.mul(a_embeddings, r_embeddings), dim=-1)
    tot_cl_ratings = torch.matmul(a_embeddings, torch.transpose(r_embeddings, 0, 1))

    pos_cl_ratings = torch.exp(pos_cl_ratings / args.tau)
    tot_cl_ratings = torch.sum(torch.exp(tot_cl_ratings / args.tau), dim=1)

    loss = -torch.sum(torch.log(pos_cl_ratings / tot_cl_ratings))

    return loss


def equivalence_cl_loss(reactant_embeddings, product_embeddings, MemoryNet, args):

    dim = args.dim_atom + args.dim_graph 
    batch_size = 128  

    head_embedding = reactant_embeddings
    tail_embedding = product_embeddings
    distance_matrix = torch.zeros(args.batch_size, args.batch_size)
    for i in range(0, head_embedding.size(0), batch_size):
        head_batch = head_embedding[i:i + batch_size]
        relation_matrix = process_batch(head_batch, tail_embedding, MemoryNet, dim)
        for j in range(relation_matrix.size(0)):
            distance_matrix[i + j] = torch.norm(head_batch[j].unsqueeze(0) + relation_matrix[j] - tail_embedding, dim=1)
    
    dist = distance_matrix
    pos = torch.diag(dist)
    mask = torch.eye(args.batch_size)
    if torch.cuda.is_available():
        mask = mask.cuda(args.gpu)
    neg = (1 - mask) * dist + mask * args.margin
    neg = torch.relu(args.margin - neg)
    loss = torch.mean(pos) + torch.sum(neg) / args.batch_size / (args.batch_size - 1)

    return loss


def process_batch(head_batch, tail_embedding, model, dim):
    head_expanded = head_batch.unsqueeze(1).expand(-1, tail_embedding.size(0), -1).reshape(-1, dim)
    tail_expanded = tail_embedding.unsqueeze(0).expand(head_batch.size(0), -1, -1).reshape(-1, dim)
    relation_vecs = model(head_expanded, tail_expanded)
    return relation_vecs.view(head_batch.size(0), tail_embedding.size(0), -1)


def evaluate(model, MemoryNet, mode, data, args):
    model.eval()
    with torch.no_grad():
        all_product_embeddings = []
        product_dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        for _, product_graphs, id_list in product_dataloader:
            product_embeddings, _, _ = model(product_graphs, id_list[1], id_list[0])
            all_product_embeddings.append(product_embeddings)
        all_product_embeddings = torch.cat(all_product_embeddings, dim=0)

        all_rankings = []
        reactant_dataloader = GraphDataLoader(data, batch_size=1)
        i = 0
        for reactant_graphs, product_graphs, id_list in reactant_dataloader:
            reactant_embeddings, _, _ = model(reactant_graphs, id_list[0], id_list[1])
            repeat_reactant_embeddings = reactant_embeddings.repeat(len(all_product_embeddings), 1)
            relation_vec = MemoryNet(repeat_reactant_embeddings, all_product_embeddings)  
            dist = torch.sqrt( torch.sum((repeat_reactant_embeddings + relation_vec-all_product_embeddings)**2, dim=1) )
            rank_list = torch.argsort(dist, dim=0)
            rank_num = np.where(rank_list == i)[0][0] + 1
            all_rankings.append(rank_num)
            i = i + 1

        all_rankings = np.array(all_rankings)
        mrr = float(np.mean(1 / all_rankings))
        h1 = float(np.mean(all_rankings <= 1))
        return mrr