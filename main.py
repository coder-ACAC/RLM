import os
import argparse
import data_processing
import train


def main():
    parser = argparse.ArgumentParser()
    # Use your own hyperparameter settings.
    parser.add_argument('--gpu', type=int, default=0, help='the index of gpu device')
    parser.add_argument('--task', type=str, default='pretrain', help='downstream task')
    parser.add_argument('--dataset', type=str, default='test_examples', help='dataset name')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size') 
    parser.add_argument('--gnn', type=str, default='gcn', help='name of the GNN model')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--dim_atom', type=int, default=32, help='dimension of molecule embeddings') 
    parser.add_argument('--num_neighbor', type=int, default='', help='number of sampling neighbors')
    parser.add_argument('--dim_graph', type=int, default=4, help='dimension of molecule embeddings') 
    parser.add_argument('--tau', type=float, default=1.0, help='cross CL task.')
    parser.add_argument('--factor', type=float, default='', help='balancing factor.')
    parser.add_argument('--RG_agg', type=str, default='', help='RG aggregation')
    parser.add_argument('--graph_agg', type=str, default='', help='name of the graph embedding aggregation')
    parser.add_argument('--margin', type=float, default=4.0, help='margin in contrastive loss')
    parser.add_argument('--wei_delay', type=float, default='', help='learning rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')


    args = parser.parse_args()
    data = data_processing.load_data(args)
    train.train(args, data)
    

if __name__ == '__main__':
    main()
