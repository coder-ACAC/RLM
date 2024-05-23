import os
import argparse
import data_processing
import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='', help='dataset name')
    parser.add_argument('--epoch', type=int, default=3, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size') 
    parser.add_argument('--gnn', type=str, default='gcn', help='name of the GNN model')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers in molecular graph')
    parser.add_argument('--dim_atom', type=int, default=16, help='dimension of output embeddings from molecular graph') 
    parser.add_argument('--num_neighbor', type=int, default=2, help='number of sampling neighbors')
    parser.add_argument('--num_func', type=int, default=40, help='number of functional groups')
    parser.add_argument('--dim_graph', type=int, default=16, help='dimension of output embeddings from reaction-aware graph') 
    parser.add_argument('--tau', type=float, default=0.1, help='temperature in CL task.')
    parser.add_argument('--factor', type=float, default=0.0001, help='balancing factor.')
    parser.add_argument('--RG_agg', type=str, default='att', help='RG aggregation')
    parser.add_argument('--graph_agg', type=str, default='linear', help='aggregation in reaction-aware graph')
    parser.add_argument('--margin', type=float, default=4.0, help='margin in contrastive loss')
    parser.add_argument('--wei_delay', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_model', type=bool, default=True, help='save the trained model to disk')
    args = parser.parse_args()

    data = data_processing.load_data(args)
    train.train(args, data)
    
if __name__ == '__main__':
    main()
