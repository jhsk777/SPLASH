import argparse
import torch
import os


def get_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='dataset name')
    parser.add_argument('--config', type=str, default='', help='path to config file')
    parser.add_argument('--batch_size', type=int, default=600)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--n_degree', default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--split_num', type=int, default=5)

    parser.add_argument('--set_degree', default=False, action='store_true', help='set maximum degree')
    parser.add_argument('--skip_alpha', type=float, default=1)
    parser.add_argument('--early_stop', type=int, default=10)
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')

    parser.add_argument('--train_ratio', type=float, default=0.1, help='train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation ratio')

    args = parser.parse_args()

    if args.data in ['WIKI', 'REDDIT', 'MOOC']:
        args.task = 'anomaly'
    elif args.data in ['Email-EU', 'GDELT_sample', 'synthetic-50', 'synthetic-70', 'synthetic-90']:
        args.task = 'class'
    elif args.data in ['tgbn-trade', 'tgbn-genre']:
        args.task = 'affinity'

    if args.data == 'MOOC':
        args.skip_alpha = 0
    else:
        args.skip_alpha = 1

    return args

    