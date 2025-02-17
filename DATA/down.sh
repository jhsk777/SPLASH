#!/bin/bash


wget -P ./DATA/MOOC https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/edges.csv   
wget -P ./DATA/MOOC https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/ext_full.npz
wget -P ./DATA/MOOC https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/int_full.npz
wget -P ./DATA/MOOC https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/int_train.npz
wget -P ./DATA/REDDIT_t https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edge_features.pt
wget -P ./DATA/REDDIT_t https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edges.csv
wget -P ./DATA/REDDIT_t https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/ext_full.npz
wget -P ./DATA/REDDIT_t https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/int_full.npz
wget -P ./DATA/REDDIT_t https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/int_train.npz
wget -P ./DATA/REDDIT_t https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/labels.csv
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edge_features.pt
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edges.csv  
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/ext_full.npz
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/int_full.npz
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/int_train.npz
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/labels.csv