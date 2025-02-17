# Simple yet Effective Node Property Prediction on Edge Streams under Distribution Shifts

## Overview

This is the source code for *Simple yet Effective Node Property Prediction on Edge Streams under Distribution Shifts*
We utilize the [TGL](https://github.com/amazon-science/tgl) framework for the implementation.


## Requirements
- python >= 3.6.13
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511

Our temporal sampler is implemented using C++, please compile the sampler first with the following command. Additionaly, we implement a degreee function in the sampler_core.cpp.
> python utils/setup.py build_ext --inplace


## Dataset

We utlize total 9 datasets (WIKI, REDDIT, MOOC, Email-EU, GDLET_sample, Synthetic-50, Synthetic-70, Synthetic-90, TGBN-trade, TGBN-genre).
Due to storage limitations, we currently have only uploaded the Email-EU and tgbn-trade datasets. 
The total dataset can be downloaded from the following [link](https://drive.google.com/file/d/1NUPGRRqnTybOqaoDqQEWFpnkdl7vb04K/view?usp=drive_link).

We created folders named after each dataset in `\DATA\` and added the following files to each dataset folder according to the TGL method.

1. `edges.csv`: The file that stores temporal edge informations. The csv should have the following columns with the header as `,src,dst,time,ext_roll` where each of the column refers to edge index (start with zero), source node index (start with zero), destination node index, time stamp, extrapolation roll (0 for training edges, 1 for validation edges, 2 for test edges). The CSV should be sorted by time ascendingly.
2. `ext_full.npz`: The T-CSR representation of the temporal graph. We provide a script to generate this file from `edges.csv`. You can use the following command to use the script 
    >python utils/gen_graph.py --data \<NameOfYourDataset>
3. `edge_features.pt` (optional): The torch tensor that stores the edge featrues row-wise with shape (num edges, dim edge features).
4. `node_features.pt` (optional): The torch tensor that stores the node featrues row-wise with shape (num nodes, dim node features). 
5. `labels.csv` (optional): The file contains node labels for dynamic node classification task. The csv should have the following columns with the header as `,node,time,label,ext_roll` where each of the column refers to node label index (start with zero), node index (start with zero), time stamp, node label, extrapolation roll (0 for training node labels, 1 for validation node labels, 2 for test node labels). The CSV should be sorted by time ascendingly.

Additionally, for the TGB datasets, data was loaded and evaluated using the separate TGB library. Only edges.csv files for sampling are stored in the respective TGBN dataset folder. (We also implemented separate feature selection and training code for TGN datasets, as TGB follows a different evaluation step.)

## Configuration Files

We provide a configuration file of our proposed model, SLIM, with four TGNN baselines (JODIE, DySAT, TGAT, and TGN).
The configuration files are located at `/config/`.


## Node Property Prediction

Overall, node property prediction consists of the proper feature selection process and model training and evaluation. In the `main.py`, proper feature selection is first performed for each dataset, followed by the overall process where the SLIM model is trained and evaluated based on the selected node features. Through the proper feature selection process, the selected node feature augmentation process is recorded as a JSON file in the `selected_augmentation_process` folder. (If the corresponding JSON file already exists, the selected features are used directly.) At this stage, evaluation is conducted for each dataset according to the corresponding subtask and evaluation metric.
- Dynamic Anomaly detection (AUC): WIKI, REDDIT, MOOC
- Dynamic Node Classification (F1 Score): Email-EU, GDELT_sample, Synthetic-70, Synthetic-70, Synthetic-90  
- Node Affinity Prediction (NDCG@10): TGBN-trade, TGBN-genre

### Run Main Experiments
We can utlize following datasets for <NameOfDATA>: WIKI, REDDIT, MOOC, Email-EU, GDLET_sample, Synthetic-50, Synthetic-70, Synthetic-90, TGBN-trade, TGBN-genre

```{bash}
python main.py --data <NameOfDATA> --config config/SLIM.yml

```

### Run Experiments according to unseen ratios
We run the below codes for REDDIT, Email-EU, and TGBN-trade datasets.

```{bash}
python main.py --data <NameOfDATA> --config config/SLIM.yml --train_ratio 0.1 --val_ratio 0.2
python main.py --data <NameOfDATA> --config config/SLIM.yml --train_ratio 0.3 --val_ratio 0.4
python main.py --data <NameOfDATA> --config config/SLIM.yml --train_ratio 0.5 --val_ratio 0.6
python main.py --data <NameOfDATA> --config config/SLIM.yml --train_ratio 0.7 --val_ratio 0.8

```

### Run Proper Feature Selection
We run seperate propery feature selection code for TGBN datasets and other datasets.
```{bash}
python feature_selection_TGB.py --data <NameOfTGBDATA> --config config/SLIM.yml
python feature_selection.py --data <NameOfOTHERDATA> --config config/SLIM.yml

```
### Run Model Training and Evalution
We run seperate model training code for TGBN datasets and other datasets. We can use random, zero, random_mix, node2vec_mix, and structural for the <SelectedFeature>. If no input is provided for the selected feature, augmented node features will not be used.
```{bash}
python train_node_property_prediction_TGB.py --data <NameOfTGBDATA> --config config/SLIM.yml --selected_feature <SelectedFeature>
python train_node_property_prediction.py --data <NameOfOTHERDATA> --config config/SLIM.yml --selected_feature <SelectedFeature>

```


### Run Other Baselines
We can run other baselines (JODIE, DySAT, TGAT, TGN) that are implemented in the TGL framework by the below codes.
```{bash}
python train_node_property_prediction_TGB.py --data <NameOfTGBDATA> --config config/<OtherBaselines>.yml
python train_node_property_prediction.py --data <NameOfOTHERDATA> --config config/<OtherBaselines>.yml
python train_node_property_prediction_TGB.py --data <NameOfTGBDATA> --config config/<OtherBaselines>.yml --selcted_feature random
python train_node_property_prediction.py --data <NameOfOTHERDATA> --config config/<OtherBaselines>.yml --selcted_feature random

```