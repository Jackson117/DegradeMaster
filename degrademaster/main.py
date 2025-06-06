import sys
import numpy as np
import copy
import torch
import os
import pickle
import logging
import json
from pathlib import  Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from degrademaster.protacloader import PROTACSet, collater
from degrademaster.model import GraphConv, ProtacModel, SageConv, GATTConv, EGNNConv
from degrademaster.train_and_test import train, valids
from degrademaster.prepare_data import construct_test_json
from degrademaster.config.config import get_args
import time
import warnings
warnings.filterwarnings("ignore")

import sys
from degrademaster.prepare_data import GraphData

from degrademaster.utils.pseudo_utils import split_dataset
from degrademaster.nn_utils import load_model_pretrained, setup_seed

TRAIN_NAME = "test"
root = "data/PROTAC"

args = get_args()
setup_seed(args.seed)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'PROTAC', f'{args.dataset}.csv')
    name_dic = construct_test_json(data_path, args.dataset)

    protac_graphs = GraphData('protac', root=root,
                             select_pocket_war=args.select_pocket_war, select_pocket_e3=args.select_pocket_e3,
                               conv_name=args.conv_name)
    ligase_pocket = GraphData("ligase_pocket", root,
                              select_pocket_war=args.select_pocket_war, select_pocket_e3=args.select_pocket_e3,
                               conv_name=args.conv_name)
    target_pocket = GraphData("target_pocket", root,
                              select_pocket_war=args.select_pocket_war, select_pocket_e3=args.select_pocket_e3,
                               conv_name=args.conv_name)





    name_list = list(name_dic.keys())


    label = torch.load(os.path.join(target_pocket.processed_dir, "label.pt"))
    feature = torch.load(os.path.join(target_pocket.processed_dir, "feature.pt"))
    if not args.feature:
        feature = np.random.rand(feature.shape[0], feature.shape[1])

    protac_set = PROTACSet(
        name_list,
        protac_graphs,
        ligase_pocket,
        target_pocket,
        feature,
        label,
    )
    data_size = len(protac_set)
    train_size = int(data_size * args.train_rate)
    test_size = data_size - train_size
    pos_num, neg_num = 0, 0
    for key, value in name_dic.items():
        if value['label'] == 0:
            neg_num += 1
        elif value['label'] == 1:
            pos_num += 1
    logging.info(f"all data: {data_size}")
    logging.info(f"train data: {train_size}")
    logging.info(f"test data: {test_size}")
    logging.info(f"positive label number: {pos_num}")
    logging.info(f"negative label number: {neg_num}")

    # train_dataset = torch.utils.data.Subset(protac_set, train_indicies)
    # test_dataset = torch.utils.data.Subset(protac_set, test_indicies)

    # trainloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collater,drop_last=False, shuffle=False)
    # testloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collater,drop_last=False, shuffle=False)
    testloader = DataLoader(protac_set, batch_size=args.batch_size, collate_fn=collater, drop_last=False,
                            shuffle=False)

    if args.conv_name == "GCN":

        ligase_pocket_model = GraphConv(num_embeddings=118, graph_dim=args.e3_dim, hidden_size=args.hidden_size)
        target_pocket_model = GraphConv(num_embeddings=118, graph_dim=args.tar_dim, hidden_size=args.hidden_size)
        protac_model = GraphConv(num_embeddings=118, graph_dim=args.protac_dim, hidden_size=args.hidden_size)
    elif args.conv_name == "GAT":
        ligase_pocket_model = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
        target_pocket_model = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
        protac_model = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)

    elif args.conv_name == "EGNN":
        ligase_pocket_model = EGNNConv(num_embeddings=118, in_node_nf=1, in_edge_nf=1,graph_nf=args.e3_dim, hidden_nf=args.hidden_size,
                                       n_layers=args.n_layers, node_attr=0, attention=args.attention)
        protac_model = EGNNConv(num_embeddings=118, in_node_nf=1, in_edge_nf=1, graph_nf=args.protac_dim, hidden_nf=args.hidden_size,
                                       n_layers=args.n_layers, node_attr=0, attention=args.attention)
        target_pocket_model = EGNNConv(num_embeddings=118, in_node_nf=1, in_edge_nf=1, graph_nf=args.tar_dim, hidden_nf=args.hidden_size,
                                       n_layers=args.n_layers, node_attr=0, attention=args.attention)
    else:
        raise ValueError("conv_type Error")
    model = ProtacModel(
        protac_model,
        ligase_pocket_model,
        target_pocket_model,
        args.hidden_size,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(f'runs/{TRAIN_NAME}')

    if args.mode == 'Train':
        model = train(
            model,
            train_loader=trainloader,
            valid_loader=testloader,
            device=device,
            writer=writer,
            LOSS_NAME=TRAIN_NAME,
            args=args
        )
    y_pred_l, y_score_l = [], []
    load_model_pretrained(model)
    for i in range(10):
        y_pred, y_score = valids(model.to(device),
                                test_loader=testloader,
                                device=device)
        y_pred_l.append(y_pred)
        y_score_l.append(y_score)

    y_pred = np.array(y_pred_l).mean(axis=0)
    y_score = np.array(y_score_l).mean(axis=0)

    test_end = time.time()
    print("------------------- Final Test -------------------")
    print('Base model: ', args.conv_name)
    print('Train rate: ', args.train_rate)
    print('Dataset: ', args.dataset)

    i = 0
    for key, value in name_dic.items():
        print(f'Compound ID {value['pro_comp_id']}')
        print('prediction: {}'.format(y_pred[i]))
        print('Predicted probability: {:.4f}'.format(y_score[i]))
        # print('label: {}'.format(output_dic_test['label'][i]))
        print('\n')
        i += 1

def entry_point():
    Path('log').mkdir(exist_ok=True)
    Path('model').mkdir(exist_ok=True)
    main()