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
from protacloader import PROTACSet, collater
from model import GraphConv, ProtacModel, SageConv, GATTConv, EGNNConv
from train_and_test import train, valids
from config.config import get_args
import time

import sys
from prepare_data import GraphData
from dataset import VOCAB
from utils.pseudo_utils import split_dataset
from nn_utils import load_model, setup_seed

TRAIN_NAME = "test"
root = "data/PROTAC"

args = get_args()
setup_seed(args.seed)

def main():
    protac_graphs = GraphData('protac', root=root,
                             select_pocket_war=args.select_pocket_war, select_pocket_e3=args.select_pocket_e3,
                               conv_name=args.conv_name)
    ligase_pocket = GraphData("ligase_pocket", root,
                              select_pocket_war=args.select_pocket_war, select_pocket_e3=args.select_pocket_e3,
                               conv_name=args.conv_name)
    target_pocket = GraphData("target_pocket", root,
                              select_pocket_war=args.select_pocket_war, select_pocket_e3=args.select_pocket_e3,
                               conv_name=args.conv_name)

    with open(os.path.join(root, '{}.json'.format(args.dataset_type)), 'r') as f:
        name_dic = json.load(f)

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
    train_indicies, test_indicies = split_dataset(os.path.join(root, '{}.json'.format(args.dataset_type)), args.train_rate)

    train_dataset = torch.utils.data.Subset(protac_set, train_indicies)
    test_dataset = torch.utils.data.Subset(protac_set, test_indicies)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collater,drop_last=False, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collater,drop_last=False, shuffle=False)

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
    loss_l, acc_l, auc_l, f1_l, pre_l, rec_l = [], [], [], [], [], []
    load_model(model, args, loaded_epoch=args.epoch)
    for i in range(10):
        loss, acc, auc, f1, pre, rec, latent_val = valids(model.to(device),
                                test_loader=testloader,
                                device=device)
        saved_dict = {'latent_emb': latent_val}
        np.save('./latent/latent_ours_{}.npy'.format(args.hidden_size), saved_dict)

        print('Test Loss: {} | Accuracy: {} | AUC: {} | F1: {} | PRE: {} | REC: {}'.format(loss, acc, auc, f1, pre, rec))
        loss_l.append(loss)
        acc_l.append(acc)
        auc_l.append(auc)
        f1_l.append(f1)
        pre_l.append(pre)
        rec_l.append(rec)
    test_end = time.time()
    print("------------------- Final Test -------------------")
    print('Base model: ', args.conv_name)
    print('Train rate: ', args.train_rate)
    print('Dataset_type: ', args.dataset_type)
    print('Loss in infer: ', np.mean(loss_l))
    print('Accuracy in infer: ', np.mean(acc_l))
    print('AUC in infer: ', np.mean(auc_l))
    print('F1 in infer: ', np.mean(f1_l))
    print('PRE in infer: ', np.mean(pre_l))
    print('REC in infer: ', np.mean(rec_l))

if __name__ == "__main__":
    Path('log').mkdir(exist_ok=True)
    Path('model').mkdir(exist_ok=True)
    main()