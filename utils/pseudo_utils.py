import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import random
import json
import pickle as pkl

class EntropyLoss(nn.Module):
    '''
    return: mean entropy of the given batch if reduction is True, n-dim vector of entropy if reduction is False.
    '''
    def __init__(self, reduction=True):
        super(EntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if self.reduction:
            b = -1.0 * b.sum()
            b = b / x.shape[0]
        else:
            b = -1.0 * b.sum(axis=1)

        return b

def cosine_similarity(x1, x2, reduction=True):
    '''
    compute cosine similarity between x1 and x2.
    :param x1: N * D tensor or 1d tensor.
    :param x2: N * D tensor or 1d tensor.
    :return: a scalar tensor if reduction is True, a tensor of shape N if reduction is False.
    '''
    cos_sim = nn.CosineSimilarity(dim=-1)
    if reduction:
        sim = cos_sim(x1, x2).mean()
    else:
        sim = cos_sim(x1, x2)

    return sim


class CE_uniform(nn.Module):
    '''
    return: CE of the given batch if reduction is True, n-dim vector of CE if reduction is False.
    '''
    def __init__(self, n_id_classes, reduction=True):
        super(CE_uniform, self).__init__()
        self.reduction = reduction
        self.n_id_classes = n_id_classes

    def forward(self, x):
        b = (1/self.n_id_classes) * F.log_softmax(x, dim=1)
        if self.reduction:
            b = -1.0 * b.sum()
            b = b / x.shape[0]
        else:
            b = -1.0 * b.sum(axis=1)

        return b


def get_consistent_loss_new(x1, x2, f1=None, f2=None):
    '''
    compute consistent loss between attention scores and output entropy.
    :param x1: ood score matrix, H * N tensor. the larger, the more likely to be ood.
    :param x2: entropy vector, N-dim tensor.
    :return: scalar tensor of computed loss.
    '''
    x1 = x1.mean(axis=0)
    if f1 is not None:
        x1 = f1(x1)
    if f2 is not None:
        x2 = f2(x2)
    loss = cosine_similarity(x1, x2)

    return -1.0 * loss

def local_ent_loss(logits, att, n_id_classes, m=0.5):
    att_norm = F.sigmoid(att.mean(axis=1)).detach()  # n-dim
    mask = torch.ge(att_norm - m, 0)
    ce_uni = CE_uniform(n_id_classes, reduction=False)
    ce = ce_uni(logits)  # N-dim
    if mask.sum() > 0:
        loss = ce[mask].mean()
    else:
        loss = 0

    return loss

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    return


# def split_dataset(json_path, train_ratio):
#     with open(json_path, 'r') as f:
#         name_dict = json.load(f)
#
#     num_label = 0
#     for val in name_dict.values():
#         if val['label'] != -1:
#             num_label += 1
#     test_p_size = int(num_label * (1 - train_ratio)/2)
#     test_n_size = test_p_size
#     train_indices, test_indices = [], []
#     labels_l = []
#     test_p_cnt, test_n_cnt = 0, 0
#     random.seed(111)    # 112
#     val = list(name_dict.values())
#     indx = list(np.arange(len(val)))
#     random.shuffle(indx)
#     for idx in indx:
#         value = val[idx]
#         if value['label'] == 1 and test_p_cnt < test_p_size:
#             test_indices.append(int(idx))
#             test_p_cnt += 1
#             labels_l.append(value['label'])
#         elif value['label'] == 0 and test_n_cnt < test_n_size:
#             test_indices.append(int(idx))
#             test_n_cnt += 1
#             labels_l.append(value['label'])
#         else:
#             train_indices.append(int(idx))
#
#     keys = np.array(list(name_dict.keys()))
#     dic = {'indices': test_indices, 'labels': labels_l, 'id': keys[test_indices]}
#     with open('./latent/test_indices.pkl', 'wb') as f:
#         pkl.dump(dic, f)
#
#     return train_indices, test_indices

def split_dataset(json_path, train_ratio):
    with open(json_path, 'r') as f:
        name_dict = json.load(f)

    num_label = 0
    for val in name_dict.values():
        if val['label'] != -1:
            num_label += 1
    test_size = int(num_label * (1 - train_ratio))
    # test_n_size = test_p_size
    train_indices, test_indices = [], []
    labels_l = []
    test_cnt = 0
    random.seed(111)    # 112
    val = list(name_dict.values())
    indx = list(np.arange(len(val)))
    random.shuffle(indx)


    for idx in indx:
        value = val[idx]
        if value['label'] != -1 and test_cnt < test_size:
            test_indices.append(int(idx))
            test_cnt += 1
        else:
            train_indices.append(int(idx))

    keys = np.array(list(name_dict.keys()))
    dic = {'indices': test_indices, 'labels': labels_l, 'id': keys[test_indices]}
    with open('./latent/test_indices.pkl', 'wb') as f:
        pkl.dump(dic, f)

    return train_indices, test_indices