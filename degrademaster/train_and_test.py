import numpy as np
import torch
import torch.nn as nn
from degrademaster.utils.pseudo_utils import EntropyLoss, get_consistent_loss_new, cosine_similarity, local_ent_loss
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from degrademaster.nn_utils import save_model

def valids(model, test_loader, device):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        y_true = []
        y_pred = []
        y_score = []
        latent_list = []
        loss = []
        iteration = 0
        for data_sample in test_loader:
            y = data_sample['label'].to(device)
            outputs, latent = model(
                data_sample['protac_graphs'].to(device),
                data_sample['ligase_pocket'].to(device),
                data_sample['target_pocket'].to(device),
                data_sample['feature'].to(device)
            )
            # labeled_mask = torch.where(y==-1, False, True)
            # outputs, y = outputs[labeled_mask], y[labeled_mask]
            # loss_val = criterion(outputs, y)
            # loss.append(loss_val.item())
            latent_list.append(latent)
            y_score = y_score + torch.nn.functional.softmax(outputs,1)[:,1].cpu().tolist()
            y_pred = y_pred + torch.max(outputs,1)[1].cpu().tolist()
            # y_true = y_true + y.cpu().tolist()
            iteration += 1

        model.train()
        # saved_dict = {'score': y_score, 'pred': y_pred}
        # np.save('./eval/PROTAC_demo_saved.npy', saved_dict)
        # latent = torch.cat(latent_list, 0)


    return y_pred, y_score


def train(model, args, train_loader=None, valid_loader=None, device=None, writer=None, LOSS_NAME=None):


    'Pre_train_gnn'
    def pre_train_gnn(model, optimizer):
        print("------------------- PreTrain -------------------")
        a = torch.tensor(0.9).to(device)
        b = torch.tensor(0.01).to(device)
        # train
        xent = nn.CrossEntropyLoss()
        ent_loss_func = EntropyLoss(reduction=False)

        count = 0
        loss_l, cons_loss_l, lcent_loss_l, sup_loss_l = [], [], [], []
        prediction = torch.zeros((len(train_loader.dataset), args.class_num)).to(device)

        for epoch in range(args.epoch_pre):
            pred_list, lbl_list, att_list, mask_list = [], [], [], []
            model.train()
            for data_sample in train_loader:
                outputs, latent = model(
                    data_sample['protac_graphs'].to(device),
                    data_sample['ligase_pocket'].to(device),
                    data_sample['target_pocket'].to(device),
                    data_sample['feature'].to(device)
                )

                y = data_sample['label'].to(device)
                labeled_mask = torch.where(y == -1, False, True)
                pred_list.append(outputs)
                lbl_list.append(y)
                mask_list.append(labeled_mask)


            loss = torch.zeros(1).to(device)
            logits = torch.cat(pred_list)
            labels = torch.cat(lbl_list)
            # att = torch.cat(att_list)
            att = None
            train_mask = torch.cat(mask_list)
            if args.start_epoch <= epoch <= args.end_epoch:
                pre = F.softmax(logits, 1)
                pre = pre ** (1 / args.T)
                pre = pre / pre.sum(dim=1, keepdim=True)
                prediction += pre
                count += 1
            if args.w_consistent is not None and args.w_consistent > 0:
                ent_loss = ent_loss_func(logits)  # ent_loss: N-dim tensor

                cos_loss_1 = get_consistent_loss_new(att.T, (ent_loss - ent_loss.mean()) / ent_loss.std(),
                                                     f1=F.sigmoid, f2=F.sigmoid)
                consistent_loss = 0.5 * (cos_loss_1)
                ac_cons_loss =  torch.pow(a, b * epoch) * args.w_consistent * consistent_loss
                loss += ac_cons_loss
                cons_loss_l.append(ac_cons_loss.item())
            if args.w_ent is not None and args.w_ent > 0:
                ac_lcent_loss = torch.pow(a, b * epoch) * args.w_ent * local_ent_loss(logits, att, args.class_num, args.margin)
                loss += ac_lcent_loss
                lcent_loss_l.append(ac_lcent_loss.item())

            sup_loss = xent(logits[train_mask], labels[train_mask])
            loss += sup_loss
            sup_loss_l.append(sup_loss.item())
            loss_l.append(loss.item())
            loss.backward()
            optimizer.step()

            # validate
            if epoch % 10 == 0:
                y_score = torch.nn.functional.softmax(logits[train_mask], 1)[:, 1].cpu().tolist()
                y_pred = torch.max(logits[train_mask], 1)[1].cpu().tolist()
                y_true = labels[train_mask].cpu().tolist()
                acc, auc, f1, pre, rec = accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_score), f1_score(y_true, y_pred, average='macro'), precision_score(y_true, y_pred), recall_score(y_true, y_pred)
                print('Epoch: {}/{}  '.format(epoch + 1, args.epoch_pre), end='')
                print(
                    'Train Loss: {} | Train acc: {}| Train auc: {}|Valid f1: {}| Train pre: {}| Train rec: {}'.format(loss.item(), acc, auc, f1, pre,
                                                                                                     rec))
                model.eval()
                val_loss, acc, auc, f1, pre, rec, _ = valids(model, valid_loader, device)
                print('Valid loss: {} | Valid acc: {}| Valid auc: {}|Valid f1: {}| Valid pre: {}| Valid rec: {}'.format(val_loss, acc, auc, f1, pre, rec))
                print('\n')

        np.save('./log/pretrain_loss.npy', {'loss': loss_l, 'cons_loss': cons_loss_l, 'lcent_loss': lcent_loss_l, 'sup_loss': sup_loss_l})
        prediction = prediction / count
        pred_labels = prediction.argmax(dim=1)
        pred_labels[train_mask] = labels[train_mask]
        pre_log = torch.log(prediction)
        pre_log = torch.where(torch.isinf(pre_log), torch.full_like(pre_log, 0), pre_log)
        pred_entropy = torch.sum(torch.mul(-prediction, pre_log), dim=1)

        return pred_labels, pred_entropy, train_mask

    def selftraining(pseudo_labels, entropy, train_mask):
        num_class = args.class_num
        num_k = args.num_k
        sorted_index = torch.argsort(entropy, dim=0, descending=False)
        index = []
        count = [0] * num_class
        for i in sorted_index:
            for j in range(num_class):
                if pseudo_labels[i] == j and count[j] < num_k and not train_mask[i]:
                    index.append(i.item())
                    count[j] += 1
        n_train_mask = torch.zeros(train_mask.shape, dtype=torch.bool).to(device)
        n_train_mask[train_mask] = True
        n_train_mask[index] = True

        return n_train_mask

    def train_oodgat(model, optimizer, pseudo_labels, train_mask):
        print("------------------- Last Train -------------------")
        a = torch.tensor(0.9).to(device)
        b = torch.tensor(0.01).to(device)
        # train
        xent = nn.CrossEntropyLoss()
        ent_loss_func = EntropyLoss(reduction=False)
        loss_l, cons_loss_l, lcent_loss_l, sup_loss_l, val_loss_l = [], [], [], [], []
        best_acc = 0
        for epoch in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            pred_list, lbl_list, att_list, mask_list, latent_list = [], [], [], [], []
            for data_sample in train_loader:
                outputs, latent = model(
                    data_sample['protac_graphs'].to(device),
                    data_sample['ligase_pocket'].to(device),
                    data_sample['target_pocket'].to(device),
                    data_sample['feature'].to(device)

                )

                y = data_sample['label'].to(device)
                labeled_mask = torch.where(y == -1, False, True)
                pred_list.append(outputs)
                lbl_list.append(y)
                mask_list.append(labeled_mask)
                latent_list.append(latent)

            loss = torch.zeros(1).to(device)
            logits = torch.cat(pred_list)
            labels = torch.cat(lbl_list)
            real_mask = torch.cat(mask_list)
            latent = torch.cat(latent_list, 0)

            att = None
            if args.w_consistent is not None and args.w_consistent > 0:
                ent_loss = ent_loss_func(logits)  # ent_loss: N-dim tensor
                cos_loss_1 = get_consistent_loss_new(att.T, (ent_loss - ent_loss.mean()) / ent_loss.std(),
                                                     f1=F.sigmoid, f2=F.sigmoid)
                consistent_loss = 0.5 * (cos_loss_1)
                ac_cons_loss = torch.pow(a, b * epoch) * args.w_consistent * consistent_loss
                loss += ac_cons_loss
                cons_loss_l.append(ac_cons_loss.item())
            if args.w_ent is not None and args.w_ent > 0:
                lcent_loss = torch.pow(a, b * epoch) * args.w_ent * local_ent_loss(logits, att, args.class_num, args.margin)
                loss += lcent_loss
                lcent_loss_l.append(lcent_loss.item())

            sup_loss = xent(logits[train_mask], pseudo_labels[train_mask])
            loss += sup_loss
            sup_loss_l.append(sup_loss.item())
            loss_l.append(loss.item())
            loss.backward()
            optimizer.step()
            # validate
            if epoch % 10 == 0:
                y_score = torch.nn.functional.softmax(logits[real_mask], 1)[:, 1].cpu().tolist()
                y_pred = torch.max(logits[real_mask], 1)[1].cpu().tolist()
                y_true = labels[real_mask].cpu().tolist()
                acc, auc, f1, pre, rec = accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_score), f1_score(
                    y_true, y_pred, average='macro'), precision_score(y_true, y_pred), recall_score(y_true, y_pred)
                print('Epoch: {}/{}  '.format(epoch + 1, args.epoch), end='')
                print(
                    'Train Loss: {} | Train acc: {}| Train auc: {}|Valid f1: {}| Train pre: {}| Train rec: {}'.format(loss.item(), acc, auc, f1, pre,
                                                                                                     rec))
                saved_dict = {'score': y_score, 'pred': y_pred, 'label': y_true}
                np.save('./eval/PROTAC_demo_train_saved.npy', saved_dict)
                model.eval()
                val_loss, acc, auc, f1, pre, rec, latent_val = valids(model, valid_loader, device)
                if best_acc < acc:
                    best_acc = acc
                    saved_dict = {'latent_emb': latent_val}
                    np.save('./latent/latent_{}.npy'.format(args.hidden_size), saved_dict)
                    save_model(epoch, model, args)
                val_loss_l.append(val_loss)
                print('Valid loss: {} | Valid acc: {}| Valid auc: {}|Valid f1: {}| Valid pre: {}| Valid rec: {}'.format(val_loss, acc, auc, f1, pre, rec))
                print('\n')

        return

    model = model.to(device)
    optmizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pre_labels, pre_entropy, train_mask = pre_train_gnn(model, optmizer)
    new_train_mask = selftraining(pre_labels, pre_entropy, train_mask)
    pseudo_ID_nodes = torch.sum(new_train_mask) - torch.sum(train_mask)
    pseudo_ID_rate = pseudo_ID_nodes / torch.sum(new_train_mask)
    for param_group in optmizer.param_groups:
        param_group['lr'] = args.lr_l
    train_oodgat(model, optmizer, pre_labels, new_train_mask)
    _, acc, auc, f1, pre, rec, _ = valids(model, valid_loader, device)
    print('Final validation')
    print('pseudo_ID_nodes: {}, pseudo_ID_rate: {}'.format(pseudo_ID_nodes, pseudo_ID_rate))
    print('Valid acc: {}| Valid auc: {}|Valid f1: {}| Valid pre: {}| Valid rec: {}'.format(acc, auc, f1, pre, rec))


    return model