import fnmatch
import numpy
import pandas as pd
import torch.nn as nn
from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train(args, subgraph_size, negsamp_ratio):
    print('Dataset: ', args.dataset)
    if args.lr is None:
        if args.dataset == 'DBLP':
            args.lr = 1e-3
        elif args.dataset in ['ACM', 'Amazon']:
            args.lr = 5e-4

    if args.num_epoch is None:
        if args.dataset in ['DBLP', 'ACM']:
            args.num_epoch = 2000
        elif args.dataset in ['Amazon', 'YelpChi']:
            args.num_epoch = 1000

    if args.dataset == 'DBLP':
        subgraph_size = torch.tensor([3, 15, 15])
        negsamp_ratio = 1.8
    elif args.dataset == 'ACM':
        subgraph_size = torch.tensor([4, 25, 25])
        negsamp_ratio = 10
    elif args.dataset == 'Amazon':
        subgraph_size = torch.tensor([10, 30, 90])
        negsamp_ratio = 5
    elif args.dataset == 'YelpChi':
        subgraph_size = torch.tensor([20, 20, 20])
        negsamp_ratio = 2

    batch_size = args.batch_size
    view_size = args.view_size

    node_pos = torch.tensor([0], dtype=torch.int32)
    for i in range(len(subgraph_size)):
        node_pos = torch.cat([node_pos, torch.tensor([torch.sum(subgraph_size[:i]) + 1])])

    subgraph_view_size = torch.sum(subgraph_size) + 1

    # Set random seed
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    adj_list = []
    neighbor_list = []
    data_path = os.path.join(args.datadir, args.dataset)
    ano_label_list = []
    str_ano_label_list = []
    attr_ano_label_list = []
    for f_name in os.listdir(data_path):
        if fnmatch.fnmatch(f_name, '{}*.mat'.format(args.dataset)):

            adj, feat, labels, idx_train, idx_val, idx_test, ano_label_, str_ano_label_, attr_ano_label_ = load_mat(
                os.path.join(data_path, f_name))

            ano_label_list.append(ano_label_)
            str_ano_label_list.append(str_ano_label_)
            attr_ano_label_list.append(attr_ano_label_)
            adj = np.asarray(adj.todense())
            neighbors = []
            for i in range(adj.shape[0]):
                t_set = set(np.where(adj[i] == 1)[0])
                if i in t_set: t_set.remove(i)
                neighbor_ = torch.tensor(list(t_set))
                neighbor_ = torch.cat((torch.tensor([i]), neighbor_))
                neighbor_ = nn.functional.pad(neighbor_, (0, 100 - len(neighbor_)), value=-1)
                neighbor_ = neighbor_[:100].numpy().astype(int)
                neighbors.append(neighbor_)

            neighbor_list.append(np.array(neighbors))
            adj_list.append(adj)
            features = feat

    ano_label = numpy.zeros_like(ano_label_list[0])
    for ano_label_ in ano_label_list:
        ano_label = np.bitwise_or(ano_label, ano_label_)

    str_ano_label = numpy.zeros_like(ano_label_list[0])
    for str_ano_label_ in str_ano_label_list:
        str_ano_label = np.bitwise_or(str_ano_label, str_ano_label_)

    attr_ano_label = numpy.zeros_like(ano_label_list[0])
    for attr_ano_label_ in attr_ano_label_list:
        attr_ano_label = np.bitwise_or(attr_ano_label, attr_ano_label_)

    ano_count = np.sum(ano_label)
    str_ano_count = np.sum(str_ano_label)
    attr_ano_count = np.sum(attr_ano_label)
    print('ano_count {}'.format(ano_count))
    print('str_ano_count {}'.format(str_ano_count))
    print('attr_ano_count {}'.format(attr_ano_count))
    features, _ = preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    features = torch.FloatTensor(features)
    adj_list = torch.IntTensor(adj_list)
    neighbor_list = torch.IntTensor(neighbor_list)
    loss_full_batch = torch.zeros((nb_nodes, 1))
    node_pos = torch.LongTensor(node_pos)

    mask_down = torch.ones(subgraph_view_size, subgraph_view_size, dtype=torch.int).cuda()
    for n in node_pos: mask_down[n, n + 1:] = 0
    mask_up = mask_down.t()

    subgraph_mask_adj = torch.zeros(subgraph_view_size, subgraph_view_size).cuda()
    subgraph_mask_adj.diagonal().fill_(1)
    subgraph_mask_adj[0, :] = 1
    subgraph_mask_adj[:, 0] = 1

    for i in range(len(subgraph_size)):
        subgraph_mask_adj[node_pos[i + 1], node_pos[i + 1]:node_pos[i + 1] + subgraph_size[i]] = 1
        subgraph_mask_adj[node_pos[i + 1]:node_pos[i + 1] + subgraph_size[i], node_pos[i + 1]] = 1

    node_idx = list(range(subgraph_view_size))
    node_idx = [item for item in node_idx if item not in node_pos]

    # Initialize model and optimiser
    model = Model(ft_size, args.embedding_dim, subgraph_view_size)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        adj_list = adj_list.cuda()
        neighbor_list = neighbor_list.cuda()
        loss_full_batch = loss_full_batch.cuda()
        node_pos = node_pos.cuda()

    if torch.cuda.is_available():
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
    else:
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))

    best_auc = 0.0
    best_auc_score = ...

    ba_all, bf_all = generate_rwr_subgraph(adj_list, neighbor_list, subgraph_size, features, subgraph_mask_adj)
    # Train model
    for epoch in range(args.num_epoch):
        epoch = epoch + 1
        model.train()

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0.
        batch_num = len(all_idx) // batch_size + 1

        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            lbl = torch.unsqueeze(
                torch.cat((torch.zeros(cur_batch_size), torch.ones(cur_batch_size * args.negsamp_ratio))),
                1).cuda()

            ba_m = ba_all[idx]
            bf_m = bf_all[idx]

            ba_a = ba_m.clone()
            bf_a = bf_m.clone()

            indices = torch.randperm(features.size(0))[:bf_a.size(0)]
            for n in node_pos: bf_a[:, n, :] = features[indices, :]

            ba_s = ba_m.clone()
            bf_s = bf_m.clone()

            random.shuffle(node_idx)

            tmp = int(subgraph_view_size // negsamp_ratio)
            if tmp > (subgraph_view_size - view_size - 1):
                tmp = subgraph_view_size - view_size - 1
            for n in range(tmp):
                indices = torch.randperm(features.size(0))[:bf_a.size(0)]
                bf_s[:, node_idx[n], :] = features[indices, :]

            ba_mas = torch.cat((ba_m, ba_a, ba_s), dim=0)
            bf_mas = torch.cat((bf_m, bf_a, bf_s), dim=0)

            bf_other = bf_mas.clone()
            for n in node_pos: bf_other[:, n, :] = 0
            ba_other = ba_mas.clone()

            logits = model(ba_mas * mask_down, bf_mas, ba_other * mask_up, bf_other)
            loss_all = b_xent(logits, lbl)

            loss = torch.mean(loss_all)

            loss.backward()
            optimiser.step()

            loss = loss.detach().cpu().numpy()
            loss_full_batch[idx] = loss_all[: cur_batch_size].detach()

            if not is_final_batch:
                total_loss += loss

            print('TRAIN epoch: {}, batch_idx: {}, loss: {}'.format(epoch, batch_idx, loss))
        if epoch != 0 and (epoch) % 100 == 0:
            # Set the directory name
            directory = f"./model/"

            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = os.path.join(directory, "best_model_{}.pkl".format(epoch))

            torch.save(model.state_dict(), filename)
            print('best_model_{}.pkl'.format(epoch), ' Model saved to {}'.format(filename))

            # Test model
            model.load_state_dict(torch.load(filename))
            multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))

            # Testing
            for round in range(args.auc_test_rounds):

                all_idx = list(range(nb_nodes))
                random.shuffle(all_idx)
                batch_num = len(all_idx) // batch_size + 1

                for batch_idx in range(batch_num):

                    optimiser.zero_grad()

                    is_final_batch = (batch_idx == (batch_num - 1))

                    if not is_final_batch:
                        idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    else:
                        idx = all_idx[batch_idx * batch_size:]

                    ba_m = ba_all[idx]
                    bf_m = bf_all[idx]

                    bf_other = bf_m.clone()
                    for n in node_pos: bf_other[:, n, :] = 0
                    ba_other = ba_m.clone()

                    with torch.no_grad():
                        logits = torch.squeeze(model(ba_m * mask_down, bf_m, ba_other * mask_up, bf_other))
                    ano_score = logits.cpu().numpy()
                    multi_round_ano_score[round, idx] = ano_score

                    print('TEST round: {}, batch_idx: {}'.format(round, batch_idx))

            ano_score_final = np.mean(multi_round_ano_score, axis=0)
            auc = roc_auc_score(ano_label, ano_score_final)

            pr_auc = calculate_auc_pr(ano_score_final, ano_label)
            print('AUC:{:.4f}'.format(auc))
            print('pr_auc:{:.4f}'.format(pr_auc[0]))


if __name__ == '__main__':
    # Set argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', dest='datadir', default=r'./dataset/')
    parser.add_argument('--dataset', type=str, default='DBLP')  # 'DBLP', 'ACM', 'Amazon', 'YelpChi'
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--view_size', type=int, default=3)
    parser.add_argument('--auc_test_rounds', type=int, default=1)
    parser.add_argument('--negsamp_ratio', type=int, default=2)
    args = parser.parse_args()

    tensor = torch.tensor([3, 15, 15])

    train(args, tensor, 1.8)
