import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class Model(nn.Module):
    def __init__(self, n_in, n_h, node_dim, activation='prelu'):
        super(Model, self).__init__()
        self.encoderMAS = GCN(n_in, n_h, activation)
        self.encoderOTH = GCN(n_in, n_h, activation)
        self.linerA = nn.Linear(node_dim * n_h, n_h)
        self.linerB = nn.Linear(node_dim * n_h, n_h)

    def forward(self, adjMAS, seqMAS, adjOTHER, seqOTHER, sparse=False):
        x_mas = self.encoderMAS(seqMAS, adjMAS, sparse)
        x_oth = self.encoderOTH(seqOTHER, adjOTHER, sparse)
        x_mas = torch.reshape(x_mas, (len(x_mas), -1))
        x_oth = torch.reshape(x_oth, (len(x_oth), -1))

        x_mas = self.linerA(x_mas)
        x_oth = self.linerB(x_oth)
        embed_node = torch.sqrt(torch.sum((x_mas - x_oth) ** 2, dim=1)).unsqueeze(-1)
        return embed_node
