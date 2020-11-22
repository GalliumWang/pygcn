import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import torch



# class SparseDropout(torch.nn.Module):       #对稀疏矩阵进行dropout
#     def __init__(self, dprob=0.1):
#         super(SparseDropout, self).__init__()
#         # dprob is ratio of dropout
#         # convert to keep probability
#         self.kprob=1-dprob

#     def forward(self, x):
#         mask=((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
#         rc=x._indices()[:,mask]
#         val=x._values()[mask]*(1.0/self.kprob)
#         return torch.sparse.FloatTensor(rc, val)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)       #两层图卷积


        self.gc1 = GraphConvolution(nfeat, 600)
        self.gc2 = GraphConvolution(600, 16)
        self.gc3 = GraphConvolution(16, 4)
        self.gc4 = GraphConvolution(4, nclass)

        # self.gc5 = GraphConvolution(800, 700)
        # self.gc6 = GraphConvolution(700, 600)
        # self.gc7 = GraphConvolution(600, 500)
        # self.gc8 = GraphConvolution(500, 400)
        # self.gc9 = GraphConvolution(400, 300)
        # self.gc10 = GraphConvolution(300, 200)
        # self.gc11 = GraphConvolution(200, 100)
        # self.gc12 = GraphConvolution(100, 16)
        # self.gc13 = GraphConvolution(16, nclass)

        self.dropout = dropout
        #self.edgedropout=0.1



    def forward(self, x, adj):

        # sdropout=SparseDropout()
        # adj_copy=SparseDropout(adj)


        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc4(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc5(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc6(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc7(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc8(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc9(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc10(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc11(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc12(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)


        x = self.gc4(x, adj)
        return F.log_softmax(x, dim=1)
