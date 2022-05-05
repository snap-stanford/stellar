import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import SAGEConv

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return 10 * out

class Encoder(nn.Module):
    def __init__(self, x_dim, num_cls):
        super(Encoder, self).__init__()
        self.x_dim = x_dim
        hid_dim = 128
        self.conv1 = nn.Linear(x_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, hid_dim)
        self.relu = nn.ReLU()
        self.linear = NormedLinear(hid_dim, num_cls)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x)
        feat = x
        x = self.relu(x)
        x = self.conv2(x,edge_index)
        out_feat = x
        out = self.linear(x)
        return out, feat, out_feat
    
class FCNet(nn.Module):
    def __init__(self, x_dim, num_cls):
        super(FCNet, self).__init__()
        self.linear = NormedLinear(x_dim, num_cls)

    def forward(self, data):
        x = data.x
        out = self.linear(x)
        return out, x, x