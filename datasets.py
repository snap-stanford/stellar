import torch
import numpy as np
from builtins import range
from torch_geometric.data import InMemoryDataset, Data
import sklearn

class CodexGraphDataset(InMemoryDataset):

    def __init__(self, labeled_X, labeled_y, unlabeled_X, labeled_pos=None, unlabeled_pos=None, distance_thres=None, transform=None,):
        super(CodexGraphDataset, self).__init__()
        self.distance_thres = distance_thres
        if labeled_pos and unlabeled_pos:
            labeled_edge_index = self.get_edge_index(labeled_pos)
            unlabeled_edge_index = self.get_edge_index(unlabeled_pos)
        else:
            labeled_edge_index = None
            unlabeled_edge_index = None

        self.labeled_data = Data(x=torch.FloatTensor(labeled_X), edge_index=labeled_edge_index, y=torch.LongTensor(labeled_y))
        self.unlabeled_data = Data(x=torch.FloatTensor(unlabeled_X), edge_index=unlabeled_edge_index)

        
    def get_edge_index(self, pos):
        edge_list = []
        num_samples = len(pos)
        dists = sklearn.metrics.pairwise_distances(pos)
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                if dists[i,j] < self.distance_thres:
                    edge_list.append([i,j])
                    edge_list.append([j,i])
        return torch.LongTensor(edge_list).T

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.labeled_data, self.unlabeled_data

