import torch
import numpy as np
from builtins import range
from torch_geometric.data import InMemoryDataset, Data
import sklearn
import pandas as pd

def get_hubmap_edge_index(pos, regions, distance_thres):
    edge_list = []
    regions_unique = np.unique(regions)
    for reg in regions_unique:
        locs = np.where(regions == reg)[0]
        pos_region = pos[locs, :]
        dists = sklearn.metrics.pairwise_distances(pos_region)
        dists_mask = dists < distance_thres
        np.fill_diagonal(dists_mask, 0)
        region_edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
        for (i, j) in region_edge_list:
            edge_list.append([locs[i], locs[j]])
    return edge_list

def get_tonsilbe_edge_index(pos, distance_thres):
    edge_list = []
    dists = sklearn.metrics.pairwise_distances(pos)
    dists_mask = dists < distance_thres
    np.fill_diagonal(dists_mask, 0)
    edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
    return edge_list

def load_hubmap_data(labeled_file, unlabeled_file, distance_thres, sample_rate):
    train_df = pd.read_csv(labeled_file)
    test_df = pd.read_csv(unlabeled_file)
    train_df = train_df.sample(n=round(sample_rate*len(train_df)), random_state=1)
    test_df = test_df.sample(n=round(sample_rate*len(test_df)), random_state=1)
    train_df = train_df.loc[np.logical_and(train_df['tissue'] == 'CL', train_df['donor'] == 'B004')]
    test_df = test_df.loc[np.logical_and(test_df['tissue'] == 'CL', test_df['donor'] == 'B005')]
    train_X = train_df.iloc[:, 1:49].values
    test_X = test_df.iloc[:, 1:49].values
    labeled_pos = train_df.iloc[:, -6:-4].values
    unlabeled_pos = test_df.iloc[:, -5:-3].values
    labeled_regions = train_df['unique_region']
    unlabeled_regions = test_df['unique_region']
    train_y = train_df['cell_type_A']
    cell_types = np.sort(list(set(train_df['cell_type_A'].values))).tolist()
    cell_type_dict = {}
    for i, cell_type in enumerate(cell_types):
        cell_type_dict[cell_type] = i
    train_y = np.array([cell_type_dict[x] for x in train_y])
    labeled_edges = get_hubmap_edge_index(labeled_pos, labeled_regions, distance_thres)
    unlabeled_edges = get_hubmap_edge_index(unlabeled_pos, unlabeled_regions, distance_thres)
    return train_X, train_y, test_X, labeled_edges, unlabeled_edges

def load_tonsilbe_data(filename, mapping_filename, distance_thres, sample_rate):
    df = pd.read_csv(filename)
    df_sample = df.sample(n=round(sample_rate*len(df)), random_state=1)
    train_df = df_sample.loc[df['sample_name'] == 'tonsil']
    test_df = df_sample.loc[df_sample['sample_name'] == 'Barretts Esophagus']
    train_X = train_df.iloc[:, 1:-4].values
    test_X = test_df.iloc[:, 1:-4].values
    train_y = train_df['final_cell_type'].str.lower()
    level_mappping_df = pd.read_csv(mapping_filename)
    level_mapping = {}
    for i in range(len(level_mappping_df)):
        level_mapping[level_mappping_df['Level_1'][i].lower()] = level_mappping_df['Level_3'][i].lower()
    train_y = [level_mapping[x] for x in train_y]
    labeled_pos = train_df.iloc[:, -4:-2].values
    unlabeled_pos = test_df.iloc[:, -4:-2].values
    cell_types = np.sort(list(set(train_y))).tolist()
    cell_type_dict = {}
    for i, cell_type in enumerate(cell_types):
        cell_type_dict[cell_type] = i
    train_y = np.array([cell_type_dict[x] for x in train_y])
    labeled_edges = get_tonsilbe_edge_index(labeled_pos, distance_thres)
    unlabeled_edges = get_tonsilbe_edge_index(unlabeled_pos, distance_thres)
    return train_X, train_y, test_X, labeled_edges, unlabeled_edges

class CodexGraphDataset(InMemoryDataset):

    def __init__(self, labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges, transform=None,):
        self.root = '.'
        super(CodexGraphDataset, self).__init__(self.root, transform)
        self.labeled_data = Data(x=torch.FloatTensor(labeled_X), edge_index=torch.LongTensor(labeled_edges).T, y=torch.LongTensor(labeled_y))
        self.unlabeled_data = Data(x=torch.FloatTensor(unlabeled_X), edge_index=torch.LongTensor(unlabeled_edges).T)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.labeled_data, self.unlabeled_data