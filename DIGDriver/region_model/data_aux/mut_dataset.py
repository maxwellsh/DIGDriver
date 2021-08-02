import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class SimpleDataset(Dataset):

    def __init__(self, data, labels_lst):
        self.data = data
        self.labels_lst = [lbl for lbl in labels_lst]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X = torch.tensor(self.data[idx]).float()
        y_lst = [torch.tensor(l[idx]).float() for l in self.labels_lst]
        return X, y_lst

    def get_data_shape(self):
        return self.data.shape

    def get_train_set_length(self, train_ratio):
        return int(train_ratio * self.data.shape[0])


class BaseDatasetFromH5(Dataset):
    def __init__(self, preprocessed_idxs, chr_locations, mappability, quantiles, selected_tracks):
        self.preprocessed_idxs = preprocessed_idxs
        self.chr_locations = chr_locations
        self.selected_tracks = selected_tracks
        self.mappability = mappability
        self.quantiles = quantiles

    def __len__(self):
        return len(self.preprocessed_idxs)

    def get_set_indices(self):
        return self.preprocessed_idxs

    def get_chromosome_locations(self):
        return self.chr_locations[self.preprocessed_idxs]

    def get_mappability_values(self):
        return self.mappability[self.preprocessed_idxs]

    def get_quantile_values(self):
        return self.quantiles[self.preprocessed_idxs]


class SimpleDatasetFromH5(BaseDatasetFromH5):
    def __init__(self, h5_file, label_ids, preprocessed_idxs, chr_locations, mappability, quantiles, selected_tracks, data_id):
        super(SimpleDatasetFromH5, self).__init__(preprocessed_idxs, chr_locations, mappability, quantiles, selected_tracks)
        print('Loading data and labels from file {}...'.format(h5_file))
        with h5py.File(h5_file, 'r') as h5f:
            self.data = torch.tensor(h5f[data_id][np.sort(self.preprocessed_idxs)]).float()
            self.labels_lst = [torch.tensor(h5f[l][np.sort(self.preprocessed_idxs)]).float() for l in label_ids]
        print('Loaded input data of size: {}'.format(self.data.shape))

    def __getitem__(self, idx):
        X = self.data[idx, :, self.selected_tracks]
        y_lst = [l[idx] for l in self.labels_lst]
        return X, y_lst

    def get_data_shape(self):
        return self.data.shape


class LazyLoadDatasetFromH5(BaseDatasetFromH5):
    def __init__(self, h5_file, label_ids, preprocessed_idxs, chr_locations, mappability, quantiles, selected_tracks, data_id, auto_context=None):
        super(LazyLoadDatasetFromH5, self).__init__(preprocessed_idxs, chr_locations, mappability, quantiles, selected_tracks)
        self.h5_file = h5_file
        self.label_ids = label_ids
        self.data_id = data_id

    def __getitem__(self, idx):
        data_idx = self.preprocessed_idxs[idx]
        with h5py.File(self.h5_file,'r') as db:
            X = torch.tensor(db[self.data_id][data_idx, :, self.selected_tracks]).float()
            y_lst = [torch.tensor(db[l][data_idx]).float() for l in self.label_ids]
        return X, y_lst

    def get_data_shape(self):
        with h5py.File(self.h5_file,'r') as db:
            return (len(self.preprocessed_idxs), db[self.data_id].shape[1], len(self.selected_tracks))


class AutoregressiveDatasetFromH5(BaseDatasetFromH5):
    def __init__(self, h5_file, label_ids, preprocessed_idxs, chr_locations, mappability, quantiles, selected_tracks, data_id, auto_context=1):
        super(AutoregressiveDatasetFromH5, self).__init__(preprocessed_idxs, chr_locations, mappability, quantiles, selected_tracks)
        self.h5_file = h5_file
        self.label_ids = label_ids
        self.data_id = data_id
        self.auto_context = auto_context

    def get_context(self, c_idx, s_idx, e_idx):
        s = s_idx if s_idx >= 0 else 0
        e = e_idx if e_idx < len(self.chr_locations) else len(self.chr_locations) - 1               
        return np.arange(s, e)[np.where(self.chr_locations[np.arange(s, e), 0] == self.chr_locations[c_idx, 0])[0]]

    def __getitem__(self, idx):
        data_idx = self.preprocessed_idxs[idx]
        pre_context = self.get_context(data_idx, data_idx-self.auto_context, data_idx)
        post_context = self.get_context(data_idx, data_idx+1, data_idx+self.auto_context+1)
        with h5py.File(self.h5_file,'r') as db:
            X = torch.tensor(db[self.data_id][data_idx, :, self.selected_tracks]).float()
            X_auto = [torch.tensor([db[l][pre_context].sum(), db[l][post_context].sum()]).float() for l in self.label_ids]
            y_lst = [torch.tensor(db[l][data_idx]).float() for l in self.label_ids]
        return X, X_auto, y_lst

    def get_data_shape(self):
        with h5py.File(self.h5_file,'r') as db:
            return (len(self.preprocessed_idxs), db[self.data_id].shape[1], len(self.selected_tracks))
