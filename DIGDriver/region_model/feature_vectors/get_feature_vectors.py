#!/usr/bin/env python
import os
import re
import sys
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

sys.path.append('/storage/yaari/mutation_density/pytorch/nets/')
sys.path.append('/storage/yaari/mutation_density/pytorch/trainers/')
sys.path.append('/storage/yaari/mutation_density/pytorch/data_aux/')

from cnn_predictors import *
from mut_dataset import *

def tokens_match(strg, search=re.compile(r'[^:0-9]').search):
    return not bool(search(strg))

def load_track_selection_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    track_lst = []
    for i, l in enumerate(lines):
        if l.startswith(('\n', '#')): continue
        l = l.rstrip()  # remove trailing '\n'
        assert tokens_match(l), \
            'Expected track selection lines to contain only digits and colons. Found: {} in line #{}.'.format(l, i)

        split_l = l.split(':')
        assert len(split_l) <= 2, \
            'Expected track selection lines to contain only one colon. Found: {} in line #{}.'.format(l, i)
        assert np.all([split_l[j].isdigit() for j in range(len(split_l))]), \
            'Expected to have a number in both sides of the colon. Found: {} in line #{}.'.format(l, i)

        if len(split_l) == 1:
            track_lst.append(int(split_l[0]))
        elif len(split_l) == 2:
            assert int(split_l[0]) < int(split_l[1]), 'Expected x < y in pair x:y. Found: {} in line #{}.'.format(l, i)
            track_lst.extend(np.arange(int(split_l[0]), int(split_l[1])).tolist())

    print('Selected {} tracks: \n{}'.format(len(track_lst), track_lst))
    return track_lst

def predict(model, data_loader, label_ids):   
    corr_coef_sums = np.zeros(len(label_ids))
    all_preds = [[] for _ in range(len(label_ids))]
    all_features = [[] for _ in range(len(label_ids))]
    all_true = [[] for _ in range(len(label_ids))]
    for j, (X, t_lst) in enumerate(data_loader):
        y_lst, features_lst, _ = model(X.cuda())
        with torch.no_grad():
            for i, t in enumerate(t_lst):
                y = y_lst[i]
                feature_vecs = features_lst[i]
                all_features[i].append(feature_vecs.cpu().detach().numpy())
                all_preds[i].extend(y.data.cpu().numpy().tolist())
                all_true[i].extend(t.data.cpu().numpy().tolist())
    all_features = [np.concatenate(all_features[j], axis=0) for j in range(len(all_features))]
    return all_preds, all_true, all_features, [r2_score(all_true[i], all_preds[i]) for i in range(len(label_ids))]

def main():
    assert len(sys.argv) == 3, 'Usage: get_feature_vectors.py <model dir path> <run id>'
    models_dir = sys.argv[1]
    run_id = sys.argv[2]

    with open(os.path.join(models_dir, 'run_params.txt'), 'r') as f:
        config_lst = [(l.split(':')) for l in f.read().split('\n')]
        config_dict = {x[0].strip(): x[1].strip() for x in config_lst if len(x) > 1}

    test_idxs = np.sort(np.load(os.path.join(models_dir, 'test_indices_fold_{}.npy'.format(run_id))))
    label_ids = config_dict['label_ids'].replace('[\'', '').replace('\']', '').split(', ')

    file_path = config_dict['data_file']
    with h5py.File(file_path, 'r') as h5f:
        chr_idxs = h5f['idx'][:]
        pred_h = h5f['x_data'].shape[2]

    track_file = config_dict['track_file']
    if track_file != 'None':
        selected_tracks = load_track_selection_file(os.path.join(os.path.dirname(__file__), track_file))
    else:
        selected_tracks = np.arange(pred_h)

    test_chr_idxs = chr_idxs[test_idxs]
    test_ds = LazyLoadDatasetFromH5(file_path, label_ids, test_idxs, test_chr_idxs, selected_tracks, 'x_data')
    test_dl = DataLoader(test_ds, batch_size=4096, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)
    train_idxs = np.delete(np.arange(len(chr_idxs)), test_idxs)
    train_chr_idxs = chr_idxs[train_idxs]
    train_ds = LazyLoadDatasetFromH5(file_path, label_ids, train_idxs, train_chr_idxs, selected_tracks, 'x_data')
    train_dl = DataLoader(train_ds, batch_size=4096, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)
    samp_num = len(test_ds)

    print('Loading model...')
    model = nn.DataParallel(SimpleMultiTaskResNet(test_ds.get_data_shape(), len(label_ids))).cuda()
    state_dict = torch.load(os.path.join(models_dir, 'best_model_fold_{}.pt'.format(run_id)))
    model.load_state_dict(state_dict)
    model.eval()

    print('Computing {} train set features...'.format(train_ds.get_data_shape()[0]))
    train_preds, train_labels, train_features, acc = predict(model, train_dl, label_ids)
    print('Model train accuracy: {}'.format(acc))

    print('Computing {} test set features...'.format(test_ds.get_data_shape()[0]))
    test_preds, test_labels, test_features, acc = predict(model, test_dl, label_ids)
    print('Model test accuracy: {}'.format(acc))

    print('Saving features, predictions and true labels...')
    with h5py.File(os.path.join(models_dir, 'gaussian_process_data_{}.h5'.format(run_id)), 'w') as h5f:
        train_group = h5f.create_group('train')
        train_group.create_dataset('true', data=np.array(train_labels))
        train_group.create_dataset('predicted', data=np.array(train_preds))
        train_group.create_dataset('idxs', data=np.array(train_chr_idxs))
        train_group.create_dataset('features', data=np.array(train_features))
        test_group = h5f.create_group('test')
        test_group.create_dataset('true', data=np.array(test_labels))
        test_group.create_dataset('predicted', data=np.array(test_preds))
        test_group.create_dataset('idxs', data=np.array(test_chr_idxs))
        test_group.create_dataset('features', data=np.array(test_features))

    print('Done!')

if __name__ == '__main__':
    main()
