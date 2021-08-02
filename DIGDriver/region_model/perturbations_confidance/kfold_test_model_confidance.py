#!/usr/bin/env python
import os
import sys
import json
import copy
import h5py
import numpy as np
import pandas as pd
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

sys.path.append('/storage/yaari/mutation_density/pytorch/nets/')
sys.path.append('/storage/yaari/mutation_density/pytorch/')

from cnn_predictors import *
from mut_dataset import *

def add_noise_to_model(model, noise):
    tmp_model = copy.deepcopy(model).cuda()
    with torch.no_grad():
        for param in tmp_model.parameters():
            param.add_(torch.normal(0, noise, param.size()).cuda())
    return tmp_model

def predict(model, data_loader, label_ids):   
    corr_coef_sums = np.zeros(len(label_ids))
    all_preds = [[] for _ in range(len(label_ids))]
    all_true = [[] for _ in range(len(label_ids))]
    for j, (X, t_lst) in enumerate(data_loader):
        y_lst = model(X.cuda())
        with torch.no_grad():
            for i, t in enumerate(t_lst):
                y = y_lst[i]
                all_preds[i].extend(y.data.cpu().numpy().tolist())
                all_true[i].extend(t.data.cpu().numpy().tolist())
    return all_preds, all_true, [r2_score(all_preds[i], all_true[i]) for i in range(len(label_ids))]


def test_with_perturbations(model, data_loader, label_ids, samp_num, params, fold, verbose=True):      
    preds = np.empty((samp_num, params.reps))
    for rep in range(params.reps):
        tmp_model = add_noise_to_model(model, params.alpha)
        tmp_preds, _, acc = predict(tmp_model, data_loader, label_ids)
        preds[:, rep] = tmp_preds[0]
        
        if verbose and rep % 10 == 0:
            print('Fold {}, repetition {}, accuracy: {}'.format(fold, rep, acc))    
    return preds


def main():
    assert len(sys.argv) >= 4, 'Usage: kfold_test_model_confidance.py <run_id> <models folder name> <cancer ids...>'
        
    cur_dir = os.path.dirname(os.path.realpath(__file__))  
    config_path = os.path.join(cur_dir, "../configs/config_confidance_kfold.json")
    with open(config_path, 'r') as f: config = json.load(f)

    run_id = sys.argv[1]
    label_ids = sys.argv[3:]
    labels_str = '-'.join(label_ids)
    models_dir = os.path.join(config['base_path'], labels_str, sys.argv[2])

    file_path = config['data_file']
    with h5py.File(file_path, 'r') as h5f:
        chr_idxs = h5f['idx'][:]
    
    k = config['k']
    params = SimpleNamespace()
    params.reps = config['repetitions']
    params.alpha = config['alpha']
    params.bs = config['bs']

    pred_df = pd.DataFrame()
    idx = 0
    for i in range(2):
        print('Running iteration {} out of {} folds...'.format(i + 1, k))
        test_idxs = np.sort(np.load(os.path.join(models_dir, 'test_indices_fold_{}.npy'.format(i))))

        test_ds = SimpleDatasetFromH5(file_path, label_ids, test_idxs, chr_idxs[test_idxs], 'x_data')
        test_dl = DataLoader(test_ds, batch_size=params.bs, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)
        samp_num = len(test_ds)
        test_chr_idxs = chr_idxs[test_idxs]

        print('Loading model...')
        model = nn.DataParallel(SimpleMultiTaskResNet(test_ds.get_data_shape(), len(label_ids))).cuda()
        state_dict = torch.load(os.path.join(models_dir, 'best_model_fold_{}.pt'.format(i)))
        model.load_state_dict(state_dict)
        model.eval()
        
        print('Computing prediction and confidance...')
        preds, labels, acc = predict(model, test_dl, label_ids)
        perturp_preds = test_with_perturbations(model, test_dl, label_ids, samp_num, params, i)

        print('Model accuracy: {}'.format(acc))
        print('Storing predictions...')

        fold_pred_df = pd.DataFrame(data=perturp_preds)
        fold_pred_df['chr'] = test_chr_idxs[:,0]
        fold_pred_df['s_idx'] = test_chr_idxs[:,1]
        fold_pred_df['e_idx'] = test_chr_idxs[:,2]
        fold_pred_df['obs_mut'] = labels[0]
        fold_pred_df['pred_mut'] = preds[0]
        pred_df = pred_df.append(fold_pred_df, ignore_index=True)

    out_dir = os.path.join(models_dir, run_id)
    out_path = os.path.join(out_dir, 'perturb_predictions.csv')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('Saving predictions to {}...'.format(out_path))
    pred_df.to_csv(out_path)
    
    print('Done!')
    
if __name__ == '__main__':
    main()
