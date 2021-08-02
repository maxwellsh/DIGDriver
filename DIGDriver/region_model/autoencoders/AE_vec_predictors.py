#!/usr/bin/env python
import os
import sys
import h5py
import argparse
import datetime
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import copy


py_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(py_file_path, 'data_aux'))
sys.path.append(os.path.join(py_file_path, 'trainers'))
sys.path.append(os.path.join(file_path, 'ae_nets'))

from dataset_generator import *
from fc_nets import *
from gp_trainer import *
from CNNs import *

def get_cmd_arguments():
    ap = argparse.ArgumentParser()

    # Required cancer type argument
    ap.add_argument('-c', '--cancer-id', required=True, nargs='*', action='store', type=str, dest='label_ids',
                    help='A list of the h5 file mutation count dataset IDs (e.g. SNV_skin_melanoma_MELAU_AU)')

    # Path arguments
    ap.add_argument('-d', "--data", required=False, nargs='?', action='store', type=str, dest='data_file',
                    default='/storage/datasets/cancer/data_vecs_PCAWG_1000000_0_0.7.h5', help='Path to h5 data file')
    ap.add_argument('-o', "--out-dir", required=False, nargs='?', action='store', type=str, dest='out_dir',
                    default='/storage/yaari/mutation-density-outputs', help='Path to output directory')
    ap.add_argument('-u', "--held-out", required=False, nargs='?', action='store', type=str, dest='heldout_file',
                    default=None, help='Path to file of held-out samples file')
    ap.add_argument('-t', "--tracks", required=False, nargs='?', action='store', type=str, dest='track_file',
                    default=None, help='Path to predictor tracks selection file')
    ap.add_argument('-m', "--mappability", required=False, nargs='?', action='store', type=float, dest='mappability',
                    default=0.7, help='Mappability lower bound')

    # Run type parameters
    ap.add_argument('-s', "--split", required=False, nargs='?', action='store', type=str, dest='split_method',
                    default='random', help='Dataset split method (random/chr)')
    ap.add_argument('-r', "--train-ratio", required=False, nargs='?', action='store', type=float, dest='train_ratio',
                    default=0.8, help='Train set split size ratio')
    ap.add_argument('-ho', "--heldout-ratio", required=False, nargs='?', action='store', type=float, dest='heldout_ratio',
                    default=0.2, help='Held-out set split size ratio (will be extracted prior to train validation split)')
    ap.add_argument('-b', "--batch", required=False, nargs='?', action='store', type=int, dest='bs',
                                        default=128, help='Batch size')
    ap.add_argument('-gr', "--gp_reruns", required=False, nargs='?', action='store', type=int, dest='gp_reruns',
                    default=10, help='Number of GP reinitializations and training runs')
    ap.add_argument('-re', "--reruns", required=False, nargs='?', action='store', type=int, dest='reruns',
                    default=1, help='Number of models retraining runs')
    ap.add_argument('-e', "--epochs", required=False, nargs='?', action='store', type=int, dest='epochs',
                    default=20, help='Number of epochs')
    ap.add_argument('-g', "--gpus", required=False, nargs='?', action='store', type=str, dest='gpus',
                                        default='all', help='GPUs devices (all/comma separted list)')
    return ap.parse_args()

def run_gp(device, train_set, test_set, ho_set=None, h5_grp = None, rerun_num=1):
    for j in range(rerun_num):
        print('GP run {}/{}...'.format(j+1, rerun_num))
        run_successeed = False
        n_inducing = 2000
        while not run_successeed and n_inducing > 0:
            gp_trainer = GPTrainer(device, train_set, test_set, ho_set, n_inducing=n_inducing)
            try:
                print('Running GP with {} inducing points...'.format(n_inducing))
                gp_test_results, gp_ho_results = gp_trainer.run()
            except RuntimeError as err:
                print('Run failed with {} inducing points. Encountered run-time error in training: {}'
                      .format(n_inducing, err))
                n_inducing -= 200
                continue
            run_successeed = True
        if run_successeed and h5_grp != None:
            gp_trainer.save_results(gp_test_results, gp_ho_results, h5_grp, str(j))

def train(model, device, epoch, train_ds, label_ids, loss_func, optimizer, net_type, writer = None):
    model.train()
    batch_num = len(train_ds)
    loss_sum = 0
    all_features = [[] for _ in range(len(label_ids))]
    all_true = [[] for _ in range(len(label_ids))]
    for batch_idx, (X, t_lst) in enumerate(train_ds):
        #flatten
        bs, w, tracks = X.size()
        if net_type == 'fc':
            X = X.view(bs,-1, w * tracks)
        X = X.to(device)
        encoded, decoded = model(X)

        loss = loss_func(decoded, X)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

        with torch.no_grad():
            for i, t in enumerate(t_lst):
                feature_vecs = encoded
                all_features[i].append(feature_vecs.cpu().detach().numpy())
                all_true[i].extend(t.data.cpu().numpy().tolist())

        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %d | Batch %03d/%03d | Loss: %.4f'
                %(epoch, batch_idx, len(train_ds), loss))
    epoch_loss = loss_sum / batch_num

    if writer is not None:
        writer.add_scalar('train_loss', epoch_loss, epoch)
    all_features = [np.concatenate(all_features[j], axis=0) for j in range(len(all_features))]

    return all_features, all_true, epoch_loss

def embed(model, device, data_ds, label_ids, net_type):
    model.eval()
    data_loader = DataLoader(data_ds, batch_size=2048, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)
    all_features = [[] for _ in range(len(label_ids))]
    all_true = [[] for _ in range(len(label_ids))]
    for j, (X, t_lst) in enumerate(data_loader):
        bs, w, tracks = X.size()
        if net_type == 'fc':
            X = X.view(bs,-1, w * tracks)
        X = X.to(device)
        features_lst = model.module.embeding(X)
        with torch.no_grad():
            for i, t in enumerate(t_lst):
                if net_type == 'fc':
                    feature_vecs = features_lst[:,0,:]
                else:
                    feature_vecs = features_lst
                all_features[i].append(feature_vecs.cpu().detach().numpy())
                all_true[i].extend(t.data.cpu().numpy().tolist())
    all_features = [np.concatenate(all_features[j], axis=0) for j in range(len(all_features))]
    return all_features, all_true

def eval(model, device, data_ds, label_ids, loss_fn, net_type, writer = None):
    model.eval()
    batch_num = len(data_ds)
    loss_sum = 0
    all_features = [[] for _ in range(len(label_ids))]
    all_true = [[] for _ in range(len(label_ids))]
    for j, (X, t_lst) in enumerate(data_ds):
        bs, w, tracks = X.size()
        if net_type == 'fc':
            X = X.view(bs,-1, w * tracks)
        X = X.to(device)
        encoded, decoded = model(X)
        with torch.no_grad():
            loss_sum += loss_fn(decoded, X)
            for i, t in enumerate(t_lst):
                feature_vecs = encoded
                all_features[i].append(feature_vecs.cpu().detach().numpy())
                all_true[i].extend(t.data.cpu().numpy().tolist())
    test_loss = loss_sum / batch_num
    all_features = [np.concatenate(all_features[j], axis=0) for j in range(len(all_features))]
    print('====> Test set loss: {}'.format(test_loss))
    return all_features, all_true, test_loss

def main():
    args = get_cmd_arguments()
    out_dir = os.path.join(args.out_dir)
    print('Saving results under: \'{}\''.format(out_dir))
    args_dict = vars(args)
    with open(os.path.join(out_dir, 'run_params.txt'), 'w') as f:
        [f.write('{}: {}\n'.format(k, args_dict[k])) for k in args_dict.keys()]

    if args.gpus is None:
        print('Using CPU device.')
        device = torch.device('cpu')
    else:
        print('Using GPU device: \'{}\''.format(args.gpus))
        device = torch.device('cuda')
        if args.gpus != 'all':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    acc_df = pd.DataFrame()
    out_h5f = h5py.File(os.path.join(out_dir, 'vector_models_output.h5'), 'w')

    label_group_lst = [out_h5f.create_group(l) for l in args.label_ids]

    for r in range(args.reruns):
        print('Run {}/{}...'.format(r + 1, args.reruns))
        run_group_lst = [lbl_grp.create_group('run_{}'.format(r)) for lbl_grp in label_group_lst]
        data_generator = DatasetGenerator(args.data_file, args.label_ids, args.mappability, args.heldout_ratio, heldout_file=args.heldout_file,)
        train_ds, test_ds = data_generator.get_datasets(args.split_method, args.train_ratio)
        ho_ds = data_generator.get_heldout_dataset()
        train_dataloader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=False, num_workers=16)
        test_dataloader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, drop_last=False, num_workers=16)


        net_type = 'cnn'
        model = ResNetAE(train_ds.get_data_shape())
        print('Running {} AE model'.format(net_type))

        if args.gpus is not None: model = nn.DataParallel(model)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
        loss_fn = nn.MSELoss()
        epochs = args.epochs

        best_epoch = 0
        best_test_loss = np.inf
        for epoch in range(1, epochs + 1):
            print('Running epoch {}/{}'.format(epoch, epochs))
            train_features_lst, train_true_lst, train_loss = train(model, device, epoch, train_dataloader, args.label_ids, loss_fn, optimizer, net_type)
            test_features_lst, test_true_lst, test_loss = eval(model, device, test_dataloader, args.label_ids, loss_fn, net_type)

            #run GP every epoch
            for l in range(len(args.label_ids)):
                train_set = (np.array(train_features_lst[l]), np.array(train_true_lst[l]), train_ds.get_chromosome_locations())
                test_set = (np.array(test_features_lst[l]), np.array(test_true_lst[l]), test_ds.get_chromosome_locations())
                try:
                    run_gp(device, train_set, test_set)
                except:
                    print('Unexpected error during GP run: \n', sys.exc_info()[0])
                    print('GP Run failed, skipping to next epoch.')
                    continue

            #if GP works, save model if it is best performing on test set
            if test_loss < best_test_loss:
                best_test_loss, best_epoch = test_loss, epoch
                best_run_model = copy.deepcopy(model)

        print('Best validation loss for run {}/{} was: {}.'.format(r + 1, args.reruns, best_test_loss))
        print('Switching to model from epoch {}'.format(best_epoch))
        model = best_run_model

        train_features, train_labels = embed(model,device, train_ds, args.label_ids, net_type)
        test_features, test_labels = embed(model, device, test_ds, args.label_ids, net_type)
        ho_features, ho_labels = embed(model, device, ho_ds, args.label_ids, net_type)

        ae_group_lst = [run_grp.create_group('ae_gp') for run_grp in run_group_lst]

        for i in range(len(args.label_ids)):
            print('Running gaussian process model for {}...'.format(args.label_ids[i]))
            train_set = (np.array(train_features[0]), np.array(train_labels[i]), train_ds.get_chromosome_locations())
            test_set = (np.array(test_features[0]), np.array(test_labels[i]), test_ds.get_chromosome_locations())
            ho_set = (np.array(ho_features[0]), np.array(ho_labels[i]), ho_ds.get_chromosome_locations())
            try:
                run_gp(device, train_set, test_set, ho_set, ae_group_lst[i], args.gp_reruns)
            except:
                print('Unexpected error during GP run: \n', sys.exc_info()[0])
                print('retrying')
                try:
                    run_gp(device, train_set, test_set, ho_set, ae_group_lst[i], args.gp_reruns)
                except:
                    print('Another nexpected error during GP run: \n', sys.exc_info()[0])
            print('Finished {} reruns for {}'.format(args.gp_reruns, args.label_ids[i]))
        print('Finished {}th rerun!'.format(r+1))
    out_h5f.close()
    print('Done!')


if __name__ == '__main__':
    main()
