#!/usr/bin/env python
import os
import sys
import h5py
import copy
import argparse
import datetime
import numpy as np
import pandas as pd
from torch import nn, optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

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
                    default='/storage/datasets/cancer/unzipped_data_matrices_PCAWG_10000_0_0.0.h5', help='Path to h5 data file')
    ap.add_argument('-o', "--out-dir", required=False, nargs='?', action='store', type=str, dest='out_dir',
                    default='/storage/yaari/mutation-density-outputs', help='Path to output directory')
    ap.add_argument('-u', "--held-out", required=False, nargs='?', action='store', type=str, dest='heldout_file',
                    default=None, help='Path to file of held-out samples file')

    # Run type parameters
    ap.add_argument('-s', "--split", required=False, nargs='?', action='store', type=str, dest='split_method',
                    default='random', help='Dataset split method (random/chr)')
    ap.add_argument('-m', "--mappability", required=False, nargs='?', action='store', type=float, dest='mappability',
                    default=0.7, help='Mappability lower bound')
    ap.add_argument('-gp', "--gaussian", required=False, nargs='?', action='store', type=bool, dest='run_gaussian',
                    default=False, help='True: train gaussian process regression on the best performing model')
    ap.add_argument('-n', "--network", required=False, nargs='?', action='store', type=str, dest='net',
                    default='cnn', help='The type of neural network model to use (\'fc\' or \'cnn\')')

    # Train parameters
    ap.add_argument('-r', "--train-ratio", required=False, nargs='?', action='store', type=float, dest='train_ratio',
                    default=0.8, help='Train set split size ratio')
    ap.add_argument('-ho', "--heldout-ratio", required=False, nargs='?', action='store', type=float, dest='heldout_ratio',
                    default=0.2, help='Held-out set split size ratio (will be extracted prior to train validation split)')
    ap.add_argument('-e', "--epochs", required=False, nargs='?', action='store', type=int, dest='epochs',
                    default=20, help='Number of epochs')
    ap.add_argument('-b', "--batch", required=False, nargs='?', action='store', type=int, dest='bs',
                    default=128, help='Batch size')
    ap.add_argument('-re', "--reruns", required=False, nargs='?', action='store', type=int, dest='nn_reruns',
                    default=1, help='Number of NN reinitializations and training runs')
    ap.add_argument('-gr', "--gp-reruns", required=False, nargs='?', action='store', type=int, dest='gp_reruns',
                    default=1, help='Number of GP reinitializations and training runs')

    ap.add_argument('-lr', "--learning-rate", required = False, nargs='?', action='store', type = float, dest = 'lr', default=1e-3, help = 'learning rate for training')

    # Run management parameters
    ap.add_argument('-sm', "--save-model", required=False, nargs='?', action='store', type=bool, dest='save_model',
                    default=False, help='True: save best model across all reruns')
    ap.add_argument('-st', "--save-training", required=False, nargs='?', action='store', type=float, dest='save_training',
                    default=False, help='True: save training process and results to Tensorboard file')
    ap.add_argument('-g', "--gpus", required=False, nargs='?', action='store', type=str, dest='gpus',
                    default='all', help='GPUs devices (all/comma separted list)')

    return ap.parse_args()

def train(model, device,  epoch, train_ds, loss_func, optimizer, net_type, writer = None):
    model.train()
    batch_num = len(train_ds)
    loss_sum = 0
    for batch_idx, (X, y) in enumerate(train_ds):
        #flatten
        bs, w, tracks = X.size()
        if net_type == 'fc':
            X = X.view(bs,-1, w * tracks)
        X = X.to(device)
        decoded = model(X)

        loss = loss_func(decoded, X)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        loss_sum += loss.item()
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %d | Batch %03d/%03d | Loss: %.4f'
                %(epoch, batch_idx, len(train_ds), loss))
    epoch_loss = loss_sum / batch_num
    if writer is not None:
        writer.add_scalar('train_loss', epoch_loss, epoch)

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

def eval(model, device, data_ds, loss_fn, label_ids, net_type, writer = None):
    model.eval()
    batch_num = len(data_ds)
    loss_sum = 0

    for j, (X, t_lst) in enumerate(data_ds):
        bs, w, tracks = X.size()
        if net_type == 'fc':
            X = X.view(bs,-1, w * tracks)
        X = X.to(device)
        decoded = model(X)
        with torch.no_grad():
            loss_sum += loss_fn(decoded, X)# + torch.norm(attention, p=1, dim=(1,2)).mean()
    test_loss = loss_sum / batch_num

    print('====> Test set loss: {}'.format(test_loss))
    if writer is not None:
        writer.add_scalar('test_loss', test_loss, epoch)

    return test_loss

def main():
    args = get_cmd_arguments()
    labels_str = '-'.join(args.label_ids)
    out_dir = os.path.join(args.out_dir)
    print('Generating prediction for cancer types: {}'.format(args.label_ids))

    # Configure GPUs
    if args.gpus is None:
        print('Using CPU device.')
        device = torch.device('cpu')
    else:
        print('Using GPU device: \'{}\''.format(args.gpus))
        device = torch.device('cuda')
        if args.gpus != 'all':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    data_generator = DatasetGenerator(args.data_file,
                                      args.label_ids,
                                      args.mappability,
                                      args.heldout_ratio,
                                      heldout_file=args.heldout_file,)

    bs = args.bs
    net_type = args.net
    train_ds, test_ds = data_generator.get_datasets(args.split_method, args.train_ratio)
    ho_ds = data_generator.get_heldout_dataset()
    train_dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False, num_workers=16)
    test_dataloader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=16)

    if net_type == 'fc':
        model = Autoencoder_FC(train_ds.get_data_shape())
    elif net_type == 'cnn_ld':
        model = ResNetAE_LD(train_ds.get_data_shape())
    elif net_type == 'cnn_sld':
        model = ResNetAE_SLD(train_ds.get_data_shape())
    else:
        model = ResNetAE(train_ds.get_data_shape())

    if args.gpus is not None: model = nn.DataParallel(model)
    model.to(device)

    print('Running {} AE model'.format(net_type))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
    loss_fn = nn.MSELoss()

    epochs = args.epochs
    # Create output directory
    if args.save_model or args.save_training or args.run_gaussian:
        print('Saving results under: \'{}\''.format(out_dir))
        os.makedirs(out_dir)
        args_dict = vars(args)
        with open(os.path.join(out_dir, 'run_params.txt'), 'w') as f:
            [f.write('{}: {}\n'.format(k, args_dict[k])) for k in args_dict.keys()]

    if args.save_training:
        writer = SummaryWriter(logdir=out_dir, comment=labels_str)
        writer.add_text('configurations', str(args), 0)
        writer.add_text('model', str(model), 0)
    else:
        writer = None
    for epoch in range(1, epochs + 1):
        print('Running epoch {}/{}'.format(epoch, epochs))
        train(model, device, epoch, train_dataloader, loss_fn, optimizer, net_type, writer)
        eval(model, device, test_dataloader, loss_fn, args.label_ids, net_type)

    print('Done Training!')

    if args.save_model:
        print('Saving model')
        torch.save(model.state_dict(), os.path.join(out_dir, 'saved_model_{}_e{}_{}.h5'.format(net_type, epochs,
                                                                                               args.label_ids[i])))
    train_features, train_labels = embed(model,device, train_ds, args.label_ids, net_type)
    test_features, test_labels = embed(model, device, test_ds, args.label_ids, net_type)
    ho_features, ho_labels = embed(model, device, ho_ds, args.label_ids, net_type)

    #run gaussian
    for i in range(len(args.label_ids)):
        print('Running gaussian process model for {}...'.format(args.label_ids[i]))
        train_set = (np.array(train_features[i]), np.array(train_labels[i]), train_ds.get_chromosome_locations())
        test_set = (np.array(test_features[i]), np.array(test_labels[i]), test_ds.get_chromosome_locations())
        ho_set = (np.array(ho_features[i]), np.array(ho_labels[i]), ho_ds.get_chromosome_locations())
        best_r2 = 0
        for j in range(args.gp_reruns):
            print('GP run {}/{}...'.format(j, args.gp_reruns))
            run_successeed = False
            n_inducing = 2000
            while not run_successeed and n_inducing > 0:
                gp_trainer = GPTrainer(device, train_set, test_set, heldout_tup=ho_set, n_inducing=n_inducing)
                try:
                    print('Running GP with {} inducing points...'.format(n_inducing))
                    gp_test_results, gp_ho_results = gp_trainer.run()
                except RuntimeError as err:
                    print('Run failed with {} inducing points. Encountered run-time error in training: {}'
                          .format(n_inducing, err))
                    n_inducing -= 200
                    continue
                run_successeed = True
            if gp_test_results['r2'] > best_r2:
                best_test_results, best_ho_results = gp_test_results, gp_ho_results
                best_r2 = gp_test_results['r2']
        gp_out_path = os.path.join(out_dir, 'model_{}_e{}_gp_results_{}.h5'.format(net_type, epochs, args.label_ids[i]))
        if best_r2 > 0:
            gp_trainer.save_results(gp_out_path, best_test_results, best_ho_results)
        else:
            gp_trainer.save_results(gp_out_path, gp_test_results, gp_ho_results)



if __name__ == '__main__':
    main()
