#!/usr/bin/env python
import os
import sys
import h5py
import copy
import argparse
import numpy as np
import pandas as pd
from torch import nn, optim
from tensorboardX import SummaryWriter
from datetime import datetime

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, 'nets'))
sys.path.append(os.path.join(file_path, 'trainers'))
sys.path.append(os.path.join(file_path, 'data_aux'))
sys.path.append(os.path.join(file_path, '../sequence_model'))

#from rnn_predictors import *
from cnn_predictors import *

from nn_trainer import *
from gp_trainer import *
from dataset_generator import *
from mutations_main import OutputGenerator
from gp_tools import *

def get_cmd_arguments(text=None):
    ap = argparse.ArgumentParser()

    # Required cancer type argument
    ap.add_argument('-c', '--cancer-id', required=True, nargs='*', action='store', type=str, dest='label_ids',
                    help='A list of the h5 file mutation count dataset IDs (e.g. SNV_skin_melanoma_MELAU_AU)')

    # Path arguments
    ap.add_argument('-d', "--data", required=False, nargs='?', action='store', type=str, dest='data_file',
                    default='/storage/datasets/cancer/unzipped_data_matrices_pcawg_10k.h5', help='Path to h5 data file')
    ap.add_argument('-o', "--out-dir", required=False, nargs='?', action='store', type=str, dest='out_dir',
                    default='/storage/yaari/mutation-density-outputs', help='Path to output directory')
    ap.add_argument('-t', "--tracks", required=False, nargs='?', action='store', type=str, dest='track_file',
                    default=None, help='Path to predictor tracks selection file')

    # Run type parameters
    ap.add_argument('-s', "--split", required=False, nargs='?', action='store', type=str, dest='split_method',
                    default='random', help='Dataset split method (random/chr)')
    ap.add_argument('-m', "--mappability", required=False, nargs='?', action='store', type=float, dest='mappability',
                    default=0.5, help='Mappability lower bound')
    ap.add_argument('-cq', "--count-quantile", required=False, nargs='?', action='store', type=float, dest='count_quantile',
                    default=0.999, help='Region mutation count quanitle threshold.')
    ap.add_argument('-a', "--attention", required=False, action='store_true', dest='get_attention',
                    help='True: train with attention map training and save attention maps')
    ap.add_argument('-gp', "--gaussian", required=False, nargs='?', action='store', type=int, dest='run_gaussian',
                    default=5, help='True: train gaussian process regression on the best performing model')
    ap.add_argument('-as', "--autoregressive-size", required=False, nargs='?', action='store', type=int,
                    dest='autoregressive_size', default=0, help='number of neighbouring regions for autoregressive features')
    # Train parameters
    ap.add_argument('-k', required=False, nargs='?', action='store', type=int, dest='k',
                    default=5, help='Number of folds')
    ap.add_argument('-gr', "--gp-reruns", required=False, nargs='?', action='store', type=int, dest='gp_reruns',
                    default=3, help='GP maximum reinitializations for convergence')
    ap.add_argument('-gd', "--gp-delta", required=False, nargs='?', action='store', type=int, dest='gp_delta',
                    default=0.03, help='Maximum difference between a fold NN and GP scores')
    ap.add_argument('-re', "--nn-reruns", required=False, nargs='?', action='store', type=int, dest='nn_reruns',
                    default=1, help='Number of model reinitializations and training runs')
    ap.add_argument('-mr', "--max-nn-reruns", required=False, nargs='?', action='store', type=int, dest='max_nn_reruns',
                    default=3, help='NN maximum reinitializations for GP to successeed')
    ap.add_argument('-vr', "--val-ratio", required=False, nargs='?', action='store', type=float, dest='val_ratio',
                    default=0.2, help='Validation set split size ratio')
    ap.add_argument('-e', "--epochs", required=False, nargs='?', action='store', type=int, dest='epochs',
                    default=20, help='Number of epochs')
    ap.add_argument('-b', "--batch", required=False, nargs='?', action='store', type=int, dest='bs',
                    default=128, help='Batch size')
    ap.add_argument('-nd', "--n-inducing", required=False, nargs='?', action='store', type=int, dest='n_inducing',
                    default=400, help='Number of GP inducing points')
    ap.add_argument('-nt', "--n-iter", required=False, nargs='?', action='store', type=int, dest='n_iter',
                    default=50, help='Number of GP iterations')

    # Run management parameters
    ap.add_argument('-sm', "--save-model", required=False, action='store_true', dest='save_model',
                    help='True: save best model across all reruns')
    ap.add_argument('-st', "--save-training", required=False, action='store_true', dest='save_training',
                    help='True: save training process and results to Tensorboard file')
    ap.add_argument('-g', "--gpus", required=False, nargs='?', action='store', type=str, dest='gpus',
                    default='all', help='GPUs devices (all/comma separted list)')
    ap.add_argument('-u', "--sub_mapp", required=False,  action='store_true',  dest='sub_mapp',
                    help='True: run model on regions below mappability threshold')

    if text:
        args = ap.parse_args(text.split())
    else:
        args = ap.parse_args()

    return args


def main(input_args=None):
    if input_args is None:
        args = get_cmd_arguments()
    else:
        args = input_args

    labels_str = '-'.join(args.label_ids)
    out_dir = os.path.join(args.out_dir, 'kfold', labels_str, str(datetime.now()))
    print('Generating prediction for cancer types: {}'.format(args.label_ids))

    if args.gpus is None:
        print('Using CPU device.')
        device = torch.device('cpu')
    else:
        print('Using GPU device: \'{}\''.format(args.gpus))
        device = torch.device('cuda')
        if args.gpus != 'all':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    out_pred = OutputGenerator(args, device, out_dir)

    os.makedirs(out_dir)
    args_dict = vars(args)
    with open(os.path.join(out_dir, 'run_params.txt'), 'w') as f:
        [f.write('{}: {}\n'.format(k, args_dict[k])) for k in args_dict.keys()]

    best_model_file = os.path.join(out_dir, 'best_model_fold_{}.pt')
    val_set_file = os.path.join(out_dir, 'val_indices_fold_{}')

    if args.save_model or args.save_training:
        print('Saving results under: \'{}\''.format(out_dir))

    data_generator = KFoldDatasetGenerator(args)
    is_autoreg = args.autoregressive_size > 0
    model_func = AutoregressiveMultiTaskResNet if is_autoreg else SimpleMultiTaskResNet
    print('Running {}-fold prediction...'.format(args.k))

    k, re = 0, 0
    gp_succeed = False
    while k < args.k and re < args.max_nn_reruns:
        train_ds, val_ds, ho_ds = data_generator.get_datasets(k)
        best_overall_acc = -np.inf
        for r in range(args.nn_reruns):
            print('Setting model and optimizers for run {}/{} and fold {}/{}...'.format(r + 1, args.nn_reruns, k + 1, args.k))
            model = model_func(train_ds.get_data_shape(), len(args.label_ids), get_attention_maps=args.get_attention)
            optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
            loss_fn = nn.MSELoss()
            if args.gpus is not None: model = nn.DataParallel(model)

            if args.save_training:
                writer = SummaryWriter(logdir=out_dir, comment=labels_str)
                writer.add_text('configurations', str(args), 0)
                writer.add_text('model', str(model), 0)
            else:
                writer = None
            trainer = NNTrainer(model,
                                optimizer,
                                loss_fn,
                                args.bs,
                                args.label_ids,
                                train_ds,
                                val_ds,
                                device,
                                writer,
                                get_attention_maps=args.get_attention)

            best_run_acc = -np.inf
            for epoch in range(1, args.epochs + 1):
                print('Running epoch {}/{}'.format(epoch, args.epochs))
                train_losses, train_accs, train_features_lst, train_pred_lst, train_true_lst = \
                    trainer.train(epoch, r, autoreg=is_autoreg)
                val_losses, val_accs, val_features_lst, val_pred_lst, val_true_lst, val_attention = \
                    trainer.test(epoch, r, autoreg=is_autoreg)

                # Keep only the best model with > 2 non-zero features according to test performance 
                non_zero_features = np.where(np.abs(train_features_lst[0]).mean(axis=0) > 0)[0]
                print('#non-zero features: {}'.format(len(non_zero_features)))
                if val_accs[0] > best_run_acc and len(non_zero_features) > 1:
                    print('Changing run model since best R2 was {} compared to previous {}'.format(val_accs[0], best_run_acc))
                    best_run_acc = val_accs[0]
                    best_run_model, best_run_att = copy.deepcopy(model), val_attention
                    train_dict = {'feat': train_features_lst, 'lbls': train_true_lst, 'ds': train_ds}
                    val_dict = {'feat': val_features_lst, 'lbls': val_true_lst, 'ds': val_ds}

            if best_run_acc > best_overall_acc:
                best_overall_acc = best_run_acc
                best_overall_model = best_run_model
                best_train_dict, best_val_dict = train_dict, val_dict

            print(bcolors.OKCYAN + 'Best epoch validation accuracy for run {}/{} was: {}.'.format(r + 1, args.nn_reruns, best_run_acc) + bcolors.ENDC)
        print(bcolors.OKCYAN + 'Best overall validation accuracy over {} reruns was: {}.'.format(args.nn_reruns, best_overall_acc) + bcolors.ENDC)

        # Save attention maps from best overall model
        if args.get_attention:
            out_pred.save_attetnion_maps('attention_maps_{}.h5'.format(k), best_run_att, val_ds, val_pred_lst, val_true_lst)

         # Save best run model
        if args.save_model:
            print('Saving model and validation indices for future evaluations to {}...'.format(val_set_file))
            np.save(val_set_file.format(k), val_ds.get_set_indices())
            torch.save(best_overall_model.state_dict(), best_model_file.format(k))

        # Run GP on best overall model
        if args.run_gaussian > 0:
            print('Computing {} validation set features...'.format(ho_ds.get_data_shape()[0]))
            ho_preds, ho_labels, ho_features, ho_acc, ho_att = out_pred.predict(best_overall_model, ho_ds)
            ho_dict = {'feat': ho_features, 'lbls': ho_labels, 'ds': ho_ds}
            print(bcolors.OKCYAN + 'Model held-out accuracy: {}'.format(ho_acc) + bcolors.ENDC)

            gp_succeed = out_pred.run_gp('gp_results_fold_{}.h5'.format(k), train_dict, val_dict, ho_dict, best_overall_acc, k)

            if args.sub_mapp:
                sub_ds = data_generator.get_below_mapp()
                print('Computing {} sub-theshold features...'.format(sub_ds.get_data_shape()[0]))
                sub_preds, sub_labels, sub_features, sub_acc, sub_att = out_pred.predict(best_overall_model, sub_ds)
                sub_dict = {'feat': sub_features, 'lbls': sub_labels, 'ds': sub_ds}
                print(bcolors.OKCYAN + 'Model sub-mappable accuracy: {}'.format(sub_acc) + bcolors.ENDC)

                # Save attention maps from unmappable regions
                sub_att_path = os.path.join(out_dir, 'attention_maps_submapp.h5')
                if args.get_attention and not os.path.exists(sub_att_path):
                    out_pred.save_attetnion_maps('attention_maps_submapp.h5', sub_att, sub_ds, sub_preds, sub_labels)

                out_pred.run_gp('sub_mapp_results_fold_{}.h5'.format(k), train_dict, val_dict, sub_dict, best_overall_acc, k, prefix='sub')

        if args.run_gaussian > 0 and not gp_succeed:
            re += 1
            print(bcolors.FAIL + 'GP run failed! Rerunning NN, attempt {}/{}'.format(re + 1, args.max_nn_reruns) + bcolors.ENDC)
        else:
            k += 1
            re = 0

    assert gp_succeed, 'GP failed at fold {} after {} NN reruns'.format(k, re)
    print('Done!')


if __name__ == '__main__':
    startTime = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - startTime))
