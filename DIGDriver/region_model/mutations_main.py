
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

from cnn_predictors import *
from nn_trainer import *
from gp_trainer import *
from dataset_generator import *

from gp_tools import *


def get_cmd_arguments():
    ap = argparse.ArgumentParser()

    # Required cancer type argument
    ap.add_argument('-c', '--cancer-id', required=True, nargs='*', action='store', type=str, dest='label_ids',
                    help='A list of the h5 file mutation count dataset IDs (e.g. SNV_skin_melanoma_MELAU_AU). Best model is selected based on first cohort in the list. ')

    # Path arguments
    ap.add_argument('-d', "--data", required=False, nargs='?', action='store', type=str, dest='data_file',
                    default='/storage/datasets/cancer/unzipped_data_matrices_pcawg_50k.h5', help='Path to h5 data file')
    ap.add_argument('-o', "--out-dir", required=False, nargs='?', action='store', type=str, dest='out_dir',
                    default='/storage/yaari/mutation-density-outputs', help='Path to output directory')
    ap.add_argument('-u', "--held-out", required=False, nargs='?', action='store', type=str, dest='heldout_file',
                    default=None, help='Path to file of held-out samples file')
    ap.add_argument('-t', "--tracks", required=False, nargs='?', action='store', type=str, dest='track_file',
                    default=None, help='Path to predictor tracks selection file')

    # Run type parameters
    ap.add_argument('-s', "--split", required=False, nargs='?', action='store', type=str, dest='split_method',
                    default='random', help='Dataset split method (random/chr)')
    ap.add_argument('-m', "--mappability", required=False, nargs='?', action='store', type=float, dest='mappability',
                    default=0.7, help='Mappability lower bound')
    ap.add_argument('-cq', "--count-quantile", required=False, nargs='?', action='store', type=float, dest='count_quantile',
                    default=0.995, help='Region mutation count quanitle threshold.')
    ap.add_argument('-a', "--attention", required=False, action='store_true', dest='get_attention',
                    help='True: train with attention map training and save attention maps')
    ap.add_argument('-at', "--attended-tracks", required=False, action='store_true', dest='get_attended_tracks',
                    help='True: train with attention map training and extract selected tracks')
    ap.add_argument('-ac', "--attended-columns", required=False, action='store_true', dest='get_attended_cols',
                    help='True: train with attention map training and extract attended columns')
    ap.add_argument('-gp', "--gaussian", required=False, nargs='?', action='store', type=int, dest='run_gaussian',
                    default=0, help='Number of GP reinitializations and training runs')
    ap.add_argument('-n', "--network", required=False, nargs='?', action='store', type=str, dest='net',
                    default='cnn', help='The type of neural network model to use (\'fc\' or \'cnn\')')
    ap.add_argument('-as', "--autoregressive-size", required=False, nargs='?', action='store', type=int,
                    dest='autoregressive_size', default=0, help='number of neighbouring regions for autoregressive features')

    # Train parameters
    ap.add_argument('-vr', "--val-ratio", required=False, nargs='?', action='store', type=float, dest='val_ratio',
                    default=0.2, help='Validation set split size ratio')
    ap.add_argument('-hr', "--heldout-ratio", required=False, nargs='?', action='store', type=float, dest='heldout_ratio',
                    default=0.2, help='Held-out set split size ratio (will be extracted prior to train validation split)')
    ap.add_argument('-e', "--epochs", required=False, nargs='?', action='store', type=int, dest='epochs',
                    default=20, help='Number of epochs')
    ap.add_argument('-b', "--batch", required=False, nargs='?', action='store', type=int, dest='bs',
                    default=128, help='Batch size')
    ap.add_argument('-nd', "--n-inducing", required=False, nargs='?', action='store', type=int, dest='n_inducing',
                    default=400, help='Number of GP inducing points')
    ap.add_argument('-nt', "--n-iter", required=False, nargs='?', action='store', type=int, dest='n_iter',
                    default=50, help='Number of GP iterations')
    ap.add_argument('-gr', "--gp-reruns", required=False, nargs='?', action='store', type=int, dest='gp_reruns',
                    default=3, help='GP maximum reinitializations for convergence')
    ap.add_argument('-gd', "--gp-delta", required=False, nargs='?', action='store', type=float, dest='gp_delta',
                    default=0.03, help='Maximum difference between a fold NN and GP scores')
    ap.add_argument('-re', "--nn-reruns", required=False, nargs='?', action='store', type=int, dest='nn_reruns',
                    default=1, help='Number of NN reinitializations and training runs')
    ap.add_argument('-mr', "--max-nn-reruns", required=False, nargs='?', action='store', type=int, dest='max_nn_reruns',
                    default=2, help='NN maximum reinitializations for GP to successeed')

    # Run management parameters
    ap.add_argument('-sm', "--save-model", required=False, action='store_true', dest='save_model',
                    help='True: save best model across all reruns')
    ap.add_argument('-st', "--save-training", required=False, action='store_true', dest='save_training',
                    help='True: save training process and results to Tensorboard file')
    ap.add_argument('-g', "--gpus", required=False, nargs='?', action='store', type=str, dest='gpus',
                    default='all', help='GPUs devices (all/comma separted list)')

    return ap.parse_args()


class OutputGenerator:
    pretrained_cols = ['CHROM', 'START', 'END', 'Y_TRUE', 'Y_PRED', 'STD', 'FLAG', 'MAPP', 'QUANT', 'FOLD']
    nn_acc_col = 'nn_acc'
    val_acc_col = 'val_acc'
    ho_acc_col = 'test_acc'

    def __init__(self, args, device, out_dir):
        self.args = args
        self.autoreg = args.autoregressive_size > 0
        self.device = device
        self.out_dir = out_dir
        self.pretrained_path = os.path.join(self.out_dir, '{}.Pretrained.h5') #.format('_'.join(self.args.label_ids)))
        self.acc_path = os.path.join(self.out_dir, '{}_pretrained_accuracy.txt')

        fold_idxs = np.arange(args.k) if hasattr(args, 'k') else np.arange(args.nn_reruns)
        run_idxs = np.arange(args.run_gaussian).astype(str)
        if hasattr(args, 'sub_mapp') and args.sub_mapp:
            run_idxs = np.concatenate([run_idxs, np.array(['sub_' + str(r) for r in range(args.run_gaussian)])])

        mult_idx = pd.MultiIndex.from_product([fold_idxs, run_idxs], names=['fold', 'gp_run'])
        self.score_dict = {l: pd.DataFrame(index=mult_idx, columns=[self.nn_acc_col, self.val_acc_col, self.ho_acc_col]) for l in self.args.label_ids}
        self.pretrained_dict = {l: pd.DataFrame(columns=self.pretrained_cols) for l in self.args.label_ids}

    def predict(self, model, data_ds):
        data_loader = DataLoader(data_ds, batch_size=self.args.bs, shuffle=False, drop_last=False, pin_memory=True, num_workers=16)
        all_preds = [[] for _ in range(len(self.args.label_ids))]
        all_features = [[] for _ in range(len(self.args.label_ids))]
        all_true = [[] for _ in range(len(self.args.label_ids))]
        all_att = []
        for j, batch in enumerate(data_loader):
            t_lst = batch[-1]
            if self.autoreg:
                y_lst, features_lst, attention = model(batch[0].to(self.device), torch.cat(batch[1], dim=1).to(self.device))
            else:
                y_lst, features_lst, attention = model(batch[0].to(self.device))
            if attention is not None: all_att.append(attention.cpu().detach().numpy())
            with torch.no_grad():
                for i, t in enumerate(t_lst):
                    y = y_lst[i]
                    feature_vecs = features_lst[i]
                    all_features[i].append(feature_vecs.cpu().detach().numpy())
                    all_preds[i].extend(y.data.cpu().numpy().tolist())
                    all_true[i].extend(t.data.cpu().numpy().tolist())
        all_features = [np.concatenate(all_features[j], axis=0) for j in range(len(all_features))]
        accs = [r2_score(all_true[i], all_preds[i]) for i in range(len(self.args.label_ids))]
        if len(all_att) > 0:
            return all_preds, all_true, all_features, accs, np.concatenate(all_att, axis=0)
        else:
            return all_preds, all_true, all_features, accs, None

    def store_pretrained(self, lbl, chr_locs, mapps, quants, y_true, means, stds, fold, is_flagged=False):
        pretrained_df = self.pretrained_dict[lbl]
        flags = np.ones(len(y_true)) if is_flagged else np.zeros(len(y_true))
        folds = np.ones(len(y_true)) * fold
        data = np.concatenate([chr_locs,
                               y_true.reshape(-1, 1),
                               means.reshape(-1, 1),
                               stds.reshape(-1, 1),
                               flags.reshape(-1, 1),
                               mapps.reshape(-1, 1),
                               quants.reshape(-1, 1),
                               folds.reshape(-1, 1)], axis=1)

        pretrained_df = pretrained_df.append(pd.DataFrame(data, columns=self.pretrained_cols), ignore_index=True)
        pretrained_df.sort_values(by=['CHROM', 'START']) \
                     .reset_index(drop=True) \
                     .to_hdf(self.pretrained_path.format(lbl), 'region_params', mode='a', complib='zlib')

        unflagged_idxs = np.where(pretrained_df['FLAG'] == 0)[0]
        pretrained_acc = r2_score(pretrained_df.loc[unflagged_idxs, 'Y_TRUE'], pretrained_df.loc[unflagged_idxs, 'Y_PRED'])
        print(bcolors.OKGREEN + 'Overall unflagged pretrained accuracy after fold {} is: {}'.format(fold + 1, pretrained_acc) + bcolors.ENDC)
        with open(self.acc_path.format(lbl), 'w') as f:
            f.write(str(pretrained_acc))

        self.pretrained_dict[lbl] = pretrained_df

    def run_gp_iteration(self, nn_score, train_set, val_set, ho_set=None):
        run_successeed = False
        n_inducing = self.args.n_inducing
        while not run_successeed and n_inducing > 0:
            for r in range(self.args.gp_reruns):
                gp_trainer = GPTrainer(self.device, train_set, val_set, heldout_tup=ho_set, n_iter=self.args.n_iter, n_inducing=n_inducing)
                try:
                    print('Running GP with {} inducing points...'.format(n_inducing))
                    gp_val_results, gp_ho_results = gp_trainer.run()
                    gp_score = r2_score(val_set[1], gp_val_results['gp_mean'])
                except RuntimeError as err:
                    print(bcolors.WARNING + 'Warning: GP Run failed with {} inducing points. Encountered run-time error in training: {}'
                          .format(n_inducing, err) + bcolors.ENDC)
                    continue

                if gp_score - nn_score < -self.args.gp_delta:
                    print(bcolors.WARNING + 'Warning: GP run R2={}, failed to reach minimal accuracy of {}'
                          .format(np.round(gp_score, 4), np.round(nn_score - self.args.gp_delta, 4)) + bcolors.ENDC)
                else:
                    run_successeed = True
                    break
            n_inducing -= 100
        if run_successeed:
            return gp_trainer, gp_val_results, gp_ho_results
        return None, None, None

    def run_gp(self, f_name, train_dict, val_dict, ho_dict, nn_scores, fold, prefix=''):
        gp_h5 = h5py.File(os.path.join(self.out_dir, f_name), 'w')

        for l, lbl in enumerate(self.args.label_ids):
            nn_score = nn_scores[l]
            print('Running gaussian process model for {}...'.format(self.args.label_ids[l]))
            lbl_grp = gp_h5.create_group(self.args.label_ids[l])
            train_set = (np.array(train_dict['feat'][l]),
                         np.array(train_dict['lbls'][l]),
                         train_dict['ds'].get_chromosome_locations(),
                         train_dict['ds'].get_mappability_values(),
                         train_dict['ds'].get_quantile_values())
            val_set = (np.array(val_dict['feat'][l]),
                       np.array(val_dict['lbls'][l]),
                       val_dict['ds'].get_chromosome_locations(),
                       val_dict['ds'].get_mappability_values(),
                       val_dict['ds'].get_quantile_values())
            ho_set = (np.array(ho_dict['feat'][l]),
                      np.array(ho_dict['lbls'][l]),
                      ho_dict['ds'].get_chromosome_locations(),
                      ho_dict['ds'].get_mappability_values(),
                      ho_dict['ds'].get_quantile_values())

            score_df = self.score_dict[lbl]
            # Run and store multiple GPs
            for j in range(self.args.run_gaussian):
                print('GP run {}/{}...'.format(j + 1, self.args.run_gaussian))
                gp_trainer, gp_val_results, gp_ho_results = self.run_gp_iteration(nn_score, train_set, val_set, ho_set)
                if gp_val_results is not None:
                    val_score, ho_score = gp_trainer.save_results(gp_val_results, gp_ho_results, lbl_grp, str(j))
                    if len(prefix) > 0:
                        score_df.loc[fold, prefix + '_' + str(j)] = np.array([nn_score, val_score, ho_score])
                    else:
                        score_df.loc[fold, str(j)] = np.array([nn_score, val_score, ho_score])
                    score_df.to_csv(os.path.join(self.out_dir, '{}_gp_runs_summary.csv'.format(lbl)))
                else:
                    return False
            chr_locs, mapps, quants, y_true, means, stds = gp_trainer.compute_pretrained(lbl_grp, self.args.run_gaussian)
            print(bcolors.OKGREEN + 'Fold {} pretrained model R2: {}'
                  .format(str(fold + 1) + '_' + prefix if len(prefix) > 0 else str(fold + 1), r2_score(y_true, means)) + bcolors.ENDC)

            print('GP fold results summary:')
            print(score_df)

            self.store_pretrained(lbl, chr_locs, mapps, quants, y_true, means, stds, fold, is_flagged=len(prefix) > 0)
            self.score_dict[lbl] = score_df

        return True

    def save_attetnion_maps(self, f_name, att, ds, pred_arr, true_arr, feats_dict=None):
        with h5py.File(os.path.join(self.out_dir, f_name), 'w') as h5f:
            h5f.create_dataset('attention_maps', data=att)
            h5f.create_dataset('chr_locs', data=ds.get_chromosome_locations())
            h5f.create_dataset('idxs', data=ds.get_set_indices())
            h5f.create_dataset('pred_lbls', data=pred_arr)
            h5f.create_dataset('true_lbls', data=true_arr)
            if feats_dict is not None:
                h5f.create_dataset('train_feats', data=feats_dict['train'])
                h5f.create_dataset('val_feats', data=feats_dict['val'])
                h5f.create_dataset('ho_feats', data=feats_dict['ho'])

    def save_prediction(self, f_name, ds, pred_arr, true_arr, feats_dict=None):
        with h5py.File(os.path.join(self.out_dir, f_name), 'w') as h5f:
            h5f.create_dataset('chr_locs', data=ds.get_chromosome_locations())
            h5f.create_dataset('idxs', data=ds.get_set_indices())
            h5f.create_dataset('pred_lbls', data=pred_arr)
            h5f.create_dataset('true_lbls', data=true_arr)
            if feats_dict is not None:
                h5f.create_dataset('train_feats', data=feats_dict['train'])
                h5f.create_dataset('val_feats', data=feats_dict['val'])
                h5f.create_dataset('ho_feats', data=feats_dict['ho'])

    @staticmethod
    def get_attended_columns(att_maps, thresh=10):
        att_col_lst = []
        for i in range(att_maps.shape[0]):
            cols = np.where(att_maps[i].sum(axis=0) > thresh)[0][1:-1]
            col_diff = np.diff(cols)
            att_col_start = cols[np.concatenate([np.array([0]), np.where(col_diff > 1)[0] + 1])]
            att_col_end = cols[np.concatenate([np.where(col_diff > 1)[0], np.array([-1])])]
            att_col_lst.append([att_col_start, att_col_end])
        return att_col_lst


def main():
    args = get_cmd_arguments()
    labels_str = '-'.join(args.label_ids)
    out_dir = os.path.join(args.out_dir, labels_str, str(datetime.now()))
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

    out_pred = OutputGenerator(args, device, out_dir)

    save_att_maps = True
    if args.get_attended_tracks or args.get_attended_cols:
        track_lst = []
        if not args.get_attention:
            args.get_attention = True
            save_att_maps = False

    # Create output directory
    if args.save_model or args.save_training or args.get_attention or args.run_gaussian:
        print('Saving results under: \'{}\''.format(out_dir))
        os.makedirs(out_dir)
        args_dict = vars(args)
        with open(os.path.join(out_dir, 'run_params.txt'), 'w') as f:
            [f.write('{}: {}\n'.format(k, args_dict[k])) for k in args_dict.keys()]

    # Train model for multiple reruns and choose best as final
    is_autoreg = args.autoregressive_size > 0
    accs_df = pd.DataFrame()
    #for r in range(args.nn_reruns):
    r, re = 0, 0
    gp_succeed = False
    while r < args.nn_reruns and re < args.max_nn_reruns:
        # Intialize new dataset split
        data_generator = DatasetGenerator(args)
        train_ds, val_ds = data_generator.get_datasets(args.split_method)
        ho_ds = data_generator.get_heldout_dataset()
        print('Using {} predictors for prediction.'.format(train_ds.get_data_shape()[2]))

        # Initialize a new model
        print('Setting model and optimizers for run {}/{}...'.format(r + 1, args.nn_reruns))
        if args.net == 'fc':
            model_func = AutoregressiveFCNet if is_autoreg else FCNet
            model = model_func(train_ds.get_data_shape(), len(args.label_ids))
        else:
            model_func = AutoregressiveMultiTaskResNet if is_autoreg else SimpleMultiTaskResNet
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
        nn_trainer = NNTrainer(model,
                               optimizer,
                               loss_fn,
                               args.bs,
                               args.label_ids,
                               train_ds,
                               val_ds,
                               device,
                               writer,
                               get_attention_maps=args.get_attention)

        # Run datasplit training and evaluation
        best_run_acc = 0
        best_run_accs = np.zeros(len(args.label_ids))
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            print('Running epoch {}/{}'.format(epoch, args.epochs))
            train_losses, train_accs, train_features_lst, train_pred_lst, train_true_lst = \
                nn_trainer.train(epoch, r, autoreg=is_autoreg)
            val_losses, val_accs, val_features_lst, val_pred_lst, val_true_lst, val_attention = \
                nn_trainer.test(epoch, r, autoreg=is_autoreg)

            # Keep only the best model with > 2 non-zero features according to test performance 
            #non_zero_features = np.where(np.abs(train_features_lst[0]).mean(axis=0) > 0)[0]
            #print('#non-zero features: {}'.format(len(non_zero_features)))
            #if val_accs[0] > best_run_acc and len(non_zero_features) > 1:

            non_zero_feature_lst = [np.where(np.abs(train_features_lst[k]).mean(axis=0) > 0)[0] for k in range(len(train_features_lst))]
            if np.mean(val_accs) > np.mean(best_run_accs) and np.all([len(non_zero_feature_lst[k]) > 1 for k in range(len(non_zero_feature_lst))]):
                #print('Changing run model since best R2 was {} compared to previous {}'.format(val_accs[0], best_run_acc))
                #best_run_acc, best_epoch = val_accs[0], epoch
                print('Changing run model since best R2 was {} compared to previous {}'.format(np.mean(val_accs), best_run_acc))
                best_run_accs, best_epoch = val_accs, epoch
                best_train_accs, best_val_accs = train_accs, val_accs
                best_run_model, best_run_att = copy.deepcopy(model), val_attention
                best_train_feat, best_val_feat = train_features_lst, val_features_lst
                train_dict = {'feat': train_features_lst, 'lbls': train_true_lst, 'ds': train_ds}
                val_dict = {'feat': val_features_lst, 'lbls': val_true_lst, 'ds': val_ds}

        # Evaluate model performance over held-out set
        print(bcolors.OKCYAN + 'Best validation accuracy for run {}/{} was: {}.'.format(r + 1, args.nn_reruns, best_run_acc) + bcolors.ENDC)
        print(bcolors.OKCYAN + 'Running best model over {} held-out set samples...'.format(ho_ds.get_data_shape()[0]) + bcolors.ENDC)
        ho_preds, ho_labels, ho_features, ho_accs, ho_att = out_pred.predict(best_run_model, ho_ds)
        ho_dict = {'feat': ho_features, 'lbls': ho_labels, 'ds': ho_ds}
        print(bcolors.OKCYAN + 'Model held-out accuracy: {}'.format(ho_accs) + bcolors.ENDC)

        # Save run performance
        for j, l in enumerate(args.label_ids):
            accs_df.loc[r, 'Train_{}'.format(l)] = best_train_accs[j]
            accs_df.loc[r, 'Va_{}'.format(l)] = best_val_accs[j]
            accs_df.loc[r, 'Held-out_{}'.format(l)] = ho_accs[j]

        if args.get_attended_tracks:
            count_arr = np.array([best_run_att[i].max(axis=1) > 0.01 for i in range(best_run_att.shape[0])]).sum(axis=0)
            track_lst.append(count_arr / best_run_att.shape[0])

        # Save attended columns
        if args.get_attended_cols:
            print('Computing and saving attended held-out columns for run {} to {}...'.format(r, out_dir))
            att_cols = out_pred.get_attended_columns(ho_att)
            np.save(os.path.join(out_dir, 'attended_columns_{}'.format(r)), att_cols)

        # Save attention maps from best overall model
        if args.get_attention and save_att_maps:
            feat_dict = {'train': best_train_feat, 'val': best_val_feat, 'ho': ho_features}
            out_pred.save_attetnion_maps('attention_maps_{}.h5'.format(r), ho_att, ho_ds, ho_preds, ho_labels, feat_dict)

        # Save best run model
        if args.save_model:
            print('Saving model and held-out indices from run {} to {}...'.format(r, out_dir))
            np.save(os.path.join(out_dir, 'ho_indices_{}'.format(r)), ho_ds.get_set_indices())
            torch.save(best_run_model.state_dict(), os.path.join(out_dir, 'best_model_{}.pt'.format(r)))

        if args.save_training:
            feat_dict = {'train': best_train_feat, 'val': best_val_feat, 'ho': ho_features}
            out_pred.save_prediction('preds_{}.h5'.format(r), ho_ds, ho_preds, ho_labels, feat_dict)

        # Run GP on best overall model
        if args.run_gaussian > 0:
            #gp_succeed = out_pred.run_gp('gp_results_run{}.h5'.format(r), train_dict, val_dict, ho_dict, best_run_acc, r)
            gp_succeed = out_pred.run_gp('gp_results_run{}.h5'.format(r), train_dict, val_dict, ho_dict, best_run_accs, r)

        if args.run_gaussian > 0 and not gp_succeed:
            re += 1
            print(bcolors.FAIL + 'GP run failed! Rerunning NN, attempt {}/{}'.format(re + 1, args.max_nn_reruns) + bcolors.ENDC)
        else:
            r += 1
            re = 0

    assert args.run_gaussian < 1 or gp_succeed, 'GP failed at run {} after {} NN reruns'.format(r, re)

    # Save attended tracks
    if args.get_attended_tracks:
        print('Saving attended tracks to {}...'.format(out_dir))
        np.save(os.path.join(out_dir, 'attended_tracks'), np.array(track_lst))

    if args.save_training: accs_df.to_csv(os.path.join(out_dir, 'run_accuracies.csv'))

    print('Results summary for {} runs:\n {}'.format(args.nn_reruns, accs_df.describe()))

    print('Done!')


if __name__ == '__main__':
    startTime = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - startTime))
