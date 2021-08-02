import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import math

from mut_dataset import *


class BaseDatasetGenerator:
    predictor_ds_id = 'x_data'
    index_ds_id = 'idx'
    mappability_ds_id = 'mappability'

    def __init__(self, args):
        print('Loading data and labels from file {}...'.format(args.data_file))
        self.file_path = args.data_file
        self.label_ids = args.label_ids
        self.val_ratio = args.val_ratio
        self.autoregressive_size = args.autoregressive_size
        self.dataset_func = AutoregressiveDatasetFromH5 if self.autoregressive_size > 0 else LazyLoadDatasetFromH5
        h5f = h5py.File(self.file_path, 'r')
        self.genome_locations = h5f[self.index_ds_id][:]
        self.mappability = h5f[self.mappability_ds_id][:]
        self.labels_lst = [h5f[l][:] for l in self.label_ids]
        self.quantiles = stats.mstats.rankdata(self.labels_lst[0]) / len(self.labels_lst[0])
        if self.mappability_ds_id in h5f.keys():
            #self.idxs = np.where(self.mappability >= args.mappability)[0]

            low_map_regions = self.mappability < args.mappability
            high_count_regions = self.labels_lst[0] > np.quantile(self.labels_lst[0], args.count_quantile)
            self.idxs = np.where(~low_map_regions & ~high_count_regions)[0]

            print('Total #regions: {}, #low mappability regions: {}, #high count and high mappability regions: {}' \
                  .format(len(self.labels_lst[0]), len(np.where(low_map_regions)[0]), len(h5f[self.label_ids[0]]) - len(self.idxs) - len(np.where(low_map_regions)[0])))
            print('Maximum region count pre-reduction: {}, maximum region count post-reduction: {}' \
                  .format(max(self.labels_lst[0]), max(self.labels_lst[0][self.idxs])))

            self.below_mapp = np.where(low_map_regions | high_count_regions)[0]
        else:
            self.idxs = np.arange(len(self.genome_locations))
            self.below_mapp = np.zeros(0)

        if args.track_file is not None:
            self.selected_tracks = self.load_track_selection_file(os.path.join(os.path.dirname(__file__), args.track_file))
        else:
            self.selected_tracks = np.arange(h5f[self.predictor_ds_id].shape[2])
        print('Input data is of size: {}'
              .format((len(self.idxs), h5f[self.predictor_ds_id].shape[1], len(self.selected_tracks))))

    @staticmethod
    def tokens_match(strg, search=re.compile(r'[^:0-9]').search):
        return not bool(search(strg))

    def load_track_selection_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        track_lst = []
        for i, l in enumerate(lines):
            if l.startswith(('\n', '#')): continue
            l = l.rstrip()  # remove trailing '\n'
            assert self.tokens_match(l), \
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

    @staticmethod
    def split_randomly(idxs, test_ratio, set_str):
        tmp_idxs = idxs
        split_idx = int((1 - test_ratio) * len(tmp_idxs))
        print('Splitting {} data at random to {} and {} samples'.format(set_str, split_idx, len(tmp_idxs) - split_idx))
        np.random.shuffle(tmp_idxs)
        return tmp_idxs[:split_idx], tmp_idxs[split_idx:len(tmp_idxs)]

    def split_by_chromosome(self, idxs, test_ratio, set_str):
        print('Splitting {} data by chromosome...'.format(set_str))
        chrs = self.genome_locations[idxs, 0]
        max_chr = np.max(chrs)
        train_idxs_lst = []
        test_idxs_lst = []
        for c in range(max_chr):
            print('Chromosome {}...'.format(c + 1))
            chr_idxs = np.where(chrs == c + 1)[0]
            split_idx = int((1 - test_ratio) * len(chr_idxs))
            train_idxs_lst.extend(list(chr_idxs[:split_idx]))
            test_idxs_lst.extend(list(chr_idxs[split_idx:len(chr_idxs)]))
        return np.sort(train_idxs_lst), np.sort(test_idxs_lst)

    def get_heldout_dataset(self):
        return self.dataset_func(self.file_path,
                                 self.label_ids,
                                 self.heldout_idxs,
                                 self.genome_locations,
                                 self.mappability,
                                 self.quantiles,
                                 self.selected_tracks,
                                 self.predictor_ds_id,
                                 self.autoregressive_size)


class DatasetGenerator(BaseDatasetGenerator):
    predictor_ds_id = 'x_data'
    index_ds_id = 'idx'

    def __init__(self, args):
        super(DatasetGenerator, self).__init__(args)

        if args.heldout_file is not None:
            print('Using predefined held-out samples from {}'.format(args.heldout_file))
            self.heldout_idxs = self.extract_heldout_set(args.heldout_file)
        else:
            if args.split_method == 'random':
                self.idxs, self.heldout_idxs = self.split_randomly(self.idxs, args.heldout_ratio, 'held-out')
            elif args.split_method == 'chr':
                self.idxs, self.heldout_idxs = self.split_by_chromosome(self.idxs, args.heldout_ratio, 'held-out')
            else:
                raise Exception('Expected split_method to be \'random\' or \'chr\', but found {}'.format(args.split_method))


    def extract_heldout_set(self, heldout_path):
        cols=['CHROM', 'START', 'END', 'Y_TRUE', 'Y_PRED', 'STD', 'PVAL', 'RANK']
        with open(heldout_path, 'r') as f:
            heldout_chr_loc_df = pd.DataFrame([l.split('\t') for l in f.read().split('\n')], columns=cols) \
                                   .drop(0).reset_index(drop=True).dropna(axis=0, how='any')
        origin_chr_loc_df = pd.DataFrame(self.genome_locations, columns=cols[:3])
        origin_chr_loc_df['Y_TRUE'] = self.labels_lst[0]
        set_origin_idxs = []
        for i in heldout_chr_loc_df.index:
            row = heldout_chr_loc_df.loc[i]
            origin_idx = origin_chr_loc_df.index[np.where((origin_chr_loc_df.CHROM == int(row.CHROM)) & \
                                                          (origin_chr_loc_df.START == int(row.START)))[0]]
            assert len(origin_idx) == 1, 'Found {} matches for location {}'.format(len(origin_idx), row)
            origin_idx = origin_idx[0]
            assert float(row.Y_TRUE) == float(origin_chr_loc_df.Y_TRUE[origin_idx]), \
                'Mismatch of ground truth mutation count. Expected {}, but found {}.' \
                .format(row.Y_TRUE, origin_chr_loc_df.Y_TRUE[origin_idx])
            assert len(np.where(self.idxs == origin_idx)[0]) != 0, \
                'Expected the following to be in the data set, but wasn\'t found \n{}'.format(row)
            assert len(np.where(self.idxs == origin_idx)[0]) == 1, 'Expected index {} to appear once, but found {}' \
                .format(origin_idx, len(np.where(self.idxs == origin_idx)[0]))
            self.idxs = np.delete(self.idxs, np.where(self.idxs == origin_idx)[0])
            set_origin_idxs.append(origin_idx)
        print('Heldout {} windows.'.format(len(set_origin_idxs)))
        return set_origin_idxs

    def get_datasets(self, split_method='random', test_ratio=0.2):
        if split_method == 'random':
            train_idxs, test_idxs = self.split_randomly(self.idxs, self.val_ratio, 'validation')
        elif split_method == 'chr':
            train_idxs, test_idxs = self.split_by_chromosome(self.idxs, self.val_ratio, 'validation')
        else:
            raise Exception('Expected split_method to be \'random\' or \'chr\', but found {}'.format(split_method))

        train_ds = self.dataset_func(self.file_path,
                                     self.label_ids,
                                     train_idxs,
                                     self.genome_locations,
                                     self.mappability,
                                     self.quantiles,
                                     self.selected_tracks,
                                     self.predictor_ds_id,
                                     self.autoregressive_size)
        test_ds = self.dataset_func(self.file_path,
                                    self.label_ids,
                                    test_idxs,
                                    self.genome_locations,
                                    self.mappability,
                                    self.quantiles,
                                    self.selected_tracks,
                                    self.predictor_ds_id,
                                    self.autoregressive_size)
        return train_ds, test_ds


class KFoldDatasetGenerator(BaseDatasetGenerator):
    predictor_ds_id = 'x_data'
    index_ds_id = 'idx'

    def __init__(self, args):
        super(KFoldDatasetGenerator, self).__init__(args)
        self.k = args.k

        self.divide_datasets(args.split_method,  args.k)

    def divide_datasets(self, split_method, kfold, resample=0):
        assert type(kfold) is int, 'Expected type of hyperparameter \'kfold\' to be int but found {}'.format(type(kfold))
        if split_method == 'random':
            self.ds_idxs = self.split_folds_randomly(kfold)
        elif split_method == 'chr':
            self.ds_idxs = self.split_folds_by_chromosome(kfold)
        else:
            raise Exception('Expected split_method to be \'random\' or \'chr\', but found {}'.format(split_method))

    def split_folds_randomly(self, k):
        print('Splitting data to {} parts at random...'.format(k))
        tmp_idxs = self.idxs
        set_size = len(tmp_idxs) / k
        np.random.shuffle(tmp_idxs)
        return np.array([tmp_idxs[math.floor(i*set_size):math.floor((i+1)*set_size)] for i in range(k)])

    def split_folds_by_chromosome(self, k):
        print('Splitting data to {} parts by chromosome...'.format(k))
        chrs = self.genome_locations[:, 0]
        max_chr = np.max(chrs)
        datasets_idxs_lst = [[] for _ in range(k)]
        for c in range(max_chr):
            print('Chromosome {}...'.format(c + 1))
            chr_idxs = np.where(chrs == c + 1)[0]
            set_size = int(len(chr_idxs) / k)
            [datasets_idxs_lst[i].extend(list(chr_idxs[i*set_size:(i+1)*set_size])) for i in range(k)]
        return np.array([np.sort(datasets_idxs_lst[i]) for i in range(k)])

    def get_datasets(self, fold_idx):
        heldout_idxs = self.ds_idxs[fold_idx]
        train_idxs = np.concatenate(self.ds_idxs[np.delete(np.arange(self.k), fold_idx)])
        #train_idxs, heldout_idxs = self.extract_heldout_random(train_idxs)
        train_idxs, val_idxs = self.split_randomly(train_idxs, self.val_ratio, 'validation')

        ho_ds = self.dataset_func(self.file_path,
                                  self.label_ids,
                                  heldout_idxs,
                                  self.genome_locations,
                                  self.mappability,
                                  self.quantiles,
                                  self.selected_tracks,
                                  self.predictor_ds_id,
                                  self.autoregressive_size)
        val_ds = self.dataset_func(self.file_path,
                                   self.label_ids,
                                   val_idxs,
                                   self.genome_locations,
                                   self.mappability,
                                   self.quantiles,
                                   self.selected_tracks,
                                   self.predictor_ds_id,
                                   self.autoregressive_size)
        train_ds = self.dataset_func(self.file_path,
                                     self.label_ids,
                                     train_idxs,
                                     self.genome_locations,
                                     self.mappability,
                                     self.quantiles,
                                     self.selected_tracks,
                                     self.predictor_ds_id,
                                     self.autoregressive_size)

        return train_ds, val_ds, ho_ds

    def get_below_mapp(self):
        #return LazyLoadDatasetFromH5(self.file_path, self.label_ids, self.below_mapp, self.genome_locations[self.below_mapp], self.selected_tracks, self.predictor_ds_id)
        return self.dataset_func(self.file_path,
                                 self.label_ids,
                                 self.below_mapp,
                                 self.genome_locations,
                                 self.mappability,
                                 self.quantiles,
                                 self.selected_tracks,
                                 self.predictor_ds_id,
                                 self.autoregressive_size)
