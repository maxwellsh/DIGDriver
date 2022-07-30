#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats
import pickle as pkl
import pathlib
import pysam
# import pyBigWig
import bbi
import json
import h5py
import argparse
from types import SimpleNamespace
from random import shuffle
import time
import gc
import multiprocessing as mp
import pybedtools

# file_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(file_path)
# from mappability_tools import *
from DIGDriver.data_tools.mappability_tools import *
from DIGDriver.data_tools.mutation_tools import *

# Hyperparameters
# window = 1000
# overlap = 0
# min_map = 0.92

mut_cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'LOCATION', 'FUNCTION', 'TYPE', 'PATHO', 'SAMPLE']

# map_file_name = './high_mapp_{}'.format(window)

tr_str = 'NACGT'  # Translation string to convert letters to integers


def str2bool(v):
    """
    Convert strings with Boolean meaning to Boolean variables.
    :param v: a string to be converted.
    :return: a boolean variable that corresponds to the input string
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')


def extract_high_mappability(bw_map, chroms, window, min_map, overlap=0, file_name=None):
    """
    Extracts all regions in the genome which have average window mappability higher than the defined minimum mappability.
    :param bw_map: mappabilities BigWig file.
    :param chroms: chromosome id.
    :param window: window size.
    :param min_map: minimum mapability value.
    :param file_name: file path to save the mappability file.
    :return: a dictionary of <chromosome id: high mappability regions start indices list>
    """
    print(window, overlap)
    hi_map_dict = {}
    for chr_id, chr_size in chroms.items():
        print(chr_id)
        hi_map_dict[chr_id] = []
        i = 0
        while i + window < chr_size:
            # print(i)
            mapp = bbi.fetch(bw_map, chr_id, i, i + window)
            if np.mean(mapp) >= min_map:
                hi_map_dict[chr_id].append(i)
                #  hi_map_dict[chr_id].append(i - overlap)
            i += window - overlap
    if file_name:
        with open(file_name, 'w') as f:
            json.dump(hi_map_dict, f)
    return hi_map_dict


def fetch_mutation_region_number(tbx_SNV, chr_id, region_idx, window):
    """
    Fetch a window from mutation file and returns the number of SNPs in that region.
    :param tbx_SNV: a Tabix file with all SNPs of a given cancer type sorted by base pair location.
    :param chr_id: relevant chromosome full id (e.g. chr1).
    :param region_idx: relevant data sample start index.
    :param window: window size of data sample.
    :return: the number of mutations in a given region.
    """
    chr_num = chr_id.split('chr')[-1]
    if not chr_num.isdigit():
        raise Exception('Expected chromosome id of the form "chr#", got {}.'.format(chr_id))

    data = [row for row in tbx_SNV.fetch(chr_num, region_idx, region_idx + window, parser=pysam.asTuple())]
    return len(data)

def fetch_cnv_region_avg(tbx_CNV, chr_id, region_idx, window):
    """
    Fetch a window from mutation file and returns a triple of the mean counts of duplications,
    CNN-LOH, and deletions occuring at each position over the window
    :param tbx_SNS: a Tabix file with all copy number variants a given cancer type sorted by origin base pair start location.
    :param chr_id: relevant chromosome full id (e.g. chr1).
    :param region_idx: relevant data sample origin seq start index.
    :param window: window size of data sample.
    :return: mean of (dup, cnn-LOH, del) for each position over the window
    """
    chr_num = chr_id.split('chr')[-1]
    if not chr_num.isdigit():
        raise Exception('Expected chromosome id of the form "chr#", got {}.'.format(chr_id))

    start = region_idx
    end = region_idx + window
    data = [row for row in tbx_CNV.fetch(reference = chr_num, start=start, end = end)]
    track = np.zeros((end - start, 3), dtype=int)
    if(len(data) == 0):
        return track
    df = pd.DataFrame([row.split('\t') for row in data])
    df.columns = ['chr', 'start', 'end', 'cp_num', 'cp_Maj', 'cp_min', 'donor']
    df[["start", "end", "cp_num", "cp_Maj", "cp_min"]] = df[["start", "end", "cp_num", "cp_Maj", "cp_min"]].apply(pd.to_numeric)
    df_sort = df.sort_values(['start', 'end'])
    for index,row in df_sort.iterrows():
        srt = row['start']
        fin = row['end']
        cp_num = row['cp_num']
        #all copy regions overlap the window, so clip ends to window size
        if srt < start:
            srt = start
        if fin >= end:
            fin = -1
        #now transform to array indices
        srt = srt - start
        if fin != -1: #if CNV goes though entire window don't subtract at last position
            fin = fin - start
        if cp_num == 2:
            track[srt] =  track[srt] + [0,1,0]
            if fin != -1:
                track[fin] =  track[fin] - [0,1,0]
        if cp_num < 2:
            dels = 2 - cp_num
            track[srt] =  track[srt] + [0,0,dels]
            if fin != -1:
                track[fin] =  track[fin] - [0,0,dels]
        if cp_num > 2:
            dups = cp_num - 2
            track[srt] =  track[srt] + [dups,0,0]
            if fin != -1:
                track[fin] =  track[fin] - [dups,0,0]
    #now walk through track and fill in intermediate values
    posVec = np.cumsum(track, axis =0)
    avg = np.mean(posVec, axis=0)
    return avg

def fetch_seq(ref_festa, chr_id, start, end, bins=None):
    """
    Fetch one dimensional fasta track.
    :param ref_festa: reference genome Festa file.
    :param chr_id: relevant chromosome full id (e.g. chr1).
    :param start: start position of region.
    :param end: end position of region.
    :param bins: number of bins to average across
    :return: a 1D matrix of size end - start.
    """
    # print(chr_id, region_idx, window)
    ref_str = ref_festa.fetch(chr_id, start, end)
    ref_seq = np.array([tr_str.index(c.upper()) for c in ref_str])

    if bins != -1:
        n_pos = int(len(ref_seq) / bins)
        ref_bins = np.array([np.mean(ref_seq[i*n_pos:(i+1)*n_pos]) for i in range(bins)])

    else:
        ref_bins = ref_seq

    return ref_bins

def fetch_GC(ref_festa, chr_id, start, end, bins=None):
    """
    Fetch one dimensional fasta track.
    :param ref_festa: reference genome Festa file.
    :param chr_id: relevant chromosome full id (e.g. chr1).
    :param start: start position of region.
    :param end: end position of region.
    :param bins: number of bins to average across
    :return: a 1D matrix of size end - start with percentage GC nucleotides.
    """
    # print(chr_id, region_idx, window)
    gc_str = 'GC'
    ref_str = ref_festa.fetch(chr_id, start, end)
    ref_seq = np.array([int(c.upper() in gc_str) for c in ref_str])

    if bins != -1:
        n_pos = int(len(ref_seq) / bins)
        ref_bins = np.array([np.mean(ref_seq[i*n_pos:(i+1)*n_pos]) for i in range(bins)])

    else:
        ref_bins = ref_seq

    return ref_bins

def fetch_bw(f, chrom_idx_lst, window, bins=-1):
    print(f, window)
    res = []
    for row in chrom_idx_lst:
        # print(row)
        if (row[0] == 'chr8') & (row[1] > 146000000):
            # print(row)
            y = bbi.fetch(str(f), row[0], row[1], row[1]+window)
            z, _ , _ = scipy.stats.binned_statistic(y, y, bins=bins)
            res.append(z)

        else:
            res.append(bbi.fetch(str(f), row[0], row[1], row[1]+window, bins=bins))

    x = np.array(res)
    # x = np.array([bbi.fetch(str(f), row[0], row[1], row[1]+window, bins=bins) for row in chrom_idx_lst])
    # return x
    return (np.around(x, decimals=2)*100)


def create_split_index(map_dict, out_dir_path, chunk_size=10000, min_map=0.92, window=1000, overlap=0, shuffled=False):
    """
    Create and store the shuffled (chrom, pos) index list (one file per chunk_size)
    :param map_dict: <chromosome id: high mappability regions start indices list> dictionary
    :param out_dir_path: path to output directory
    :param chunk_size: number of data samples per chunk.
    """
    chr_idx_lst = []
    for chr_id in map_dict:
        chr_idx_lst += [(chr_id, idx) for idx in map_dict[chr_id]]
        # chr_idx_lst += [(int(chr_id.split('chr')[-1]), idx) for idx in map_dict[chr_id]]

    # print(chr_idx_lst[0:5])
    chr_idx_lst = sorted(chr_idx_lst, key=lambda item: (int(item[0].split('chr')[-1]), item[1]))
    # print(chr_idx_lst[0:5])

    if shuffled:
        shuffle(chr_idx_lst)

    idx_list = range(0, len(chr_idx_lst), chunk_size)
    for i, idx in enumerate(idx_list):
        print(i)
        # with open(os.path.join(paths.out_dir_path, '{:03d}_data_indices_{}_{}_{}.pkl'
        with open(os.path.join(out_dir_path, '{:03d}_data_indices_{}_{}_{}.pkl'
                  .format(i, window, overlap, min_map)), 'wb') as f:
            l = sorted(chr_idx_lst[idx:idx+chunk_size],
                       key=lambda item: (int(item[0].split('chr')[-1]), item[1]))
            pkl.dump(l, f)


def build_data_chunk(chr_idx_lst, epi_lst, ref_festa, window, bins, compressed=True):
    """
    Create and store a data chunk from a list of (chr ID, start) tuples
    :param chr_idx_list: (chrom ID, start) list of windows in d
    :param data: SimpleNamespace with all compressed data files.
    :param compressed: Boolean variable to indicate if data should be save compressed.
    :param chunk_count: count label to give this chunk
    """
    print(window)
    chunk_size = len(chr_idx_lst)
    if bins != -1:
        ncol = bins
    else:
        ncol = window

    if ref_festa:
        x_data = np.empty([chunk_size, ncol, len(epi_lst)+1])  # chunk_size X window_size X (epi_num + rep_num + ref)
    # x_data = np.empty([chunk_size, window, len(epi_lst) + len(repl_lst) + 1])  # chunk_size X window_size X (epi_num + rep_num + ref)
    else:
        x_data = np.empty([chunk_size, ncol, len(epi_lst)])  # chunk_size X window_size X (epi_num + rep_num + ref)

    start = 0
    end = 0

    # bw_all = epi_lst + repl_lst
    bw_all = epi_lst

    # Build epigenome arrays
    # p = mp.Pool(5)
    # a_epi = p.starmap(fetch_bw, [[f, chr_idx_lst, window, bins] for i, f in enumerate(bw_all)])
    # p.close()
    # p.join()
    a_epi = [fetch_bw(f, chr_idx_lst, window, bins) for i, f in enumerate(bw_all)]

    for i, mat in enumerate(a_epi):
        x_data[:, :, i] = mat

    ## Build ref seq array
    if ref_festa:
        ref = (np.around(np.array([fetch_seq(ref_festa, row[0], row[1], row[1]+window) for row in chr_idx_lst]), decimals=2)*100)
        x_data[:, :, -1] = ref


    return x_data


## DEPRECATED
def extract_and_store_data(map_dict, data, paths, shuffle, compress=True):
    """
    Wrapping function for different methods of data extraction and storage.
    :param map_dict: <chromosome id: high mappability regions start indices list> dictionary
    :param data: SimpleNamespace with all compressed data files.
    :param paths: SimpleNamespaces with all relevant file and directory paths.
    :param compressed: Boolean variable to indicate if data should be save compressed.
    :param chunk_size: number of data samples per chunk.
    """
    if shuffle:
        store_shuffled_data_chunks(map_dict, data, paths, compressed=compress)
    else:
        store_data_by_chromosome(map_dict, data, paths, compressed=compress)


def build_tensor(chr_idx_lst, track_lst, ref_festa):
    x_data = np.empty([len(chr_idx_lst), window, len(track_lst) + 1])  # total_window_num X window_size X (epi_num + rep_num + ref)

    # Build tensor one epigenome track at a time
    for i, track in enumerate(track_lst):
        if i % 100 == 0:
            print(i)
        x_data[:, :, i] = pd.read_table(str(track), header=None).values

    ## Build ref seq matrix
    ref = np.array([fetch_seq(ref_festa, row[0], row[1], row[2]) for row in chr_idx_lst])
    x_data[:, :, -1] = ref

    return x_data
    # return x_data


def save_tensor(f, x_data, files=[], compressed=True):
    h5f = h5py.File(str(f), 'w')
    # f= os.path.join(paths.base_dir, paths.out_dir_path, '{}_shuf_data_{}_{}_{}'
    #                              .format(chunk_count, window, overlap, min_map))

    print('Saving tensor to {}'.format(f))
    # np.save(f, x_data)
    if compressed:
        h5f.create_dataset('x_data', compression='gzip', data=x_data)
    else:
        h5f.create_dataset('x_data', data=x_data)

    if len(files) > 0:
        with open(str(f) + '.files', 'wb') as pc:
            pkl.dump(files, pc)
            # print("{}\t{}\t{}".format(row[0], row[1], row[1]+window), file=f)

    print('Tensor stored at {}.'.format(f))
    h5f.close()


def rescale_tensor(x_data):
    """
    Convert a tensor of floats to ints with 2-decimal point precision
    :param x_data: numpy tensor
    """
    return (np.around(x_data, decimals=2)*100).astype(int)

def merge_tensor_rows(x_data, merge_lst):
    x_data2 = np.empty((x_data.shape[0], x_data.shape[1], len(merge_lst)))

    for i, merge in enumerate(merge_lst):
        x_data2[:, :, i] = np.mean(x_data[:, :, merge], axis=2)

    return x_data2


# def split_data_idx(paths, args):
def split_data_idx(args):
    map_file_name = "high_mapp_{}_{}_{}".format(args.min_map, args.window, args.overlap)
    print(map_file_name)
    assert pathlib.Path(os.path.join(args.base_dir, map_file_name)).exists(),\
           'Mappability over window of {} and overlap {} not calculated. '.format(args.window, args.overlap)

    with open(os.path.join(args.base_dir, map_file_name), 'r') as f:
        map_dict = json.load(f)

    print('Number of high-mappability frames per chromosome:')
    counter = 0
    for chr_id in map_dict:
        counter += len(map_dict[chr_id])
        print('{}: {}'.format(chr_id, len(map_dict[chr_id])))
    print('Total number of high-mappability frames: {}'.format(counter))

    create_split_index(map_dict, args.out_dir, chunk_size=args.chunk_size, window=args.window, min_map=args.min_map, overlap=args.overlap, shuffled=args.shuffle)


# def create_chunk(paths, args):
def create_chunk(args):
    """ Now being more memory efficient by writing to the HDF5 file in chunks
        instead of building the whole tensor and writing once
    """
    # paths.ref_file_path = args.ref_file
    # paths.epi_dir_path = args.epi_dir

    chr_idx_lst = pkl.load(open(args.chunkIdx, 'rb'))
    # chr_idx_lst = pkl.load(open(os.path.join(paths.base_dir, args.chunkIdx), 'rb'))

    print('Loading data...')
    data = SimpleNamespace()

    if args.ref_file:
        data.rg_fasta = pysam.FastaFile(args.ref_file)  # open festa file for reading the reference genome
        depth = 2 #one spot for seq, second for gc content
    else:
        data.rg_fasta = None
        depth = 0

    data.epi_lst = sorted(pathlib.Path(args.epi_dir).glob("*.big*ig"))  # get names of all epigenomics files

    chunk_size = len(chr_idx_lst)
    depth += len(data.epi_lst)
    if args.bins == -1:
        ncol = args.window
    else:
        ncol = args.bins

    name = args.chunkIdx.split("/")[-1].replace('data_indices', 'split_data')[:-4] + '.h5'
    fname = os.path.join(args.out_dir, name)

    print("Preallocating HDF5 archive at {}".format(fname))
    fout = h5py.File(fname, 'w')
    d = fout.create_dataset('x_data', shape=(chunk_size, ncol, depth),
                            maxshape=(chunk_size, ncol, None), ## Untested line
                            dtype=float, data=None, compression='gzip')

    print('Extracting chunks from bigwig ...')
    files = []
    for i, f in enumerate(data.epi_lst):
        d[:, :, i] = fetch_bw(f, chr_idx_lst, args.window, args.bins)
        files.append(str(f))

    if data.rg_fasta:
        print('Adding sequence context tracks')
        #Add sequence track
        ref = (np.around(np.array([fetch_seq(data.rg_fasta, row[0], row[1], row[1]+args.window, bins=args.bins)
                                   for row in chr_idx_lst]), decimals=2)*100)
        d[:, :, -2] = ref
        files.append(args.ref_file)
        #add GC content track
        gc_ref = (np.around(np.array([fetch_GC(data.rg_fasta, row[0], row[1], row[1]+args.window, bins=args.bins)
                                   for row in chr_idx_lst]), decimals=2)*100)
        d[:, :, -1] = gc_ref
        files.append('GC_content')
        # files.append(os.path.join(paths.base_dir, paths.ref_file_path))

    ## Add mutation counts if necessary
    if args.mut_file:
        print('counting mutations in {}'.format(args.mut_file))
        tbx = pysam.TabixFile(args.mut_file)
        mut_counts = np.array([fetch_mutation_region_number(tbx, idx[0], idx[1], args.window)
                               for idx in chr_idx_lst])
        if args.cancer_key is None:
            cancer = args.mut_file.split('/')[-1].split('.bed.gz')[0]
        else:
            cancer = args.cancer_key
        fout.create_dataset(cancer, data=mut_counts, dtype=int)

    ## Add data indices
    idx = np.array([[int(t[0].split('chr')[-1]), t[-1], t[-1]+args.window] for t in chr_idx_lst])
    fout.create_dataset('idx', dtype=int, data=idx)

    if args.save_files == 'True':
        with open(str(fname) + '.files', 'wb') as pc:
            pkl.dump(files, pc)

    # print(data.epi_lst)
    # x_data = build_data_chunk(chr_idx_lst, data.epi_lst, data.rg_fasta, args.window, args.bins, compressed=True)
    # x_data = build_data_chunk(chr_idx_lst, data.epi_lst, data.repl_lst, data.rg_fasta, compressed=True)

    # print('Saving chunk...')
    # # count = args.chunkIdx.split('/')[-1].split('_')[0]
    # # suffix = args.chunkIdx.split('ces_')[-1].split('.pkl')[0]
    # name = args.chunkIdx.replace('data_indices', 'shuf_data')[:-4]
    # fout = os.path.join(paths.base_dir, name)
    # # print(fout)
    # save_tensor(fout, x_data, files=np.array([strza(f) for f in data.epi_lst]))


def rescale(paths, args):

    for f in args.tensor:
        print('Rescaling: {}'.format(f))
        fout = os.path.join(args.base_dir, args.out_dir, f.split('/')[-1]+"_int")
        x_data = h5py.File(f, 'r')['x_data']
        x_data2 = rescale_tensor(x_data)

        save_tensor(fout, x_data2)
        del x_data, x_data2


# def mappability(paths, args):
def mappability(args):

    map_file_name = "high_mapp_{}_{}_{}".format(args.min_map, args.window, args.overlap)
    print(map_file_name)

    if not pathlib.Path(os.path.join(args.out_dir, map_file_name)).exists():  # if first run with given window and overlap
        print('Computing mappability over window of {} with min map {} and overlap {}'.format(args.window, args.min_map, args.overlap))
        bw_map = args.map_file
        chroms = bbi.chromsizes(bw_map)  # get chromosomes length dictionary
        chroms.pop('chrM')
        chroms.pop('chrX')
        chroms.pop('chrY')
        map_dict = extract_high_mappability(bw_map, chroms, args.window, args.min_map, file_name=os.path.join(args.out_dir, map_file_name), overlap=args.overlap)


## DEPRECATED
def count_mutations(paths, args):
    chr_idx_lst = pkl.load(open(os.path.join(paths.base_dir, args.chunkIdx), 'rb'))
    print(args.mut_file)
    tbx = pysam.TabixFile(os.path.join(paths.base_dir, args.mut_file))
    mut_counts = np.array([fetch_mutation_region_number(tbx, idx[0], idx[1], args.window)
                           for idx in chr_idx_lst])

    # print(len(mut_counts))
    cancer = args.mut_file.split('/')[-1].split('_')[0]
    fout = os.path.join(args.out_dir, args.chunkIdx.split('/')[-1].replace('data_indices', cancer))

    with open(fout, 'wb') as f:
        pkl.dump(mut_counts, f)

# def add_objectives(paths, args):
def add_objectives(args):
    ## Change to take index from h5
    # chr_idx_lst = pkl.load(open(os.path.join(paths.base_dir, args.chunkIdx), 'rb'))
    f = h5py.File(args.h5_file, 'r+')
    chr_idx_lst = f['idx']
    window = chr_idx_lst[0, 2] - chr_idx_lst[0, 1]
    print(args.mut_file)

    ## Change to absolute path
    # tbx = pysam.TabixFile(os.path.join(paths.base_dir, args.mut_file))
    if args.cnv:
        print('Adding CNV counts from {} to {}'.format(args.mut_file, args.h5_file))
        tbx = pysam.TabixFile(args.mut_file)
        mut_counts = np.array([fetch_cnv_region_avg(tbx, str(idx[0]), idx[1], window)
                           for idx in chr_idx_lst])
    else:
        print('Adding mutation counts from {} to {}'.format(args.mut_file, args.h5_file))
        idx = f['idx'][:]
        df_idx = pd.DataFrame(idx, columns=['CHROM', 'START', 'END'])
        df_idx['ELT'] = ['{}:{}-{}'.format(row[0], row[1], row[2]) for row in idx]
        bed_idx = pybedtools.BedTool.from_dataframe(df_idx)
        df_mut = tabulate_muts_per_sample_per_element(args.mut_file, bed_idx.fn, 
            bed12=False, drop_duplicates=True)

        if args.max_muts_per_elt_per_sample:
            df_mut = cap_muts_per_element_per_sample(df_mut, 
                args.max_muts_per_elt_per_sample)

        if args.sample_filter_stdev:
            df_mut = filter_samples_by_stdev(df_mut, args.sample_filter_stdev)

        if args.max_muts_per_sample:
            df_mut = filter_hypermut_samples(df_mut, args.max_muts_per_sample)

        df_elt = df_mut.pivot_table(index='ELT', values='OBS_SNV', aggfunc=np.sum)
        df_cnt = df_idx.merge(df_elt, on='ELT', how='left')
        df_cnt.loc[df_cnt.OBS_SNV.isna(), 'OBS_SNV'] = 0
        mut_counts = df_cnt.OBS_SNV.astype(int)

        # print(df_mut[df_mut.OBS_MUT > 100])
        # mut_counts = np.array([fetch_mutation_region_number(tbx, str(idx[0]), idx[1], window)
        #                    for idx in chr_idx_lst])
        # df_mut = read_mutation_file(args.mut_file, drop_sex=True, drop_duplicates=True)

    cancer = args.mut_file.split('/')[-1].split('.annot')[0].split('.txt')[0].split('.bed')[0] + args.suffix
    print('Saving dataset as {}'.format(cancer))
    f.create_dataset(cancer, data = mut_counts, dtype = float)
    f.close()

def merge_rows(path, args):
    for tensor in args.tensor:
        print(tensor)
        fout = os.path.join(tensor+"_merged")
        x_data = h5py.File(tensor, 'r')['x_data'][:]
        files = pkl.load(open(tensor+'.files', 'rb'))

        track_types = np.unique([f.split('/')[-1].split('-')[-1].split('.pval')[0] for f in files])
        merge_lst = [[i for i, f in enumerate(files) if track in f] for track in track_types]
        print(list(zip(track_types, merge_lst)))

        x_data2 = merge_tensor_rows(x_data, merge_lst)

        save_tensor(fout, x_data2, track_types)

# def concatH5(path, args):
def concatH5(args):
    h5_files = sorted(args.dir.glob('*split_data*.h5'))
    h5_list = [h5py.File(str(f)) for f in h5_files]

    key_list = [set([key for key in h5.keys()]) for h5 in h5_list]
    keys = set.intersection(*key_list)
    keys_all = set.union(*key_list)

    if keys_all - keys:
        print('WARNING: the following archives are not in all h5 files: ', keys_all - keys)

    print('The following archives will be merged: ', keys)

    keys = keys - set(['idx', 'x_data'])

    ## Merge x_data
    _, n_pos, n_tracks = h5_list[0]['x_data'].shape
    n_rows = sum([h5['x_data'].shape[0] for h5 in h5_list])

    suffix = h5_files[0].name.split('.h5')[0].split('split_data')[-1]
    fname = os.path.join(args.out_dir, 'data_matrices' + suffix + '.h5')

    ## TODO: add user-defined compression option
    print("Preallocating HDF5 archive at {}".format(fname))
    fout = h5py.File(fname, 'w')
    d = fout.create_dataset('x_data', shape=(n_rows, n_pos, n_tracks),
                            maxshape=(n_rows, n_pos, None), ## Untested line
                            dtype=float, data=None, compression='gzip')

    print('Merging x_data')
    i = 0
    for h5 in h5_list:
        size = h5['x_data'].shape[0]
        d[i:i+size, :, :] = h5['x_data']
        i += size

    print('Merging idx')
    idx_list = [h5['idx'][:] for h5 in h5_list]
    idx = np.concatenate(idx_list)
    fout.create_dataset('idx', dtype=int, data=idx, compression='gzip')

    for key in keys:
        print('Merging {}'.format(key))
        np_lst = [h5[key][:] for h5 in h5_list]
        nd = np.concatenate(np_lst)
        fout.create_dataset(key, dtype=int, data=nd, compression='gzip')

def add_mappability(args):
    f = h5py.File(args.h5_file, 'r+')
    idx = f['idx'][:]

    print('Adding mappability from {} to {}'.format(args.map_file, args.h5_file))
    mapp_res = mappability_by_idx(args.map_file, idx)
    mapp = np.array([w[-1] for w in mapp_res])

    print('Saving dataset as mappability')
    f.create_dataset('mappability', data=mapp, dtype=float)
    f.close()

def add_tracks(args):
    f_old = h5py.File(args.h5, 'a')
    idx = f_old['idx'][:]

    print("Loading index")
    chr_idx_lst = [('chr{}'.format(row[0]), row[1]) for row in idx]
    window = idx[0, 2] - idx[0, 1]

    if args.track_file:
        tracks = [line.strip() for line in open(args.track_file, 'r')]
    else:
        tracks = args.tracks

    shape_old = f_old['x_data'].shape
    shape_new = (shape_old[0], shape_old[1], shape_old[2]+len(tracks))
    bins = shape_old[1]
    print(shape_new)

    if args.inplace:
        try:
            d = f_old['x_data']
            d.resize(shape_new)

        except ValueError:
            print('ERROR: cannot extend dataset. Please rerun without --inplace.')
            return 1

    else:
        print('Copying original tracks to new archive. This might take awhile...')
        compress = None
        if args.compress:
            compress = 'gzip'

        f_new = h5py.File(args.out_file, 'w')
        d = f_new.create_dataset('x_data', shape=shape_new, 
                                 maxshape=(shape_new[0], shape_new[1], None),
                                 dtype=float, data=None, compression=compress)
                                 # dtype=np.float32, data=None, compression=compress)

        chunksize = 8192
        n_chunks = int(np.ceil(shape_old[0] / chunksize))

        for i in range(int(n_chunks)):
            print('copying chunk {} / {}'.format(i, n_chunks))
            data = f_old['x_data'][i*chunksize:(i+1)*chunksize, :, :]
            d[i*chunksize:(i+1)*chunksize, :, :shape_old[2]] = data

        # d[shape_old] = f_old['x_data'][:]

        print('Copying additional data to new archive')
        for key in f_old.keys():
            if key != "x_data":
                f_old.copy(key, f_new)

    print('Adding new track(s) to archive')
    for i, f in enumerate(tracks):
        d[:, :, shape_old[2]+i] = fetch_bw(f, chr_idx_lst, window, bins)
        # files.append(str(f))

    f_track_orig = args.h5.split('.h5')[0] + '.files'
    try:
        with open(f_track_orig, 'rb') as f:
            tracks_in = pkl.load(f)
    except FileNotFoundError:
        print("WARNING: file of track names not found. Not writing new track names to a file")
    else:
        if args.inplace:
            f_track_out = f_track_orig
        else:
            f_track_out = args.out_file.split('.h5')[0] + '.files'

    tracks_all = tracks_in + tracks
    with open(f_track_out, 'wb') as f_out:
        pkl.dump(tracks_all, f_out)

def unzip_h5(args):
    
    zipped_file_path = args.h5

    print('Opening zipped h5 file...')
    zipped_h5f = h5py.File(zipped_file_path, 'r')

    split_path = zipped_file_path.split('.h5')
    unzipped_file_path = split_path[0] + ".unzipped.h5"
    unzipped_h5f = h5py.File(unzipped_file_path, 'w')

    print('Loading unzipped data to {}...'.format(unzipped_file_path))
    for k in zipped_h5f.keys():
        print('Unzipping {}'.format(k))
        if k == 'x_data':
            shape = zipped_h5f['x_data'].shape
            d = unzipped_h5f.create_dataset('x_data', shape=shape, 
                                     maxshape=(shape[0], shape[1], None),
                                     dtype=float, data=None)
                                     # dtype=np.float32, data=None)
            chunksize = 8192
            n_chunks = int(np.ceil(shape[0] / chunksize))

            for i in range(int(n_chunks)):
                print('Unzipping chunk {} / {}'.format(i, n_chunks))
                data = zipped_h5f['x_data'][i*chunksize:(i+1)*chunksize, :, :]
                d[i*chunksize:(i+1)*chunksize, :, :] = data
        
        else:
            unzipped_h5f[k] = zipped_h5f[k][:]

def create_mean_predictors(args):
    f = h5py.File(args.h5, 'a')

    shape = f['x_data'].shape
    n_regions = shape[0]
    n_tracks = shape[2]
    # chunk_idx = np.concatenate(np.arange(0, n_regions, args.chunksize), [n_regions])
    n_chunks = np.ceil(n_regions / args.chunk_size)

    ## Preallocated space in h5 archive
    try:
        d_mean = f.create_dataset('x_mean', shape=(n_regions, n_tracks),
                                  maxshape=(n_regions, None),
                                  dtype=float, data=None)
    except RuntimeError:
        d_mean = f['x_mean']
        d_mean.resize((n_regions, n_tracks))

    print('Creating mean predictors')
    for i in range(int(n_chunks)):
        print("Loading chunk {} / {}".format(i, n_chunks))
        data = f['x_data'][i*args.chunk_size:(i+1)*args.chunk_size, :, :]
        d_mean[i*args.chunk_size:(i+1)*args.chunk_size, :] = np.mean(data, axis=1)

def parse_args(text=None):
    parser = argparse.ArgumentParser(description="CLI tools for (epi)genome preprocessing for CNNs")
    subparsers = parser.add_subparsers(help='CLI commands')

    parser_0 = subparsers.add_parser('mappability', help='Determine high map regions')
    # parser_0.add_argument('--base-dir', type=str, default='.', help='base directory of analysis')
    parser_0.add_argument('map_file', type=str, help='path to mappability file')
    parser_0.add_argument('--out-dir', type=str, default='.', help='Output directory relative to base-dir')
    parser_0.add_argument('--window', type=int, default=10000, help='window size for mappability')
    parser_0.add_argument('--overlap', type=int, default=0, help='base pairs of overlap for contiguous windows')
    parser_0.add_argument('--min-map', type=float, default=0.80, help='minimum mappability for window')
    parser_0.set_defaults(func=mappability)

    parser_a = subparsers.add_parser('splitDataIdx', help='Split data window indeces into chunks')
    parser_a.add_argument('--base-dir', type=str, default='.', help='Directory of mappability file')
    # parser_a.add_argument('--chr-map-file', type=str, default='.', help='chromosome mappability file')
    parser_a.add_argument('--out-dir', type=str, default='sorted_data', help='Output directory relative')
    parser_a.add_argument('--chunk-size', type=int, default=10000, help='chunk size for data')
    parser_a.add_argument('--window', type=int, default=10000, help='window size for mappability')
    parser_a.add_argument('--overlap', type=int, default=0, help='base pairs of overlap for contiguous windows')
    parser_a.add_argument('--min-map', type=float, default=0.80, help='minimum mappability for window')
    parser_a.add_argument('--shuffle', action='store_true', default=False, help='randomly shuffle data indeces')
    parser_a.set_defaults(func=split_data_idx)

    parser_b = subparsers.add_parser('createChunk', help='Build data chunk from saved index')
    parser_b.add_argument('chunkIdx', help='path to indeces for chunk windows relative to base-dir')
    # parser_b.add_argument('--base-dir', type=str, default='.', help='base directory of analysis')
    parser_b.add_argument('--out-dir', type=str, default='sorted_data', help='Output directory relative to base-dir')
    parser_b.add_argument('--ref-file', type=str, default=None, help='path to ref genome relative to base-dir')
    parser_b.add_argument('--epi-dir', type=str, default='epigenomes/raw', help='path to epigenomes dir relative to base-dir')
    parser_b.add_argument('--mut-file', type=str, default=None, help='path to file of mutations for a particular cancer')
    # parser_b.add_argument('--repli-dir', type=str, default='replication_timing/', help='path to replication time dir relative to base-dir')
    parser_b.add_argument('--window', type=int, default=10000, help='window size for mappability')
    parser_b.add_argument('--bins', type=int, default=-1, help='bins to use for coarsegraining')
    parser_b.add_argument('--save-files', type=str, default='True', help='save file of track extraction')
    parser_b.add_argument('--cancer-key', type=str, dest='cancer_key', help='key for saving cancer mut targets')
    # parser_b.add_argument('--min-map', type=float, default=0.92, help='minimum mappability for window')
    parser_b.set_defaults(func=create_chunk)

    ## DEPRECATED
    parser_c = subparsers.add_parser('rescaleTensor', help='rescale an h5f tensor of floats to ints')
    parser_c.add_argument('tensor', nargs='+', help='path(s) to tensors to rescale relative to base-dir')
    parser_c.add_argument('--base-dir', type=str, default='.', help='base directory of analysis')
    parser_c.add_argument('--out-dir', type=str, default='sorted_data_1k', help='Output directory relative to base-dir')
    parser_c.set_defaults(func=rescale)

    ## DEPRECATED
    parser_d = subparsers.add_parser('countMutations', help='count mutations for a cancer type in windows')
    parser_d.add_argument('mut_file', help='path to file of mutations')
    parser_d.add_argument('chunkIdx', help='path to indeces for chunk windows relative to base-dir')
    parser_d.add_argument('--base-dir', type=str, default='.', help='base directory of analysis')
    parser_d.add_argument('--out-dir', type=str, default='sorted_data', help='Output directory relative to base-dir')
    parser_d.add_argument('--window', type=int, default=1000, help='window size for mappability')
    parser_d.set_defaults(func=count_mutations)

    parser_e = subparsers.add_parser('mergeTracks', help='merge tensor rows of same epigenome tracks')
    parser_e.add_argument('tensor', nargs="+", help='path to tensor')
    # parser_e.add_argument('chunkIdx', help='path to indeces for chunk windows relative to base-dir')
    parser_e.add_argument('--base-dir', type=str, default='.', help='base directory of analysis')
    parser_e.add_argument('--out-dir', type=str, default='sorted_data', help='Output directory relative to base-dir')
    # parser_e.add_argument('--window', type=int, default=1000, help='window size for mappability')
    parser_e.set_defaults(func=merge_rows)

    parser_f = subparsers.add_parser('concatH5', help='concatenate h5 archives of data matrices')
    parser_f.add_argument('dir', type=pathlib.Path, help='Directory in which to find h5 files')
    # parser_f.add_argument('--base-dir', type=str, default='.', help='base directory of analysis')
    parser_f.add_argument('--out-dir', type=str, default='.', help='Output directory')
    # parser_e.add_argument('--window', type=int, default=1000, help='window size for mappability')
    parser_f.set_defaults(func=concatH5)

    parser_g = subparsers.add_parser('addObjectives', help='count mutations (SNVs or CNVs) for a cancer type to an hd5 dataset.')
    parser_g.add_argument('h5_file',  help='path to h5 file')
    parser_g.add_argument('mut_file', help='path to file of mutations')
    parser_g.add_argument('--max-muts-per-sample', type=int, default=None, help='Maximum mutations allowed per sample. Samples with higher mutation counts are removed.')
    parser_g.add_argument('--sample-filter-stdev', type=float, default=None, help='Remove samples with # mutations > filter-stdev * stdev of mutation counts across cohort.')
    parser_g.add_argument('--max-muts-per-elt-per-sample', type=int, default=None, help='Cap the number of mutations a sample can contribute to any one window.')
    parser_g.add_argument('--suffix', type=str, default='', help='suffix to add to end of cancer name when saving mutation counts to h5 archive.')
    # parser_g.add_argument('chunkIdx', help='path to indeces for chunk windows relative to base-dir')
    # parser_g.add_argument('--out-dir', type=str, default='sorted_data', help='Output directory')
    # parser_g.add_argument('--base-dir', type=str, default='.', help='base directory of analysis')
    parser_g.add_argument('--cnv', help='designates the mut file is of CNVs', action = 'store_true')
    # parser_g.add_argument('--window', type=int, default=1000, help='window size for mappability')
    parser_g.set_defaults(func=add_objectives)

    parser_h = subparsers.add_parser('addMappability', help='add mappability information for each window in an hd5 dataset.')
    parser_h.add_argument('h5_file',  help='path to h5 file')
    parser_h.add_argument('map_file', help='path to mappability file')
    parser_h.set_defaults(func=add_mappability)

    ## Add tracks
    parser_i = subparsers.add_parser('addTracks', help='Add new track from a bigwig file.\nWARNING: creates a new h5 archive to avoid destructive operations.')
    parser_i.add_argument('--h5', required=True, help='path to h5 containing data to be augmented')
    parser_i.add_argument('--tracks', type=str, nargs="+", default=[], help='Track(s) to add as a space separated list')
    parser_i.add_argument('--track-file', type=str, default="", help='File of tracks to add, one per line. Supersedes --tracks.')
    parser_i.add_argument('--inplace', default=False, action='store_true', help='Add tracks to the existing h5 archive. WARNING: may not always be possible; if you get an error, don\'t run without this flag.')
    parser_i.add_argument('--out-file', type=str, help='output file name if not altering in place')
    parser_i.add_argument('--compress', default=False, action='store_true', help='Compress the archive if not altering in place?')
    parser_i.add_argument('--remove_old', action="store_true", default='False', help='Remove original h5 archive if not altering in place?')
    parser_i.set_defaults(func=add_tracks)

    ## Unzip h5 archive
    parser_j = subparsers.add_parser('unzipH5', help='Decompress an H5 file for faster reading.')
    parser_j.add_argument('h5', help='path to h5 to be decompressed')
    parser_j.set_defaults(func=unzip_h5)

    ## Mean vectors
    parser_k = subparsers.add_parser('createMeanPred', help='Create a matrix of mean vectors from predictor matrices.')
    parser_k.add_argument('h5', help='path to h5 archive')
    parser_k.add_argument('--chunk-size', type=int, default=8192, help='chunksize for data extraction')
    parser_k.set_defaults(func=create_mean_predictors)

    if text:
        args = parser.parse_args(text.split())
    else:
        args = parser.parse_args()

    return args


def main():
    # if len(sys.argv) < 7:
    #     raise Exception('Expected 6 input parameters, got {}. '
    #                     '\nUsage: <base dir> <SNVs file path> <reference gen. file path> <mappability file path> '
    #                     '<epigenomics dir path> <replication dir path>'.format(len(sys.argv) - 1))

    args = parse_args()

    ## This should be removed
    # paths = SimpleNamespace()
    # paths.base_dir = args.base_dir
    # paths.out_dir_path = args.out_dir

    val = args.func(args)
    # args.func(paths, args)

    # paths.base_dir = sys.argv[1]
    # paths.snv_file_path = sys.argv[2]
    # paths.ref_file_path = sys.argv[3]
    # paths.map_file_path = sys.argv[4]
    # paths.epi_dir_path = sys.argv[5]
    # paths.repl_dir_path = sys.argv[6]
    # shuffle_data = str2bool(sys.argv[7])
    # paths.out_dir_path = 'sorted_data'
    # paths.idx_dir_path = 'sorted_idx'

    # print('Loading data...')
    # data = SimpleNamespace()
    # data.tbx_snv = pysam.TabixFile(os.path.join(paths.base_dir, paths.snv_file_path))  # open tabix file for reading SNVs data
    # data.rg_fasta = pysam.FastaFile(os.path.join(paths.base_dir, paths.ref_file_path))  # open festa file for reading the reference genome
    # data.epi_lst = sorted(pathlib.Path(os.path.join(paths.base_dir, paths.epi_dir_path)).glob("*comp.bw"))  # get names of all epigenomics files
    # data.epi_files = [pyBigWig.open(str(f)) for f in data.epi_lst]
    # data.repl_lst = sorted(pathlib.Path(os.path.join(paths.base_dir, paths.repl_dir_path)).glob("*.bigWig"))  # get names of all replication files
    # data.repl_files = [pyBigWig.open(str(f)) for f in data.repl_lst]
    # data.chroms = data.epi_files[0].chroms()  # get chromosomes length dictionary
    # data.chroms.pop('chrM')
    # data.chroms.pop('chrX')
    # if not pathlib.Path(os.path.join(paths.base_dir, map_file_name)).exists():  # if first run with given window and overlap
    #     raise InputError 'Mappability over window of {} and overlap {} not calculated. '.format(window, overlap))
        # bw_map = pyBigWig.open(os.path.join(paths.base_dir, paths.map_file_path))  # open BigWig file to read the mapability data
        # map_dict = extract_high_mappability(bw_map, data.chroms, window, min_map, file_name=os.path.join(paths.base_dir, map_file_name))
    # else:
    #     with open(os.path.join(paths.base_dir, map_file_name), 'r') as f:
    #         map_dict = json.load(f)

    # print('Number of high-mappability frames per chromosome:')
    # counter = 0
    # for chr_id in map_dict:
    #     counter += len(map_dict[chr_id])
    #     print('{}: {}'.format(chr_id, len(map_dict[chr_id])))
    # print('Total number of high-mappability frames: {}'.format(counter))

    # extract_and_store_data(map_dict, data, paths, shuffle_data)
    # create_shuffle_index(map_dict, paths)

    if not val:
        print('Done!')


if __name__ == '__main__':
    main()
