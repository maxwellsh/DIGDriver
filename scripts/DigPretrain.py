#!/usr/bin/env python

'''
Top-level script to create a pre-trained DIG model

TODO:
    * CNN+GP kfold run to regional reate estimates
    * Pre-computed sequence counts
    * Pre-computed sequence model
    * Run CNN+GP kfold
    * Run all elements of pre-training sequentially
'''

import argparse
import os
import sys
import pathlib
import h5py
import scipy.stats
import pandas as pd
import numpy as np
import pkg_resources

from DIGDriver.region_model import region_model_tools
from DIGDriver.data_tools import mutation_tools
from DIGDriver.sequence_model import sequence_tools
from DIGDriver.sequence_model import genic_driver_tools
from DIGDriver.sequence_model import nb_model
from DIGDriver.auxilaries import utils

def pretrain_region_model(args):
    """ MS Notes
        * mapp_thresh should no longer be necessary since CNN+GP submapp bug was fixed
    """
    ## Check I/O paths
    kfold_path = pathlib.Path(args.kfold_dir)
    assert kfold_path.is_dir(), \
        "The supplied kfold results path {} is not a directory.".format(args.kfold_dir)

    output_path = pathlib.Path(args.outputFile)
    if not output_path.parent.is_dir():
        print('Making output directory {}.'.format(kfold_path.parent))
        kfold_path.parent.mkdir()

    if not args.cohort_name:
        args.cohort_name = kfold_path.name
        # print('WARNING: --cohort-name not set. Guessing cohort name is {}.'.format(args.cohort_name))
        # TODO: explicitly check if guessed cohort name is a key in the kfold files

    train_h5 = h5py.File(args.train_h5, 'r')
    idx_all = train_h5['idx'][:]
    mapp = train_h5['mappability'][:]

    # idx_lowmap = []
    # if args.map_thresh:
    #     idx_lowmap = idx_all[mapp < args.map_thresh]

    df = region_model_tools.kfold_results(kfold_path, args.cohort_name, key=args.key)
    # try:
    #     # df = region_model_tools.kfold_results(kfold_path, args.cohort_name, key=args.key, idx_submap=idx_lowmap)
    # except Exception:
    #     print('ERROR: failed to load kfold {}. You should rerun the CNN+GP kfold.'.format(kfold_path))
    #     return -1

    # print(df.END.max())
    # if len(df) != len(df_dedup):
    #     print(kfold_path)
    #     print(len(df), len(df_dedup))

    # else:
    print('SUCCESS: loaded {}'.format(kfold_path))

    ## SAVE
    # if output_path.is_file() and not args.overwrite:
    #     print('ERROR: output file {} already exists. Use the --overwrite flag to force writing.')
    #     return -1
    if args.append:
        mode = 'a'
    else:
        mode = 'w'

    output_h5 = h5py.File(str(output_path), mode)
    if not 'idx' in output_h5.keys():
        output_h5.create_dataset('idx', data=idx_all, dtype=np.int32, compression='gzip')
    if not 'mappability' in output_h5.keys():
        output_h5.create_dataset('mappability', data=mapp, dtype=np.float32, compression='gzip')

    # output_h5.attrs['mappability_threshold'] = args.map_thresh
    output_h5.attrs['cohort_name'] = args.cohort_name.encode('utf-8')

    # if args.indels:
    #     print('Creating an INDEL model')
    #     df.to_hdf(output_path, 'region_params_indels', mode='a')
    # else:
    print('Creating an SNV model')
    df.to_hdf(output_path, 'region_params', mode='a')
    # df.to_hdf(output_path, 'region_params', mode='a', complib='zlib')

    if args.fmut:
        print('Adding mutation counts...')
        count_training_mutations(args)

def count_training_mutations(args):
    h5 = h5py.File(args.outputFile, 'a')
    # map_thresh = h5.attrs['mappability_threshold']
    # mapp = h5['mappability'][:]

    df = pd.read_hdf(args.outputFile, 'region_params')
    N_MUT_TOTAL = df.loc[:, 'Y_TRUE'].sum()
    N_MUT_TRAIN = df.loc[~df.FLAG, 'Y_TRUE'].sum()
    # N_MUT_TRAIN = df.loc[mapp > map_thresh, 'Y_TRUE'].sum()

    ## Count CDS mutations
    # df_mut = pd.read_table(args.mutation_file, header=None, low_memory=False)
    df_mut = mutation_tools.read_mutation_file(args.fmut, drop_duplicates=True)
    df_mut_cds = df_mut[df_mut.ANNOT != 'Noncoding']  ## Counts Essential Splice which boarder CDS regions
    N_SAMPLE = len(df_mut.SAMPLE.unique())
    df_mut = mutation_tools.drop_duplicate_mutations(df_mut)
    # df_bed = pd.read_table(args.cds_file, header=None, low_memory=False)
    # df_mut_cds = mutation_tools.restrict_mutations_by_bed(df_mut, df_bed, unique=True, remove_X=False)
    N_MUT_CDS = len(df_mut_cds)
    N_MUT_SAMPLE_CDS = df_mut_cds.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts().sum()

    ## Count MSK-IMPACT mutations
    df_230 = pd.read_table(pkg_resources.resource_stream('DIGDriver', 'data/genes_MSK_230.txt'), names=['GENE'])
    df_341 = pd.read_table(pkg_resources.resource_stream('DIGDriver', 'data/genes_MSK_341.txt'), names=['GENE'])
    df_410 = pd.read_table(pkg_resources.resource_stream('DIGDriver', 'data/genes_MSK_410.txt'), names=['GENE'])
    df_468 = pd.read_table(pkg_resources.resource_stream('DIGDriver', 'data/genes_MSK_468.txt'), names=['GENE'])
    df_metabric = pd.read_table(pkg_resources.resource_stream('DIGDriver', 'data/genes_metabric_173.txt'), names=['GENE'])
    df_ucla = pd.read_table(pkg_resources.resource_stream('DIGDriver', 'data/genes_ucla_1202.txt'), names=['GENE'])

    df_mut_230 = df_mut_cds[(df_mut_cds.GENE.isin(df_230.GENE)) & (df_mut_cds.ANNOT != 'Synonymous') & (df_mut_cds.ANNOT != 'Essential_Splice') & (df_mut_cds.ANNOT != "Noncoding")]
    df_mut_341 = df_mut_cds[(df_mut_cds.GENE.isin(df_341.GENE)) & (df_mut_cds.ANNOT != 'Synonymous') & (df_mut_cds.ANNOT != 'Essential_Splice') & (df_mut_cds.ANNOT != "Noncoding")]
    df_mut_410 = df_mut_cds[(df_mut_cds.GENE.isin(df_410.GENE)) & (df_mut_cds.ANNOT != 'Synonymous') & (df_mut_cds.ANNOT != 'Essential_Splice') & (df_mut_cds.ANNOT != "Noncoding")]
    df_mut_468 = df_mut_cds[(df_mut_cds.GENE.isin(df_468.GENE)) & (df_mut_cds.ANNOT != 'Synonymous') & (df_mut_cds.ANNOT != 'Essential_Splice') & (df_mut_cds.ANNOT != "Noncoding")]
    df_mut_metabric = df_mut_cds[(df_mut_cds.GENE.isin(df_metabric.GENE)) & (df_mut_cds.ANNOT != 'Synonymous') & (df_mut_cds.ANNOT != 'Essential_Splice') & (df_mut_cds.ANNOT != "Noncoding")]
    df_mut_ucla = df_mut_cds[(df_mut_cds.GENE.isin(df_ucla.GENE)) & (df_mut_cds.ANNOT != 'Synonymous') & (df_mut_cds.ANNOT != 'Essential_Splice') & (df_mut_cds.ANNOT != "Noncoding")]

    N_MUT_MSK_230 = len(df_mut_230)
    N_MUT_MSK_341 = len(df_mut_341)
    N_MUT_MSK_410 = len(df_mut_410)
    N_MUT_MSK_468 = len(df_mut_468)
    N_MUT_MSK_metabric = len(df_mut_metabric)
    N_MUT_MSK_ucla = len(df_mut_ucla)

    N_MUT_SAMPLE_MSK_230 = df_mut_230.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts().sum()
    N_MUT_SAMPLE_MSK_341 = df_mut_341.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts().sum()
    N_MUT_SAMPLE_MSK_410 = df_mut_410.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts().sum()
    N_MUT_SAMPLE_MSK_468 = df_mut_468.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts().sum()
    N_MUT_SAMPLE_MSK_metabric = df_mut_metabric.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts().sum()
    N_MUT_SAMPLE_MSK_ucla = df_mut_ucla.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts().sum()

    N_SAMPLE_MSK_230 = len(df_mut_230.SAMPLE.unique())

    ## Save values
    h5.attrs['N_SAMPLES']   = N_SAMPLE
    h5.attrs['N_MUT_TOTAL'] = N_MUT_TOTAL
    h5.attrs['N_MUT_TRAIN'] = N_MUT_TRAIN

    h5.attrs['N_MUT_CDS']   = N_MUT_CDS
    h5.attrs['N_MUT_SAMPLE_CDS']   = N_MUT_CDS

    h5.attrs['N_MUT_MSK_230']   = N_MUT_MSK_230
    h5.attrs['N_MUT_MSK_341']   = N_MUT_MSK_341
    h5.attrs['N_MUT_MSK_410']   = N_MUT_MSK_410
    h5.attrs['N_MUT_MSK_468']   = N_MUT_MSK_468
    h5.attrs['N_MUT_metabric_173']   = N_MUT_MSK_metabric
    h5.attrs['N_MUT_ucla_1202']   = N_MUT_MSK_ucla

    h5.attrs['N_MUT_SAMPLE_MSK_230']   = N_MUT_SAMPLE_MSK_230
    h5.attrs['N_MUT_SAMPLE_MSK_341']   = N_MUT_SAMPLE_MSK_341
    h5.attrs['N_MUT_SAMPLE_MSK_410']   = N_MUT_SAMPLE_MSK_410
    h5.attrs['N_MUT_SAMPLE_MSK_468']   = N_MUT_SAMPLE_MSK_468
    h5.attrs['N_MUT_SAMPLE_metabric_173']   = N_MUT_SAMPLE_MSK_metabric
    h5.attrs['N_MUT_SAMPLE_ucla_1202']   = N_MUT_SAMPLE_MSK_ucla

    h5.attrs['N_SAMPLE_MSK_230'] = N_SAMPLE_MSK_230

def pretrain_sequence_model(args):
    """ MS Notes:
        * Mappability threshold for sequence model training should be imposed here!
    """
    ## Load genome counts
    print('Loading genome-wide context counts')
    df_genome = pd.read_hdf(args.genome_counts, 'all_window_genome_counts')

    with h5py.File(args.genome_counts, 'r') as h5:
        idx = h5['idx'][:]
        mapp = h5['mappability'][:]
        n_up = h5.attrs['n_up']
        n_down = h5.attrs['n_down']

    idx = idx[mapp > args.map_thresh]
    df_genome = df_genome[mapp > args.map_thresh]
    S_genome = df_genome.sum(axis=0)

    print('Loading mutation file')
    df_mut = mutation_tools.read_mutation_file(args.fmut, drop_duplicates=True)
    df_mut = df_mut[df_mut.ANNOT != 'INDEL']
    # print('Dropping duplicate mutation eintries')
    # df_mut = mutation_tools.drop_duplicate_mutations(df_mut)

    print('Training sequence model')
    df_freq_mut, df_freq_context = sequence_tools.train_sequence_model(idx, df_mut, S_genome)

    print('Saving sequence models to {}'.format(args.output_h5))
    df_freq_mut.to_hdf(args.output_h5, 'sequence_model_192', mode='a')
    df_freq_context.to_hdf(args.output_h5, 'sequence_model_64', mode='a')

## Now deprecated. Use DIGPreprocess for up-to date version
def count_nonc_contexts(args):
    """ MS Notes:
        * Should be moved to DIGPreprocess
        * Si counts per element should be done here instead of in pretrain_nonc_model? (Leaning yes)
        * Should create L counts from a sites file if provided here
    """
    with h5py.File(args.f_genome_counts, 'r') as h5:
        keys = list(h5.keys())
    assert 'idx' in keys and 'all_window_genome_counts' in keys, \
        "f_genome_counts file does not contain necessary groups. Please check that the correct file is passed"

    L_restuls = sequence_tools.precount_nonc_contexts_parallel(args.f_nonc_bed, args.f_nonc_data, args.f_genome_counts, args.f_fasta, args.N_procs, args.use_sub_elts)
    L_restuls.to_hdf(args.f_nonc_data, args.L_out_key)

def pretrain_genic_model(args):
    # print('Finding observed mutations')
    # df_obs = mutation_tools.tabulate_genic_mutations(args.cds_bed_file, args.fmut)

    print('Running Genic model in parallel')
    genic_pretrained = genic_driver_tools.genic_model_parallel(args.f_pretrained, args.f_genic, args.N_procs, counts_key=args.counts_key, indels_direct=args.indels_direct)
    # genic_pretrained = genic_driver_tools.genic_model_parallel(df_obs, args.f_pretrained, args.f_genic, args.f_genome_counts, args.mapp, args.N_procs)
    #save
    if args.output_h5:
        genic_pretrained.to_hdf(args.output_h5, 'genic_model', mode='a')
    else:
        genic_pretrained.to_hdf(args.f_pretrained, 'genic_model', mode='a')

def pretrain_nonc_model(args):
    ''' MS NOTES:
            * Should not take mutation file used for training region model since a user will not be expected to have this.
            * Rename pretrain_element_model?
    '''

    # if args.f_sites:
        # print('Finding observed mutations')
        # print('Using sites file {}'.format(args.f_sites))
        # df_mutsites = mutation_tools.tabulate_nonc_mutations_at_sites(args.f_sites, args.fmut)

    #     print('Running noncoding model in parallel')
    #     nonc_pretrained = genic_driver_tools.nonc_model_parallel(df_mutsites, args.f_pretrained, args.f_nonc_data, args.save_key, args.N_procs, True)
    #     print("done")
    # else:
        # assert args.f_nonc_bed, "ERROR: Need to pass in a bed12 file with element annotations"
        # num_cols = pd.read_csv(args.f_nonc_bed, sep='\t', nrows=1).shape[1]
        # assert num_cols == 12, "Expected bed file with 12 columns, got {}. please pass in bed12 file containing elements".format(num_cols)
        #not using df_split for current analyses
        # _, df_whole = mutation_tools.tabulate_nonc_mutations_split(args.f_nonc_bed, args.fmut)
        # print(df_whole[0:5])

    print('Pretraining element model in parallel')
    # nonc_pretrained = genic_driver_tools.nonc_model_region_parallel(args.f_nonc_bed, args.f_pretrained, args.f_nonc_data, args.nonc_L_key, args.N_procs, False)
    nonc_pretrained = genic_driver_tools.nonc_model_parallel(args.f_pretrained, args.f_element_data, args.save_key, args.N_procs, indels_direct=args.indels_direct)
    #save
    print("saving")
    if args.output_h5:
        nonc_pretrained.to_hdf(args.output_h5, args.save_key, mode='a')
    else:
        nonc_pretrained.to_hdf(args.f_pretrained, args.save_key, mode='a')


def pretrain_tiled(args):
    nonc_pretrained = genic_driver_tools.tiled_model_parallel(args.f_pretrained, args.f_element_data, args.save_key, args.N_procs)
    #save                                                                                                                                       
    print("saving")
    if args.output_h5:
        nonc_pretrained.to_hdf(args.output_h5, args.save_key, mode='a')
    else:
        nonc_pretrained.to_hdf(args.f_pretrained, args.save_key, mode='a')
    
def parse_args(text=None):
    parser = argparse.ArgumentParser()
    subparse = parser.add_subparsers()

    parser_a = subparse.add_parser('regionModel',
        help='Pre-train regional rate parameters from a completed CNN+GP kfold run'
    )
    parser_a.add_argument('kfold_dir', type=str,
        help='Path to directory containing CNN+GP kfold results.'
    )
    parser_a.add_argument('train_h5', type=str,
        help='Path to h5 file used for training CNN+GP.'
    )
    parser_a.add_argument('outputFile',
        help='Output file in which to store pre-trained regional rate parameters.'
    )
    parser_a.add_argument('--cohort-name', type=str, default='',
        help='Cohort name used to store CNN+GP kfold results.'
    )
    parser_a.add_argument('--key', type=str, default='held-out',
        help='Key to use to load CNN+GP results from h5 archive.'
    )
    parser_a.add_argument('--map-thresh', type=float, default=0.5,
        help='Mappability threshold used when training CNN+GP.'
    )
    parser_a.add_argument('--mutation-file', type=str, default=None, dest='fmut',
        help='Path to mutation file in DIG format.'
    )
    # parser_a.add_argument('--indels', action='store_true', default=False, 
    #     help='Create a region model based indels.'
    # )
    parser_a.add_argument('--cds-file', type=str, default="../data/dndscv_gene_cds.bed.gz", 
        help='Path to bed file of coding sequence regions.'
    )
    parser_a.add_argument('--append', action='store_true', default=False, 
        help='Append to an existing output file if it exists instead of creating a new file.'
    )
    # parser_a.set_defaults(func=count_training_mutations)
    parser_a.set_defaults(func=pretrain_region_model)

    #################################################

    parser_a1 = subparse.add_parser('countMutations',
        help='Add mutation counts to a pretrained model'
    )
    parser_a1.add_argument('--outputFile', required=True,
        help='Filename of h5 archive storing pretrained models.'
    )
    parser_a1.add_argument('--mutation-file', required=True, type=str, dest='fmut',
        help='Path to mutation file in DIG format.'
    )
    # parser_a1.set_defaults(func=count_training_mutations)
    parser_a1.set_defaults(func=count_training_mutations)

    ############################################

    parser_b = subparse.add_parser('sequenceModel',
        help='Pre-train the sequence context parameters using pre-computed genome counts and annotated mutations. See `DIDPreprocess.py` to precompute these values if they are not on-hand'
    )
    parser_b.add_argument('fmut',
        help='mutation file in DIG format\nCHROM\tSTART\tEND\tREF\tALT\tSAMPLE\tANNOT\tMUT_TYPE\tCONTEXT'
    )
    parser_b.add_argument('genome_counts',
        help='h5 file of genome-wide context counts generated from DIGProprocess.py countGenomeContext.'
    )
    parser_b.add_argument('output_h5',
        help='h5 archive storing the pre-trained model parameter estimates.'
    )
    parser_b.add_argument('--map-thresh', default=0.5, type=float,
        help='Minimum mappability of region required to be included in estimation (optional)'
    )
    parser_b.set_defaults(func=pretrain_sequence_model)

    ##############################################
    parser_e1 = subparse.add_parser('countNonc_context',
        help='pre-count the context counts in the regions that overlap each noncoding region for fast nonc pretrained calculation'
    )
    parser_e1.add_argument('f_nonc_bed',
        help='Bed12 file containing noncoding annotations'
    )
    parser_e1.add_argument('f_nonc_data',
        help='h5 file containing nonc window context counts and mutation transition index.'
    )
    parser_e1.add_argument('f_genome_counts',
        help='h5 file containing all window context counts.'
    )
    parser_e1.add_argument('f_fasta',
        help='path to reference sequence file (hg19).'
    )
    parser_e1.add_argument('L_out_key',
        help='key for saving L counts'
    )
    parser_e1.add_argument('--ignore_sub_elts',
        action='store_false', default=True, dest = 'use_sub_elts',
        help='flag designates using whole element start-end instead of sub elements'
    )
    parser_e1.add_argument('--n-procs', default=utils.get_cpus(), type=int, dest='N_procs',
        help='Number of cores to use for running the analysis. Default is 2 less than the total number of cores'
    )
    parser_e1.set_defaults(func=count_nonc_contexts)

    ##############################################

    parser_d = subparse.add_parser('genicModel',
        help='Pre-train the sequence context parameters using pre-computed genome counts and annotated mutations. See `DIDPreprocess.py` to precompute these values if they are not on-hand'
    )
    parser_d.add_argument('f_pretrained',
        help='Pretrained file in DIG format containing region model and sequence model'
    )
    # parser_d.add_argument('fmut',
    #     help='mutation file in DIG format\nCHROM\tSTART\tEND\tREF\tALT\tSAMPLE\tANNOT\tMUT_TYPE\tCONTEXT'
    # )
    # parser_d.add_argument('cds_bed_file',
    #      help= 'Bed file storing Gene information.'
    # )
    parser_d.add_argument('f_genic',
        help= 'Path to file of preprocessed gene data.'
    )
    parser_d.add_argument('--counts-key', default="window_10kb/counts",
                          help='key for accessing window counts data in f_genic.'
    )
    parser_d.add_argument('--output_h5',
                          help='h5 archive storing the pre-trained model parameter estimates, if different from pretrain.'
    )
    parser_d.add_argument('--indels-direct', action='store_true', default=False, 
                          help='Estimate indel parameters directly from an indel regions model. Otherwise transferred from SNV model.' 
    )
    # parser_d.add_argument('--map-thresh', default=0.5, type=float, dest='mapp',
    #     help='Minimum mappability of region required to be included in estimation (optional)'
    # )
    parser_d.add_argument('--n-procs', default=utils.get_cpus(), type=int, dest='N_procs',
        help='Number of cores to use for running the analysis. Default is 2 less than the total number of cores'
    )
    parser_d.set_defaults(func=pretrain_genic_model)

    ##############################################

    parser_e = subparse.add_parser('elementModel',
        help='Pre-train the sequence context parameters using pre-computed genome counts and annotated mutations. See `DIDPreprocess.py` to precompute these values if they are not on-hand'
    )
    parser_e.add_argument('f_pretrained',
        help='Pretrained file in DIG format containing region model and sequence model'
    )
    parser_e.add_argument('f_element_data',
        help= 'Path to precounted region and elemet contexts.'
    )
    parser_e.add_argument('save_key',
        help= 'Path to save results to in pretrained or output file'
    )
    # parser_e.add_argument('--f_nonc_bed',
    #      help= 'Bed file storing noncoding element information in bed12 format.'
    # )
    # parser_e.add_argument('--f_sites',
    #     help= 'Path to file containing sites (base pairs) of interest'
    # )
    ##deprecated
    # parser_e.add_argument('--nonc_L_key',
    #     help= 'Path to precounted element contexts in nonc_data'
    # )
    parser_e.add_argument('--output_h5',
                          help='h5 archive storing the pre-trained model parameter estimates, if different from pretrain.'
    )
    parser_e.add_argument('--indels-direct', action='store_true', default=False, 
                          help='Estimate indel parameters directly from an indel regions model. Otherwise transferred from SNV model.' 
    )
    parser_e.add_argument('--n-procs', default=30, type=int, dest='N_procs',
        help='number of cores to use for running the analysis'
    )
    parser_e.set_defaults(func=pretrain_nonc_model)


    parser_f = subparse.add_parser('tiledModel',
        help='Pre-train the sequence context parameters using pre-computed genome counts and annotated mutations. See `DIDPreprocess.py` to precompute these values if they are not on-hand'
    )
    parser_f.add_argument('f_pretrained',
        help='Pretrained file in DIG format containing region model and sequence model'
    )
    parser_f.add_argument('f_element_data',
        help= 'Path to precounted region and elemet contexts.'
    )
    parser_f.add_argument('save_key',
        help= 'Path to save results to in pretrained or output file'
    )
    parser_f.add_argument('--output_h5',
                          help='h5 archive storing the pre-trained model parameter estimates, if different from pretrain.'
    )
    parser_f.add_argument('--n-procs', default=utils.get_cpus(), type=int, dest='N_procs',
        help='number of cores to use for running the analysis. Default is 2 less than the total number of cores'
    )
    parser_f.set_defaults(func=pretrain_tiled)

    
    if text:
        args = parser.parse_args(text.split())
    else:
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    args.func(args)
