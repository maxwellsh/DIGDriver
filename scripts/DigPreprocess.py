#!/usr/bin/env python

import pandas as pd
import numpy as np
import h5py
import argparse
import pysam
import json
import subprocess
import pkg_resources

from DIGDriver.sequence_model import sequence_tools
from DIGDriver.data_tools import mappability_tools
from DIGDriver.data_tools import mutation_tools
from DIGDriver.auxilaries import utils
# from . import nb_model
# from . import gp_tools

def countGenomeContext(args):
    """ MS Notes:
        * Should not restrict mappability; should just save mappability and all counts to h5
    """
    ## Create bed-like dataframe
    assert (args.h5 or args.bed), \
        "One of --h5 or --bed must be supplied."

    assert not (args.h5 and args.bed), \
        "At most one of --h5 or --bed can be supplied."

    if args.h5:
        h5 = h5py.File(args.h5, 'r')
        df_bed = pd.DataFrame(h5['idx'][:])
    else:
        df_bed = pd.read_table(args.bed, header=None, low_memory=False)
        df_bed.iloc[:, 0] = df_bed.iloc[:, 0].astype(str)
        chroms = [str(i) for i in range(1, 23)]
        # df_bed = df_bed[~df_bed.iloc[:, 0].isin(['X', 'Y'])]  ## Restrict to autosomes
        df_bed = df_bed[df_bed.iloc[:, 0].isin(chroms)]  ## Restrict to autosomes
        df_bed.iloc[:, 0] = df_bed.iloc[:, 0].astype(int)

    df_bed = df_bed.sort_values(by=[0, 1])

    print('Counting nucleotide contexts in {} regions'.format(len(df_bed)))
    df = sequence_tools.count_contexts_in_bed(args.fasta, df_bed,
                                                 n_up=args.up, n_down=args.down,
                                                 N_proc = args.n_procs, N_chunk=args.n_procs,
                                                 collapse=False)

    idx = df_bed.iloc[:, 0:3].values
    if args.map_file:
        print('Calculating mappability for each window')
        mapp_res = mappability_tools.mappability_by_idx(args.map_file, idx)
        mapp = np.array([w[-1] for w in mapp_res])

        # print('Restricting to regions with mappability > {}'.format(args.map_thresh))
        # df = df[mapp > args.map_thresh]
        # idx = idx[mapp > args.map_thresh, :]

    S_count = df.sum(axis=0)
    # print(S_count.sum())

    print('Saving context counts to {}'.format(args.fout))
    with h5py.File(args.fout, 'w') as h5:
        S_count.to_hdf(args.fout, key='genome_counts', mode='a')
        df.to_hdf(args.fout, key='all_window_genome_counts', mode='a')
        h5.create_dataset('idx', data=idx, dtype=np.int32, compression='gzip')

        h5.attrs['n_up'] = args.up
        h5.attrs['n_down'] = args.down
        h5.attrs['collapse'] = 0

        if args.map_file:
            h5.create_dataset('mappability', data=mapp, dtype=float)

def addMutationContext(args):
    n_up = args.up
    n_down = args.down

    print('Reading in mutation file')
    # df_mut = pd.read_csv(args.fmut, sep="\t", low_memory=False,
    #                 names=['CHROM', 'START', 'END', 'REF', 'ALT', 'ID', 'ANNOT'])
    df_mut = mutation_tools.read_mutation_file(args.fmut, drop_duplicates=False)
    # df_mut = mutation_tools.drop_duplicate_mutations(df_mut)

    # print(df_mut[0:5])

    print('Extracting mutation contexts')
    df_mut2 = sequence_tools.add_context_to_mutations(args.fasta, df_mut,
                                                      n_up=n_up, n_down=n_down,
                                                      N_proc=args.n_procs,
                                                      collapse=False
    )

    print('Saving annotated mutation file: {}'.format(args.fout))
    if args.fout.endswith('.gz'):
        args.fout = args.fout[:-3]

    df_mut2.to_csv(args.fout, sep="\t", index=False, header=False)
    # sequence_tools.bgzip(args.fout)
    # sequence_tools.tabix_index(args.fout+'.gz', skip=0)

def addMutationFunction(args):
    if args.fout.endswith('.gz'):
        args.fout = args.fout[:-3]

    refdb = pkg_resources.resource_filename('DIGDriver', 'data/refcds_hg19.rda')
    cmd = ['mutationFunction.R', args.fmut, refdb, args.fout]
    # cmd = ['mutationFunction.R', args.fmut, args.refdb, args.fout]
    subprocess.run(cmd)

def annotMutationFile(args):
    print('Adding mutation function')
    addMutationFunction(args)

    print('Adding mutation context')
    args.fmut = args.fout
    addMutationContext(args)

def preprocess_cds_contexts(args):

    with h5py.File(args.f_genic, 'r') as h5:
        keys = list(h5.keys())
    assert 'cds_intervals' in keys and 'chr' in keys and 'strands' in keys and 'substitution_idx' in keys, \
        "f_genic file does not contain necessary groups. Please check that the correct file is passed"

    results = sequence_tools.si_count_parallel(args.f_genic, args.f_fasta, args.window, args.N_procs)
    results.to_hdf(args.out_file, args.out_key)

def preprocess_nonc_contexts(args):
    """ MS Notes:
        * Si counts per element should be done here instead of in pretrain_nonc_model? (Leaning yes)
        * Should create L counts from a sites file if provided here
        ---both done
    """
    assert  args.f_sites or args.f_element_bed, "ERROR: need to pass in a f_sites file or a elements file for preprocessing"

    if args.f_sites:
        print("preprocessing sites data")
        sequence_tools.preprocess_sites(args.f_sites, args.f_element_data, args.f_pretrained, args.save_key, args.window)
    else:
        print("Preprocessing elements")
        L_region_results = sequence_tools.precount_region_contexts_parallel(args.f_element_bed,  args.f_fasta, args.N_procs, args.window, args.use_sub_elts)
        print('window counts by elt')
        sequence_tools.preprocess_nonc(args.f_element_bed, args.f_element_data, args.f_pretrained, L_region_results, args.save_key, args.window)

#function to make f_elements
def initialize_data(args):
    with h5py.File(args.f_genome_counts, 'r') as h5:
        keys = list(h5.keys())
        assert 'idx' in keys and 'all_window_genome_counts' in keys, \
            "f_genome_counts file does not contain necessary groups. Please check that the correct file is passed"
        args.window = h5['idx'][0, 2] - h5['idx'][0, 1]
    sequence_tools.initialize_nonc_data(args.f_annot_data, args.f_genome_counts, args.window)

def preprocess_tiled(args):
    print("Counting sequence contexts in regions")
    L = sequence_tools.precount_region_contexts_parallel(args.f_nonc_bed, args.f_fasta, args.N_procs, args.window, False)
    L.to_hdf(args.f_nonc_data, "{}/L_counts".format(args.save_key))
    del L
#     print("determining region overlaps for each ")
#     region_counts = tiled_region_counts_parallel(f_nonc_bed, f_nonc_data, f_pretrained, n_procs, window)
#     region_counts.to_hdf(f_nonc_bed, "region_counts".format(save_key))



def parse_args(text = None):
    parser = argparse.ArgumentParser(description='Preprocess mutation files for use with DIGDriver.')
    subparser = parser.add_subparsers()

    #############################################

    parse_a = subparser.add_parser('countGenomeContext',
                                   help='count the number of occurences of nucleotide contexts in a genome')
    # parse_a.add_argument('train_h5', type=str, help='path to h5 file for training CNN+GP')
    # parse_a.add_argument('window', type=int, help='window size of mappable regions')
    parse_a.add_argument('fasta', type=str, help='path to fasta file')
    parse_a.add_argument('fout', type=str, help='output h5 file name')
    parse_a.add_argument('--h5', type=str, default='', help='path to h5 file containing genome regions')
    parse_a.add_argument('--bed', type=str, default='', help='path to bed file containing whitelist regions to include. The bed file SHOULD NOT have a header.')
    # parse_a.add_argument('--map-thresh', type=float, default=0.5,
    #                      help='minimum mappability for a region to be considered (only used with --h5)')
    parse_a.add_argument('--up', type=int, default=1,
                         help='number of bases upstream of position to include as context')
    parse_a.add_argument('--down', type=int, default=1,
                         help='number of bases downstream of position to include as context')
    parse_a.add_argument('--n-procs', type=int, default=utils.get_cpus(),
                         help='number of threads to parallelize over')
    # parse_a.add_argument('--collapse', action='store_true', default=False,
    #                      help='collapse mutations to C or T using reverse complementation')
    parse_a.add_argument('--map-file', type=str, default='',
                         help='mappability file (optional)')
    parse_a.add_argument('--map-thresh', type=float, default=0.5,
                         help='minimum mappability threshold to include region in counts (optional). Only used when --map-file is provided.')
    parse_a.set_defaults(func=countGenomeContext)

    #############################################

    parse_b = subparser.add_parser('addMutationContext',
                                   help='Annotations mutation files with sequence context')
    parse_b.add_argument('fmut', type=str, help='path to mutation file')
    # parse_b.add_argument('h5genome', type=str,
    #                      help='h5 file containing genome-wide context counts (from countGenomeContext)')
    parse_b.add_argument('fasta', type=str, help='path to fasta file')
    parse_b.add_argument('fout', type=str, help='output file name')
    parse_b.add_argument('--up', type=int, default=1,
                         help='number of bases upstream of mutation to include as context')
    parse_b.add_argument('--down', type=int, default=1,
                         help='number of bases downstream of mutation to include as context')
    parse_b.add_argument('--n-procs', type=int, default=utils.get_cpus(),
                         help='number of threads to parallelize over')
    # parse_b.add_argument('--collapse', action='store_true', default=False,
    #                      help='collapse mutations to C or T using reverse complementation')
    parse_b.set_defaults(func=addMutationContext)

    #############################################

    parse_c = subparser.add_parser('addMutationFunction',
                                   help='Annotations mutation file with mutation function')
    parse_c.add_argument('fmut', type=str, help='path to mutation file')
    # parse_b.add_argument('h5genome', type=str,
    #                      help='h5 file containing genome-wide context counts (from countGenomeContext)')
    # parse_c.add_argument('refdb', type=str, help='path to refdb rdata file')
    parse_c.add_argument('fout', type=str, help='output file name')
    # parse_b.add_argument('--collapse', action='store_true', default=False,
    #                      help='collapse mutations to C or T using reverse complementation')
    parse_c.set_defaults(func=addMutationFunction)


    #############################################
    parse_d = subparser.add_parser('annotMutationFile',
                                   help='Annotations mutation files with sequence context')
    parse_d.add_argument('fmut', type=str, help='path to mutation file')
    # parse_b.add_argument('h5genome', type=str,
    #                      help='h5 file containing genome-wide context counts (from countGenomeContext)')
    # parse_d.add_argument('refdb', type=str, help='path to refdb rdata file')
    parse_d.add_argument('fasta', type=str, help='path to fasta file')
    parse_d.add_argument('fout', type=str, help='output file name')
    parse_d.add_argument('--up', type=int, default=1,
                         help='number of bases upstream of mutation to include as context')
    parse_d.add_argument('--down', type=int, default=1,
                         help='number of bases downstream of mutation to include as context')
    parse_d.add_argument('--n-procs', type=int, default=utils.get_cpus(),
                         help='number of threads to parallelize over')
    parse_d.set_defaults(func=annotMutationFile)

    #############################################

    # parse_c = subparser.add_parser('countMutationContext',
    #                                help='count the number of occurences of mutations by context in a genome')
    # parse_c.add_argument('mapDict', type=str, help='path to mappable regions file')
    # parse_c.add_argument('fmut', type=str, help='path to mutation file')
    # parse_c.add_argument('h5genome', type=str,
    #                      help='h5 file containing genome-wide context counts (from countGenomeContext)')
    # parse_c.add_argument('window', type=int, help='window size of mappable regions')
    # parse_c.add_argument('keyPrefix', type=str, help='Prefix for mutation count key in h5 file')
    # parse_c.add_argument('--n-procs', type=int, default=20,
    #                      help='number of threads to parallelize over')
    # parse_c.add_argument('--collapse', action='store_true', default=False,
    #                      help='collapse mutations to C or T using reverse complementation')
    # # parse_c.add_argument('fout', type=str, help='output h5 file name (same as for countGenomeContext')
    # # parse_c.add_argument('--up', type=int, default=2,
    # #                      help='number of bases upstream of position to include as context')
    # # parse_c.add_argument('--down', type=int, default=2,
    # #                      help='number of bases downstream of position to include as context')
    # parse_c.set_defaults(func=countMutationContext)

    parser_c1 = subparser.add_parser('preprocess_genic_model',
        help='pre-count the context counts in the regions that overlap each gene for fast genic pretrained calculation'
    )
    parser_c1.add_argument('f_genic',
        help='h5 file containing genic context counts and cds intervals.'
    )
    parser_c1.add_argument('f_fasta',
        help='path to reference sequence file (hg19).'
    )
    parser_c1.add_argument('out_file',
        help='path to save h5 file.'
    )
    parser_c1.add_argument('--out-key', default='cds/window_10kb',
        help='h5 archive key for saving.'
    )
    parser_c1.add_argument('--n-procs', default=utils.get_cpus(), type=int, dest='N_procs',
        help='number of cores to use for running the analysis. default is two less than the total number of cpus'
    )
    parser_c1.add_argument('--window', type = int, default=10000,
        help='window size for the analysis.'
    )
    parser_c1.set_defaults(func=preprocess_cds_contexts)

    ##############################################
    parser_e1 = subparser.add_parser('preprocess_element_model',
        help='pre-count the context counts in the regions that overlap each element region for fast pretrained calculation'
    )
    parser_e1.add_argument('f_element_data',
        help='h5 file containing  window context counts and mutation transition index if it exists, otherwise one will be created at this path'
    )
    parser_e1.add_argument('f_pretrained',
        help='h5 file of any pretrained model. Note that the pretrained here can be any cancer, \
         as it is only used for the sequence model.'
    )
    parser_e1.add_argument('f_fasta',
        help='path to reference sequence file (hg19).'
    )
    parser_e1.add_argument('save_key',
        help='key for saving L counts'
    )
    parser_e1.add_argument('--f-bed', dest='f_element_bed',
        help='Bed12 file containing noncoding annotations'
    )
    parser_e1.add_argument('--f-sites', type =str, default=None,
        help='path to sites file for preprocessing'
    )
    parser_e1.add_argument('--ignore-sub_elts',
        action='store_false', default=True, dest = 'use_sub_elts',
        help='flag designates using whole element start-end instead of sub elements'
    )
    parser_e1.add_argument('--n-procs', default=utils.get_cpus(), type=int, dest='N_procs',
        help='number of cores to use for running the analysis. Default is two less than the total number of cores.'
    )
    parser_e1.add_argument('--window', type=int, default=10000,
        help='window size in bp'
    )
    parser_e1.set_defaults(func=preprocess_nonc_contexts)

    #############################################
    parser_f1 = subparser.add_parser('initialize_f_data',
        help='construct a element or sites data file from f_genome_counts'
    )
    parser_f1.add_argument('f_annot_data',
        help='path to place new data file'
    )
    parser_f1.add_argument('f_genome_counts',
        help='h5 containing the sequence contexts counts for every 10kb window'
    )
    parser_f1.set_defaults(func=initialize_data)
    ##############################################

    parser_g = subparser.add_parser('preprocess_tiled',
        help='preprocess a tiled genome'
    )
    parser_g.add_argument('f_nonc_bed',
        help = 'bed file containing tiled elemennts')
    parser_g.add_argument('f_nonc_data',
	help = 'path to elements data h5 file')
#    parser_g.add_argument('f_pretrained',
#	help = 'path to a pretrained model')
    parser_g.add_argument('f_fasta',
	help = 'hg19 fasta file')
    parser_g.add_argument('--n-procs', default=utils.get_cpus(), type=int, dest='N_procs',
        help='number of cores to use for running the analysis. Default is two less than the total number of cores.')
    parser_g.add_argument('window',
        help='size of windows', type = int, default=10000)
    parser_g.add_argument('save_key', help="key to save L_counts under in nonc_data")

    parser_g.set_defaults(func=preprocess_tiled)
    #####################################################
    if text:
        args = parser.parse_args(text.split())
    else:
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    args.func(args)
