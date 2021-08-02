#!/usr/bin/env python

import numpy as np
import pandas as pd
import scipy.stats
import h5py
import pysam
import pathlib
import argparse
import os

from DIGDriver.data_tools import mutation_tools
from DIGDriver.sequence_model import nb_model
from DIGDriver.sequence_model import gp_tools
from DIGDriver.driver_model import transfer_tools
from DIGDriver.driver_model import onthefly_tools

def gene_driver(args):
    print('Running gene driver detection')
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    ## TODO: add sample / mutation filtering
    df_dig = transfer_tools.run_gene_model(
        args.fmut,
        args.model,
        scale_by_sample=args.scale_by_samples,
        pval_burden_nb=args.pval_burden,
        # pval_burden_dnds=args.pval_burden_dnds,
        # pval_sel=args.pval_sel,
        max_muts_per_sample = args.max_muts_per_sample,
        max_muts_per_gene_per_sample = args.max_muts_per_gene_per_sample,
        scale_factor=args.scale_factor_manual,
        scale_by_expectation=args.scale_by_expectation,
        cgc_genes=args.cgc_genes
    )

    f_out = os.path.join(args.outdir, args.outpfx + '.results.txt')
    print('\tSaving results to {}'.format(f_out))
    # cols = ['CHROM'] + df_dig.columns[12:].to_list()
    # print(cols)
    # df_dig[cols].to_csv(f_out, header=True, index=True, sep="\t")
    df_dig.to_csv(f_out, header=True, index=True, sep="\t")

def target_driver(args):
    print('Running MSK-IMPACT driver detection')
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    df_dig = transfer_tools.run_target_model(args.fmut,
                                          args.model,
                                          scale_by_sample=args.scale_by_samples,
                                          panel=args.panel,
                                          max_muts_per_sample=args.max_muts_per_sample,
                                          max_muts_per_gene_per_sample=args.max_muts_per_gene_per_sample,
                                          cgc_genes=args.cgc_genes,
                                          scale_factor=args.scale_factor_manual,
                                          drop_synonymous=False
    )

    f_out = os.path.join(args.outdir, args.outpfx + '.results.txt')
    print('\tSaving results to {}'.format(f_out))
    # cols = ['CHROM'] + df_dig.columns[12:].to_list()
    # print(cols)
    df_dig.to_csv(f_out, header=True, index=True, sep="\t")

def element_driver(args):
    assert args.f_bed or args.f_sites, "ERROR: you must provide --f-bed or --f-sites."

    print('Running user-defined element driver detection')
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    args.scale_by_expectation = True
    if args.scale_type or args.scale_factor_manual:
        args.scale_by_expectation = False

    if args.scale_factor_manual or args.scale_factor_indel_manual:
        assert (args.scale_factor_manual and args.scale_factor_indel_manual), \
            print("ERROR: must specify both --scale-factor-manual and --scale-factor-indel-manual.")

    if args.f_sites:
        df_dig = transfer_tools.run_sites_region_model(
            args.fmut,
            args.f_sites,
            args.model,
            args.pretrain_key,
            scale_factor=args.scale_factor_manual,
            scale_type=args.scale_type,
            scale_by_expectation=args.scale_by_expectation
        )

    else:
        df_dig = transfer_tools.run_element_region_model(
            args.fmut,
            args.f_bed,
            args.model,
            args.pretrain_key,
            scale_type = args.scale_type,
            scale_factor=args.scale_factor_manual,
            scale_factor_indel=args.scale_factor_indel_manual,
            max_muts_per_sample = args.max_muts_per_sample,
            max_muts_per_elt_per_sample = args.max_muts_per_elt_per_sample,
            scale_by_expectation=args.scale_by_expectation,
            skip_pvals=args.skip_pvals
        )

    df_dig['OBS_SAMPLES'] = df_dig.OBS_SAMPLES.astype(int)
    df_dig['OBS_SNV'] = df_dig.OBS_SNV.astype(int)

    if 'OBS_INDEL' in df_dig.columns:
        df_dig['OBS_INDEL'] = df_dig.OBS_INDEL.astype(int)
    # df_dig['OBS_MUT'] = df_dig.OBS_MUT.astype(int)

    f_out = os.path.join(args.outdir, args.outpfx + '.results.txt')
    print('\tSaving results to {}'.format(f_out))
    # cols = ['CHROM'] + df_dig.columns[12:].to_list()
    df_dig.to_csv(f_out, header=True, index=True, sep="\t")

def onthefly(args):
    assert args.f_elts_bed or args.region_str, "ERROR: you must provide --f-bed or --region_str."

    print('Running user-defined element driver detection')
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    args.scale_by_expectation = True
    if args.scale_type or args.scale_factor_manual:
        print(args.scale_type)
        args.scale_by_expectation = False

    if args.scale_factor_manual or args.scale_factor_indel_manual:
        assert (args.scale_factor_manual and args.scale_factor_indel_manual), \
            print("ERROR: must specify both --scale-factor-manual and --scale-factor-indel-manual.")

    df_dig = onthefly_tools.DIG_onthefly(args.model,
        args.fmut,
        args.f_fasta,
        f_elts_bed=args.f_elts_bed,
        region_str=args.region_str,
        scale_factor=args.scale_factor_manual,
        scale_factor_indel=args.scale_factor_indel_manual,
        scale_type = args.scale_type,
        max_muts_per_sample = args.max_muts_per_sample,
        max_muts_per_elt_per_sample = args.max_muts_per_elt_per_sample,
        scale_by_expectation=args.scale_by_expectation,
        skip_pvals=args.skip_pvals)

    df_dig['OBS_SAMPLES'] = df_dig.OBS_SAMPLES.astype(int)
    df_dig['OBS_SNV'] = df_dig.OBS_SNV.astype(int)

    if 'OBS_INDEL' in df_dig.columns:
        df_dig['OBS_INDEL'] = df_dig.OBS_INDEL.astype(int)

    f_out = os.path.join(args.outdir, args.outpfx + '.results.txt')
    print('\tSaving results to {}'.format(f_out))
    # cols = ['CHROM'] + df_dig.columns[12:].to_list()
    df_dig.to_csv(f_out, header=True, index=True, sep="\t")

def parse_args(text = None):
    parser = argparse.ArgumentParser(description='Preprocess mutation files for use with DIGDriver.')
    subparser = parser.add_subparsers()

    #############################################

    parse_a = subparser.add_parser('geneDriver',
                                   help='detect driver genes in a cohort.')
    parse_a.add_argument('fmut', type=str, help='Mutation file created using DigPreprocess.py annotMutationFile')
    parse_a.add_argument('model', type=str, help='Path to h5 archive storing Dig pretrained gene model.')
    parse_a.add_argument('--outpfx', type=str, required=True, help='output prefix')
    parse_a.add_argument('--outdir', type=str, required=True, help='output directory')
    parse_a.add_argument('--max-muts-per-sample', type=int, default=3e9,
        help='Maximum number of mutations per sample. Samples exceeding this threshold will be removed.')
    parse_a.add_argument('--max-muts-per-gene-per-sample', type=int, default=3e9,
        help='Maximum number of mutations per sample per gene.')
    parse_a.add_argument('--scale-by-mutations', action='store_false', default=True, dest="scale_by_expectation",
                         help='Estimate scale parameter based on mutation counts.')
    parse_a.add_argument('--scale-by-samples', action='store_true', default=False,
                         help='Estimate scale parameter based on number of samples instead of total mutations.')
    parse_a.add_argument('--scale-factor-manual', default=None, type=float,
                         help='Provide a user-defined scale factor. Useful if testing on the training dataset so scale_factor=1.0')
    parse_a.add_argument('--cgc-genes', choices=['CGC_ALL', 'CGC_ONC', 'CGC_TSG'], default=False,
                         help='Calculate p-values for the specified set of CGC genes (choices: CGC_ALL, CGC_ONC, or CGC_TSG).')
    parse_a.add_argument('--no-pval-burden', dest='pval_burden', action='store_false', default=True,
                         help='DO NOT calculate the default burden p-value.')
    # parse_a.add_argument('--no-pval-burden-dnds', dest='pval_burden_dnds', action='store_false', default=True,
    #                      help='DO NOT calculate the dnds-corrected burden p-value.')
    # parse_a.add_argument('--no-pval-sel', dest='pval_sel', action='store_false', default=True,
    #                      help='DO NOT calculate the selection p-value.')
    parse_a.set_defaults(func=gene_driver)

    #############################################

    parse_b = subparser.add_parser('targetDriver',
                                   help='detect driver genes in an MSK-IMPACT targeted sequencing cohort.')
    parse_b.add_argument('fmut', type=str, help='Mutation file created using DigPreprocess.py annotMutationFile')
    parse_b.add_argument('model', type=str, help='Path to h5 archive storing Dig pretrained gene model.')
    parse_b.add_argument('--outpfx', type=str, required=True, help='output prefix')
    parse_b.add_argument('--outdir', type=str, required=True, help='output directory')
    parse_b.add_argument('--panel', type=str, choices=['MSK_230', 'MSK_341', 'MSK_410', 'MSK_468', 'metabric_173', 'ucla_1202'], help='# genes in panel (MSK_230, MSK_341, MSK_410, MSK_468, metabric_173, ucla_1202)')
    parse_b.add_argument('--max-muts-per-sample', type=int, default=3e9,
        help='Maximum number of mutations per sample. Samples exceeding this threshold will be removed.')
    parse_b.add_argument('--max-muts-per-gene-per-sample', type=int, default=3e9,
        help='Maximum number of mutations per sample per gene.')
    # parse_b.add_argument('--max-muts-per-sample', type=int, default=3000,
    #     help='Maximum number of mutations per sample. Samples exceeding this threshold will be removed.')
    # parse_b.add_argument('--max-muts-per-sample-gene', type=int, default=3,
    #     help='Maximum number of mutations per sample per gene.')
    parse_b.add_argument('--scale-by-samples', action='store_true', default=False,
                         help='Estimate scale parameter based on number of samples instead of total mutations.')
    parse_b.add_argument('--scale-factor-manual', default=None, type=float,
                         help='Provide a user-defined scale factor. Useful if testing on the training dataset so scale_factor=1.0')
    parse_b.add_argument('--cgc-genes', choices=['CGC_ALL', 'CGC_ONC', 'CGC_TSG'], default=False,
                         help='Calculate p-values for the specified set of CGC genes (choices: CGC_ALL, CGC_ONC, or CGC_TSG).')
    # parse_b.add_argument('--no-pval-burden', dest='pval_burden', action='store_false', default=True,
    #                      help='DO NOT calculate the default burden p-value.')
    # parse_b.add_argument('--no-pval-burden-dnds', dest='pval_burden_dnds', action='store_false', default=True,
    #                      help='DO NOT calculate the dnds-corrected burden p-value.')
    # parse_b.add_argument('--no-pval-sel', dest='pval_sel', action='store_false', default=True,
    #                      help='DO NOT calculate the selection p-value.')
    parse_b.set_defaults(func=target_driver)

    #############################################
    parse_c = subparser.add_parser('elementDriver',
                                   help='detect driver genes in arbitrary user-defined elements.')
    parse_c.add_argument('fmut', type=str, help='Mutation file created using DigPreprocess.py annotMutationFile')
    parse_c.add_argument('model', type=str, help='Path to h5 archive storing Dig pretrained element model.')
    parse_c.add_argument('pretrain_key', type=str, help='Name of key used to strore pretrained model.')
    parse_c.add_argument('--f-bed', type=str, default="", help='Bed file of annotations used to pretrain the element model')
    parse_c.add_argument('--f-sites', type=str, default="", help='Sites file')
    parse_c.add_argument('--outpfx', type=str, required=True, help='output prefix')
    parse_c.add_argument('--outdir', type=str, required=True, help='output directory')
    parse_c.add_argument('--max-muts-per-sample', type=int, default=3e9,
        help='Maximum number of mutations per sample within elements. Samples exceeding this threshold will be removed.')
    parse_c.add_argument('--max-muts-per-elt-per-sample', type=int, default=3e9,
        help='Maximum number of mutations per element per sample. Element-sample pairs exceeding this threshold will be capped at this number.')
    parse_c.add_argument('--scale-type', default=None, choices=['genome', 'exome', 'sample', 'MSK_230', 'PCAWG_cds'],
                         help='How to calculate the scaling factor for the transfer model.')
    parse_c.add_argument('--scale-factor-manual', default=None, type=float,
                         help='Provide a user-defined scale factor. Useful if testing on the training dataset so scale_factor=1.0')
    parse_c.add_argument('--skip_pvals', default=False, action='store_true', help="flag to skip pval calculations")
    parse_c.add_argument('--scale-factor-indel-manual', default=None, type=float,
                         help='Provide a user-defined scale factor for INDELs. Useful if testing on the training dataset so scale_factor=1.0')
    parse_c.set_defaults(func=element_driver)

    #########################################
    parse_d = subparser.add_parser('quickDriver',
                                   help='detect driver genes on the fly in user-defined bed files or in string defined region.')
    parse_d.add_argument('fmut', type=str, help='Mutation file created using DigPreprocess.py annotMutationFile')
    parse_d.add_argument('model', type=str, help='Path to h5 archive storing Dig pretrained element model.')
    parse_d.add_argument('f_fasta', type=str, help='Path to hg19 fasta file.')
    parse_d.add_argument('--f_elts_bed', type=str, default="", help='Bed file of annotations used to pretrain the element model')
    parse_d.add_argument('--region_str', type=str, default="", help='region of interset defined by chr\{\}:start-end')
    parse_d.add_argument('--outpfx', type=str, required=True, help='output prefix')
    parse_d.add_argument('--outdir', type=str, required=True, help='output directory')
    parse_d.add_argument('--max-muts-per-sample', type=int, default=3e9,
        help='Maximum number of mutations per sample within elements. Samples exceeding this threshold will be removed.')
    parse_d.add_argument('--max-muts-per-elt-per-sample', type=int, default=3e9,
        help='Maximum number of mutations per element per sample. Element-sample pairs exceeding this threshold will be capped at this number.')
    parse_d.add_argument('--scale-type', default=None, choices=['genome', 'exome', 'sample', 'MSK_230', 'PCAWG_cds'],
                         help='How to calculate the scaling factor for the transfer model.')
    parse_d.add_argument('--scale-factor-manual', default=None, type=float,
                         help='Provide a user-defined scale factor. Useful if testing on the training dataset so scale_factor=1.0')
    parse_d.add_argument('--skip_pvals', default=False, action='store_true', help="flag to skip pval calculations")
    parse_d.add_argument('--scale-factor-indel-manual', default=None, type=float,
                         help='Provide a user-defined scale factor for INDELs. Useful if testing on the training dataset so scale_factor=1.0')
    parse_d.set_defaults(func=onthefly)


    if text:
        args = parser.parse_args(text.split())
    else:
        args = parser.parse_args()

    return args

def welcome():
    msg = """
************************
|                      |
| Welcome to DigDriver |
| Version: 1.0         |
| Copyright: 2020      |
|                      |
| Designers:           |
|     Maxwell Sherman, |
|     Adam Yaari,      |
|     Oliver Priebe,   |
|     Bonnie Berger    |
|                      |
************************
"""
    print(msg)

if __name__ == "__main__":
    args = parse_args()
    welcome()
    args.func(args)
