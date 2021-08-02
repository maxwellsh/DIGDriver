import numpy as np
import pandas as pd
import scipy.stats
import h5py
import pkg_resources
import pybedtools

from DIGDriver.data_tools import mutation_tools
from DIGDriver.sequence_model import nb_model

def load_pretrained_model(h5, key='genic_model', restrict_cols=True):
    """ Load a pretrained gene model
    """
    ## TODO: THIS WILL NEED UPDATING WHEN THE PRETRAINED MODULE IS FINALIZED
    df_pretrain = pd.read_hdf(h5, key)

    alpha, theta = nb_model.normal_params_to_gamma(df_pretrain.MU, df_pretrain.SIGMA)
    df_pretrain['ALPHA'] = alpha
    df_pretrain['THETA'] = theta

    if key == 'genic_model':
        df_pretrain.set_index(df_pretrain.GENE, inplace=True)
        ## Rename "P" columns (should do this at pretraining level!)
        df_pretrain.rename({'P_MIS': 'Pi_MIS',
                            'P_NONS': 'Pi_NONS',
                            'P_SILENT': 'Pi_SYN',
                            'P_SPLICE': 'Pi_SPL',
                            'P_TRUNC': 'Pi_TRUNC',
                            'P_INDEL': 'Pi_INDEL',
                           }, axis=1, inplace=True
        )

        ## Add in nonsynymous relative probability (should do this at pretraining level!)
        df_pretrain['Pi_NONSYN'] = df_pretrain.Pi_MIS + df_pretrain.Pi_TRUNC

        alpha_ind, theta_ind = nb_model.normal_params_to_gamma(df_pretrain.MU_INDEL, df_pretrain.SIGMA_INDEL)
        df_pretrain['ALPHA_INDEL'] = alpha_ind
        df_pretrain['THETA_INDEL'] = theta_ind

    elif 'P_INDEL' in df_pretrain.columns:
        df_pretrain.set_index(df_pretrain.ELT, inplace=True)
        df_pretrain.rename({'P_SUM': 'Pi_SUM',
                            'P_INDEL': 'Pi_INDEL',
                           }, axis=1, inplace=True
        )
        alpha_ind, theta_ind = nb_model.normal_params_to_gamma(df_pretrain.MU_INDEL, df_pretrain.SIGMA_INDEL)
        df_pretrain['ALPHA_INDEL'] = alpha_ind
        df_pretrain['THETA_INDEL'] = theta_ind

    else:
        df_pretrain.set_index(df_pretrain.ELT, inplace=True)
        df_pretrain.rename({'P_SUM': 'Pi_SUM',
                           }, axis=1, inplace=True
        )

    if restrict_cols:
        if key == 'genic_model':
            cols = ['CHROM', 'GENE_LENGTH', 'R_SIZE', 'R_OBS', 'R_INDEL',
                    'MU', 'SIGMA', 'ALPHA', 'THETA',
                    'MU_INDEL', 'SIGMA_INDEL', 'ALPHA_INDEL', 'THETA_INDEL', 'FLAG',
                    'Pi_SYN' ,'Pi_MIS', 'Pi_NONS', 'Pi_SPL', 'Pi_TRUNC', 'Pi_NONSYN', 'Pi_INDEL']
        # elif "spliceAI" in key or "sites" in key:
        elif 'Pi_INDEL' in df_pretrain.columns:
            cols = ['ELT_SIZE', 'FLAG', 'R_SIZE', 'R_OBS', 'R_INDEL',
                    'MU', 'SIGMA', 'ALPHA', 'THETA',
                    'MU_INDEL', 'SIGMA_INDEL', 'ALPHA_INDEL', 'THETA_INDEL',
                    'Pi_SUM', 'Pi_INDEL'
            ]
        else:
            cols = ['R_OBS', 'MU', 'SIGMA', 'ALPHA', 'THETA', 'Pi_SUM']
        # else:
        #     cols = ['CHROM', 'R_OBS', 'MU', 'SIGMA', 'ALPHA', 'THETA', 'Pi_SUM']

        df_pretrain = df_pretrain[cols]

    return df_pretrain

def read_mutations_cds(f_mut, f_cds=None):
    """ Read mutations from a WES cohort annotated by DIGPreprocess
    """
    # df_bed['LENGTH'] = df_bed.END - df_bed.START
    df_mut = mutation_tools.read_mutation_file(f_mut, drop_duplicates=False, drop_sex=True)
    df_mut_cds = df_mut[df_mut.GENE != '.']
    # df_mut_cds = df_mut[df_mut.ANNOT != 'Noncoding']

    if f_cds:
        df_cds = pd.read_table(f_cds, names=['CHROM', 'START', 'END', 'GENE'], low_memory=False)
        df_mut_cds = mutation_tools.restrict_mutations_by_bed(df_mut_cds, df_cds,
                        unique=True, replace_cols=True, remove_X=False
                     )

    return df_mut_cds

def calc_scale_factor(df_mut, h5_pretrain, scale_type='genome'):
    """ General purpose function to calculate cohort scaling factor
    """
    df_dedup = mutation_tools.drop_duplicate_mutations(df_mut)

    with h5py.File(h5_pretrain, 'r')  as h5:
        if scale_type == 'genome':
            idx = h5['idx'][:]
            mapp = h5['mappability'][:]
            mapp_thresh = h5.attrs['mappability_threshold']

            idx_mapp = idx[mapp > mapp_thresh]
            df_idx = pd.DataFrame(idx_mapp, columns=['CHROM', 'START', 'END'])
            df_inter = mutation_tools.restrict_mutations_by_bed(df_dedup, df_idx, remove_X=False)

            N_MUT_GENOME = len(df_inter)
            N_MUT_TRAIN = h5.attrs['N_MUT_TRAIN']

            return N_MUT_GENOME / N_MUT_TRAIN

        elif scale_type == 'exome':
            N_MUT_EXOME = len(df_dedup[df_dedup.ANNOT != 'Noncoding'])
            N_MUT_TRAIN = h5.attrs['N_MUT_CDS']

            return N_MUT_EXOME / N_MUT_TRAIN

        elif scale_type == 'sample':
            N_SAMPLE = len(df_dedup.SAMPLE.unique())
            N_SAMPLE_TRAIN = h5.attrs['N_SAMPLES']

            return N_SAMPLE / N_SAMPLE_TRAIN

        else:
            raise ValueError("scale_type {} is not recognized".format(scale_type))

def calc_scale_factor_efficient(f_mut, h5_pretrain, scale_type='genome'):
    """ General purpose function to calculate cohort scaling factor
    """
    # df_dedup = mutation_tools.drop_duplicate_mutations(df_mut)

    with h5py.File(h5_pretrain, 'r')  as h5:
        if scale_type == 'genome':
            # idx = h5['idx'][:]
            # mapp = h5['mappability'][:]
            # mapp_thresh = h5.attrs['mappability_threshold']

            # idx_mapp = idx[mapp > mapp_thresh]
            # df_idx = pd.DataFrame(idx_mapp, columns=['CHROM', 'START', 'END'])
            # bed_idx = pybedtools.BedTool.from_dataframe(df_idx)
            # # df_inter = mutation_tools.restrict_mutations_by_bed(df_dedup, df_idx, remove_X=False)
            # df_inter = mutation_tools.restrict_mutations_by_bed_efficient(f_mut, bed_idx.fn, bed12=False, drop_duplicates=True)

            # N_MUT_GENOME = len(df_inter)
            # N_MUT_TRAIN = h5.attrs['N_MUT_TRAIN']
            regions = pd.read_hdf(h5_pretrain, 'region_params')
            regions_pass = regions[~regions.FLAG]
            bed_idx = pybedtools.BedTool.from_dataframe(regions_pass[['CHROM', 'START', 'END']])
            df_inter = mutation_tools.restrict_mutations_by_bed_efficient(f_mut, bed_idx.fn, bed12=False, drop_duplicates=True)
            N_SNV_EXP = regions_pass.Y_PRED.sum()
            N_SNV_OBS = len(df_inter[df_inter.ANNOT != 'INDEL'])
            N_IND_OBS = len(df_inter[df_inter.ANNOT == 'INDEL'])

            cj_snv = N_SNV_OBS / N_SNV_EXP
            cj_ind = N_IND_OBS / N_SNV_EXP

            return cj_snv, cj_ind

        # elif scale_type == 'exome':
        #     df_dedup = mutation_tools.read_mutation_file(f_mut, drop_duplicates=True)
        #     N_MUT_EXOME = len(df_dedup[df_dedup.ANNOT != 'Noncoding'])
        #     N_MUT_TRAIN = h5.attrs['N_MUT_CDS']

        #     return N_MUT_EXOME / N_MUT_TRAIN

        # elif scale_type == 'sample':
        #     df_dedup = mutation_tools.read_mutation_file(f_mut, drop_duplicates=True)
        #     N_SAMPLE = len(df_dedup.SAMPLE.unique())
        #     N_SAMPLE_TRAIN = h5.attrs['N_SAMPLES']

        #     return N_SAMPLE / N_SAMPLE_TRAIN

        else:
            raise ValueError("scale_type {} is not recognized".format(scale_type))

def scale_factor_by_cds(h5_pretrain, df_mut_cds):
    """ Calculate a cohort scaling factor based on the number of mutations
        in the CDS regions of a cohort
    """
    with h5py.File(h5_pretrain, 'r') as h5:
        N_MUT_CDS = h5.attrs['N_MUT_CDS']

    return len(df_mut_cds) / N_MUT_CDS

def scale_factor_by_samples(h5_pretrain, df_mut):
    """ Calculate a cohort scaling factor based on the number of samples
        in the target and reference cohorts
    """
    with h5py.File(h5_pretrain, 'r') as h5:
        N_SAMPLES = h5.attrs['N_SAMPLES']

    return len(df_mut.SAMPLE.unique()) / N_SAMPLES

def transfer_gene_model(df_mut_cds, df_counts, df_pretrain, cj):
    """ Transfer a pretrained gene model to a new cohort

    Args:
        df_mut_cds:     dataframe of cds mutations from new cohort
        df_pretrain:    dataframe of pretrained gene model parameters
        cj:             scaling factor for transfer model
    """
    ## 1. Count mutations per gene
    # df_counts = _mutations_per_gene(df_mut_cds)
    # df_counts = pd.crosstab(df_mut_cds.GENE, df_mut_cds.ANNOT)
    # df_counts.rename({'Missense': 'OBS_MIS',
    #                   'Nonsense': 'OBS_NONS',
    #                   'Synonymous': 'OBS_SYN',
    #                   'Essential_Splice': 'OBS_SPL'
    #                  },
    #     axis=1, inplace=True
    # )

    cols_left = ['CHROM', 'GENE_LENGTH', 'R_SIZE', 'R_OBS', 'R_INDEL',
                 'MU', 'SIGMA', 'ALPHA', 'THETA',
                 'MU_INDEL', 'SIGMA_INDEL', 'ALPHA_INDEL', 'THETA_INDEL', 'FLAG',
                 'Pi_SYN' ,'Pi_MIS', 'Pi_NONS', 'Pi_SPL', 'Pi_TRUNC', 'Pi_NONSYN', 'Pi_INDEL']
    cols_right = ['OBS_SYN', 'OBS_MIS', 'OBS_NONS', 'OBS_SPL', 'OBS_INDEL']

    df_model = df_pretrain[cols_left].merge(df_counts[cols_right],
        left_index=True, right_index=True, how='left'
    )

    ## Fill nans
    df_model.loc[df_model.OBS_MIS.isna(),    'OBS_MIS']    = 0
    df_model.loc[df_model.OBS_NONS.isna(),   'OBS_NONS']   = 0
    df_model.loc[df_model.OBS_SPL.isna(),    'OBS_SPL']    = 0
    df_model.loc[df_model.OBS_SYN.isna(),    'OBS_SYN']    = 0
    df_model.loc[df_model.OBS_INDEL.isna(),  'OBS_INDEL']  = 0
    df_model['OBS_TRUNC'] = df_model.OBS_NONS + df_model.OBS_SPL
    df_model['OBS_NONSYN'] = df_model.OBS_MIS + df_model.OBS_TRUNC

    ## 2. Count number of mutated samples per gene
    df_syn = df_mut_cds[df_mut_cds.ANNOT == 'Synonymous']
    df_mis = df_mut_cds[df_mut_cds.ANNOT == 'Missense']
    df_non = df_mut_cds[df_mut_cds.ANNOT == 'Nonsense']
    df_spl = df_mut_cds[df_mut_cds.ANNOT == 'Essential_Splice']
    df_trunc = df_mut_cds[df_mut_cds.ANNOT.isin(['Nonsense', 'Essential_Splice'])]
    df_nonsyn = df_mut_cds[df_mut_cds.ANNOT.isin(['Missense', 'Nonsense', 'Essential_Splice'])]
    df_indel = df_mut_cds[df_mut_cds.ANNOT == 'INDEL']

    sample_syn_cnt = df_syn.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts()
    sample_mis_cnt = df_mis.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts()
    sample_non_cnt = df_non.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts()
    sample_spl_cnt = df_spl.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts()
    sample_trunc_cnt = df_trunc.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts()
    sample_nonsyn_cnt = df_nonsyn.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts()
    sample_indel_cnt = df_indel.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts()

    df_model['N_SAMP_SYN'] = 0
    df_model['N_SAMP_MIS'] = 0
    df_model['N_SAMP_NONS'] = 0
    df_model['N_SAMP_SPL'] = 0
    df_model['N_SAMP_TRUNC'] = 0
    df_model['N_SAMP_NONSYN'] = 0
    df_model['N_SAMP_INDEL'] = 0

    df_model.loc[sample_syn_cnt.index, 'N_SAMP_SYN'] = sample_syn_cnt
    df_model.loc[sample_mis_cnt.index, 'N_SAMP_MIS'] = sample_mis_cnt
    df_model.loc[sample_non_cnt.index, 'N_SAMP_NONS'] = sample_non_cnt
    df_model.loc[sample_spl_cnt.index, 'N_SAMP_SPL'] = sample_spl_cnt
    df_model.loc[sample_trunc_cnt.index, 'N_SAMP_TRUNC'] = sample_trunc_cnt
    df_model.loc[sample_nonsyn_cnt.index, 'N_SAMP_NONSYN'] = sample_nonsyn_cnt
    df_model.loc[sample_indel_cnt.index, 'N_SAMP_INDEL'] = sample_indel_cnt

    ## Scale theta
    df_model.THETA = df_model.THETA * cj

    return df_model

def transfer_element_model_with_indels(df_mut_tab, df_pretrain, cj, use_chrom=False):
    """ Transfer a pretrained gene model to a new cohort

    Args:
        df_mut_tab:     dataframe of mutations tabulated by element
        df_pretrain:    dataframe of pretrained element model
        cj:             scaling factor for transfer model
    """
    if use_chrom:
        cols_left = ['CHROM', 'R_OBS', 'MU', 'SIGMA', 'ALPHA', 'THETA', 'Pi_SUM']
    else:
        cols_left = ['ELT_SIZE', 'FLAG', 'R_SIZE', 'R_OBS', 'R_INDEL',
                     'MU', 'SIGMA', 'ALPHA', 'THETA',
                     'MU_INDEL', 'SIGMA_INDEL', 'ALPHA_INDEL', 'THETA_INDEL',
                     'Pi_SUM', 'Pi_INDEL'
        ]

    cols_right = ['OBS_SAMPLES', 'OBS_SNV', 'OBS_INDEL']
    # cols_right = ['OBS_SAMPLES', 'OBS_MUT']

    df_model = df_pretrain[cols_left].merge(df_mut_tab[cols_right],
        left_index=True, right_index=True, how='left'
    )
    df_model.loc[df_model.OBS_SNV.isna(),     'OBS_SNV']     = 0
    df_model.loc[df_model.OBS_INDEL.isna(),   'OBS_INDEL']   = 0
    df_model.loc[df_model.OBS_SAMPLES.isna(), 'OBS_SAMPLES'] = 0

    ## Scale theta
    df_model.THETA = df_model.THETA * cj

    return df_model

def transfer_element_model(df_mut_tab, df_pretrain, cj, use_chrom=False):
    """ Transfer a pretrained gene model to a new cohort

    Args:
        df_mut_tab:     dataframe of mutations tabulated by element
        df_pretrain:    dataframe of pretrained element model
        cj:             scaling factor for transfer model
    """
    if use_chrom:
        cols_left = ['CHROM', 'R_OBS', 'MU', 'SIGMA', 'ALPHA', 'THETA', 'Pi_SUM']
    else:
        cols_left = ['R_OBS', 'MU', 'SIGMA', 'ALPHA', 'THETA', 'Pi_SUM']

    cols_right = ['OBS_SAMPLES', 'OBS_SNV']
    # cols_right = ['OBS_SAMPLES', 'OBS_MUT']

    df_model = df_pretrain[cols_left].merge(df_mut_tab[cols_right],
        left_index=True, right_index=True, how='left'
    )
    df_model.loc[df_model.OBS_SNV.isna(),     'OBS_SNV']     = 0
    df_model.loc[df_model.OBS_SAMPLES.isna(), 'OBS_SAMPLES'] = 0

    ## Scale theta
    df_model.THETA = df_model.THETA * cj

    return df_model

def gene_expected_muts_nb(df_model):
    """ Calculated expected mutations in genes based on transferred NB model
    """
    df_model['EXP_SYN']    = df_model.ALPHA * df_model.THETA * df_model.Pi_SYN
    df_model['EXP_MIS']    = df_model.ALPHA * df_model.THETA * df_model.Pi_MIS
    df_model['EXP_NONS']   = df_model.ALPHA * df_model.THETA * df_model.Pi_NONS
    df_model['EXP_SPL'] = df_model.ALPHA * df_model.THETA * df_model.Pi_SPL
    df_model['EXP_TRUNC']  = df_model.ALPHA * df_model.THETA * df_model.Pi_TRUNC
    df_model['EXP_NONSYN'] = df_model.ALPHA * df_model.THETA * df_model.Pi_NONSYN

    return df_model

def element_expected_muts_nb(df_model):
    df_model['EXP_SNV'] = df_model.ALPHA * df_model.THETA * df_model.Pi_SUM

    # OBS_MUT = df_model.OBS_MUT.values.copy()
    # OBS_SAMPLES = df_model.OBS_SAMPLES.values.copy()

    # OBS_SAMPLES[OBS_MUT == 0] = 1
    # OBS_MUT[OBS_MUT == 0] = 1

    # SCALE_SAMPLE = OBS_SAMPLES / OBS_MUT
    # df_model['EXP_SAMPLES'] = df_model.ALPHA * df_model.THETA * df_model.Pi_SUM * SCALE_SAMPLE

    return df_model

    # def _calc_sample_scale(N_SAMP, OBS_MUT):
    #     if OBS_MUT == 0:
    #         return 1
    #     else:
    #         return N_SAMP / OBS_MUT

def gene_expected_muts_dnds(df_model):
    """ Calculate expected mutations in genes using dNdS correction
    """
    ## Baseline expected mutations from transfer model
    df_model['EXP_SYN']    = df_model.ALPHA * df_model.THETA * df_model.Pi_SYN
    df_model['EXP_MIS']    = df_model.ALPHA * df_model.THETA * df_model.Pi_MIS
    df_model['EXP_NONS']   = df_model.ALPHA * df_model.THETA * df_model.Pi_NONS
    df_model['EXP_SPL'] = df_model.ALPHA * df_model.THETA * df_model.Pi_SPL
    df_model['EXP_TRUNC']  = df_model.ALPHA * df_model.THETA * df_model.Pi_TRUNC
    df_model['EXP_NONSYN'] = df_model.ALPHA * df_model.THETA * df_model.Pi_NONSYN

    ## MLE estimate of neutral mutation rate
    df_model['T_SYN'] = [_mle_t(row.OBS_SYN, 1, row.ALPHA, row.THETA*row.Pi_SYN)
                         for i, row in df_model.iterrows()
    ]

    ## Mutation rate correction factor
    df_model['MRFOLD'] = [_mrfold_factor(row.T_SYN, row.EXP_SYN)
                          for i, row in df_model.iterrows()
    ]

    ## Rate-corrected expected mutations
    df_model['EXP_SYN_ML']    = df_model.EXP_SYN    * df_model.MRFOLD
    df_model['EXP_MIS_ML']    = df_model.EXP_MIS    * df_model.MRFOLD
    df_model['EXP_NONS_ML']   = df_model.EXP_NONS   * df_model.MRFOLD
    df_model['EXP_SPL_ML'] = df_model.EXP_SPL * df_model.MRFOLD
    df_model['EXP_TRUNC_ML']  = df_model.EXP_TRUNC  * df_model.MRFOLD
    df_model['EXP_NONSYN_ML'] = df_model.EXP_NONSYN * df_model.MRFOLD

    return df_model

def gene_pvalue_burden_nb(df_model):
    """ Calculate burden P-values based on the transfered NB model params
    """

    # PVAL_SYN, PVAL_MIS, PVAL_NONS, PVAL_SPL, PVAL_TRUNC, PVAL_NONSYN = [], [], [], [], [], []
    # for i, row in df_model.iterrows():
    #     PVAL_SYN.append(nb_model.nb_pvalue_greater_midp(row.OBS_SYN, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_SYN + 1)
    #                    )
    #     )
    #     PVAL_MIS.append(nb_model.nb_pvalue_greater_midp(row.OBS_MIS, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_MIS + 1)
    #                    )
    #     )
    #     PVAL_NONS.append(nb_model.nb_pvalue_greater_midp(row.OBS_NONS, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_NONS + 1)
    #                    )
    #     )
    #     PVAL_SPL.append(nb_model.nb_pvalue_greater_midp(row.OBS_SPL, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_SPL + 1)
    #                    )
    #     )
    #     PVAL_TRUNC.append(nb_model.nb_pvalue_greater_midp(row.OBS_TRUNC, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_TRUNC + 1)
    #                    )
    #     )
    #     PVAL_NONSYN.append(nb_model.nb_pvalue_greater_midp(row.OBS_NONSYN, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_NONSYN + 1)
    #                    )
    #     )

    df_model['PVAL_SYN_BURDEN'] = nb_model.nb_pvalue_greater_midp(
        df_model.OBS_SYN,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_SYN + 1)
    )
    df_model['PVAL_MIS_BURDEN'] = nb_model.nb_pvalue_greater_midp(
        df_model.OBS_MIS,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_MIS + 1)
    )
    df_model['PVAL_NONS_BURDEN'] = nb_model.nb_pvalue_greater_midp(
        df_model.OBS_NONS,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_NONS + 1)
    )
    df_model['PVAL_SPL_BURDEN'] = nb_model.nb_pvalue_greater_midp(
        df_model.OBS_SPL,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_SPL + 1)
    )
    df_model['PVAL_TRUNC_BURDEN'] = nb_model.nb_pvalue_greater_midp(
        df_model.OBS_TRUNC,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_TRUNC + 1)
    )
    df_model['PVAL_NONSYN_BURDEN'] = nb_model.nb_pvalue_greater_midp(
        df_model.OBS_NONSYN,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_NONSYN + 1)
    )

    return df_model

def element_pvalue_burden_nb_DEPRECATED(df_model):
    """ Calculate burden P-values based on the transfered NB model params
    """

    PVAL_SNV = []
    for i, row in df_model.iterrows():
        PVAL_SNV.append(nb_model.nb_pvalue_greater_midp(row.OBS_SNV, row.ALPHA,
                            1 / (row.THETA * row.Pi_SUM + 1)
                       )
        )

    df_model['PVAL_SNV_BURDEN'] = PVAL_SNV

    return df_model

def element_pvalue_burden_nb(df_model):
    """ Calculate burden P-values based on the transfered NB model params
    """
    df_model['PVAL_SNV_BURDEN'] = nb_model.nb_pvalue_greater_midp(
        df_model.OBS_SNV,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_SUM + 1)
    )

    return df_model

def gene_pvalue_burden_nb_by_sample(df_model):
    """ Calculate burden P-values based on the transfered NB model params.
        Test based only on the number of *mutated* samples per gene
    """

    # def _calc_sample_scale(OBS_MUT, N_SAMP):
    # def _calc_sample_scale(N_SAMP, OBS_MUT):
    #     if OBS_MUT == 0:
    #         return 1
    #     else:
    #         return N_SAMP / OBS_MUT

    # PVAL_SYN, PVAL_MIS, PVAL_NONS, PVAL_SPL, PVAL_TRUNC, PVAL_NONSYN = [], [], [], [], [], []
    # # C_SYN, C_MIS, C_NONS, C_SPL, C_TRUNC, C_NONSYN = [], [], [], [], [], []
    # for i, row in df_model.iterrows():
    #     # c_syn = _calc_sample_scale(row.N_SAMP_SYN, row.OBS_SYN)
    #     PVAL_SYN.append(nb_model.nb_pvalue_greater_midp(row.N_SAMP_SYN, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_SYN + 1)
    #                         # 1 / (row.THETA * row.Pi_SYN * c_syn + 1)
    #                    )
    #     )
    #     # C_SYN.append(c_syn)

    #     # c_mis = _calc_sample_scale(row.N_SAMP_MIS, row.OBS_MIS)
    #     PVAL_MIS.append(nb_model.nb_pvalue_greater_midp(row.N_SAMP_MIS, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_MIS + 1)
    #                         # 1 / (row.THETA * row.Pi_MIS * c_mis + 1)
    #                    )
    #     )
    #     # C_MIS.append(c_mis)

    #     # c_nons = _calc_sample_scale(row.N_SAMP_NONS, row.OBS_NONS)
    #     PVAL_NONS.append(nb_model.nb_pvalue_greater_midp(row.N_SAMP_NONS, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_NONS + 1)
    #                         # 1 / (row.THETA * row.Pi_NONS * c_nons + 1)
    #                    )
    #     )
    #     # C_NONS.append(c_nons)

    #     # c_spl = _calc_sample_scale(row.N_SAMP_SPL, row.OBS_SPL)
    #     PVAL_SPL.append(nb_model.nb_pvalue_greater_midp(row.N_SAMP_SPL, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_SPL + 1)
    #                         # 1 / (row.THETA * row.Pi_SPL * c_spl + 1)
    #                    )
    #     )
    #     # C_SPL.append(c_spl)

    #     # c_trunc = _calc_sample_scale(row.N_SAMP_TRUNC, row.OBS_TRUNC)
    #     PVAL_TRUNC.append(nb_model.nb_pvalue_greater_midp(row.N_SAMP_TRUNC, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_TRUNC + 1)
    #                         # 1 / (row.THETA * row.Pi_TRUNC * c_trunc + 1)
    #                    )
    #     )
    #     # C_TRUNC.append(c_trunc)

    #     # c_nonsyn = _calc_sample_scale(row.N_SAMP_NONSYN, row.OBS_NONSYN)
    #     PVAL_NONSYN.append(nb_model.nb_pvalue_greater_midp(row.N_SAMP_NONSYN, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_NONSYN + 1)
    #                         # 1 / (row.THETA * row.Pi_NONSYN * c_nonsyn + 1)
    #                    )
    #     )
    #     # C_NONSYN.append(c_nonsyn)

    # df_model['PVAL_SYN_BURDEN_SAMPLE'] = PVAL_SYN
    # df_model['PVAL_MIS_BURDEN_SAMPLE'] = PVAL_MIS
    # df_model['PVAL_NONS_BURDEN_SAMPLE'] = PVAL_NONS
    # df_model['PVAL_SPL_BURDEN_SAMPLE'] = PVAL_SPL
    # df_model['PVAL_TRUNC_BURDEN_SAMPLE'] = PVAL_TRUNC
    # df_model['PVAL_NONSYN_BURDEN_SAMPLE'] = PVAL_NONSYN

    df_model['PVAL_SYN_BURDEN_SAMPLE'] = nb_model.nb_pvalue_greater_midp(
        df_model.N_SAMP_SYN,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_SYN + 1)
    )
    df_model['PVAL_MIS_BURDEN_SAMPLE'] = nb_model.nb_pvalue_greater_midp(
        df_model.N_SAMP_MIS,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_MIS + 1)
    )
    df_model['PVAL_NONS_BURDEN_SAMPLE'] = nb_model.nb_pvalue_greater_midp(
        df_model.N_SAMP_NONS,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_NONS + 1)
    )
    df_model['PVAL_SPL_BURDEN_SAMPLE'] = nb_model.nb_pvalue_greater_midp(
        df_model.N_SAMP_SPL,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_SPL + 1)
    )
    df_model['PVAL_TRUNC_BURDEN_SAMPLE'] = nb_model.nb_pvalue_greater_midp(
        df_model.N_SAMP_TRUNC,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_TRUNC + 1)
    )
    df_model['PVAL_NONSYN_BURDEN_SAMPLE'] = nb_model.nb_pvalue_greater_midp(
        df_model.N_SAMP_NONSYN,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_NONSYN + 1)
    )

    # df_model['C_SYN_BURDEN_SAMPLE'] = C_SYN
    # df_model['C_MIS_BURDEN_SAMPLE'] = C_MIS
    # df_model['C_NONS_BURDEN_SAMPLE'] = C_NONS
    # df_model['C_SPL_BURDEN_SAMPLE'] = C_SPL
    # df_model['C_TRUNC_BURDEN_SAMPLE'] = C_TRUNC
    # df_model['C_NONSYN_BURDEN_SAMPLE'] = C_NONSYN

    return df_model

def element_pvalue_burden_nb_by_sample(df_model):
    """ Calculate burden P-values based on the transfered NB model params.
        Test based only on the number of *mutated* samples per gene
    """

    # PVAL_MUT = []
    # for i, row in df_model.iterrows():
    #     # c_mut = _calc_sample_scale(row.OBS_SAMPLES, row.OBS_MUT)
    #     PVAL_MUT.append(nb_model.nb_pvalue_greater_midp(row.OBS_SAMPLES, row.ALPHA,
    #                         1 / (row.THETA * row.Pi_SUM + 1)
    #                         # 1 / (row.THETA * row.Pi_SUM * c_mut + 1)
    #                    )
    #     )

    # df_model['PVAL_SAMPLE_BURDEN'] = PVAL_MUT
    df_model['PVAL_SAMPLE_BURDEN'] = nb_model.nb_pvalue_greater_midp(
        df_model.OBS_SAMPLES,
        df_model.ALPHA,
        1 / (df_model.THETA * df_model.Pi_SUM + 1)
    )

    return df_model

def gene_pvalue_burden_dnds(df_model):
    """ Calculate burden P-values based on the dnds-corrected expected values
    """

    PVAL_SYN, PVAL_MIS, PVAL_NONS, PVAL_SPL, PVAL_TRUNC, PVAL_NONSYN = [], [], [], [], [], []
    for i, row in df_model.iterrows():
        PVAL_SYN.append(nb_model.nb_pvalue_greater_midp(row.OBS_SYN, row.ALPHA,
                            1 / (row.EXP_SYN_ML / row.ALPHA + 1)
                       )
        )
        PVAL_MIS.append(nb_model.nb_pvalue_greater_midp(row.OBS_MIS, row.ALPHA,
                            1 / (row.EXP_MIS_ML / row.ALPHA + 1)
                       )
        )
        PVAL_NONS.append(nb_model.nb_pvalue_greater_midp(row.OBS_NONS, row.ALPHA,
                            1 / (row.EXP_NONS_ML / row.ALPHA + 1)
                       )
        )
        PVAL_SPL.append(nb_model.nb_pvalue_greater_midp(row.OBS_SPL, row.ALPHA,
                            1 / (row.EXP_SPL_ML / row.ALPHA + 1)
                       )
        )
        PVAL_TRUNC.append(nb_model.nb_pvalue_greater_midp(row.OBS_TRUNC, row.ALPHA,
                            1 / (row.EXP_TRUNC_ML / row.ALPHA + 1)
                       )
        )
        PVAL_NONSYN.append(nb_model.nb_pvalue_greater_midp(row.OBS_NONSYN, row.ALPHA,
                            1 / (row.EXP_NONSYN_ML / row.ALPHA + 1)
                       )
        )

    df_model['PVAL_SYN_BURDEN_DNDS']    = PVAL_SYN
    df_model['PVAL_MIS_BURDEN_DNDS']    = PVAL_MIS
    df_model['PVAL_NONS_BURDEN_DNDS']   = PVAL_NONS
    df_model['PVAL_SPL_BURDEN_DNDS'] = PVAL_SPL
    df_model['PVAL_TRUNC_BURDEN_DNDS']  = PVAL_TRUNC
    df_model['PVAL_NONSYN_BURDEN_DNDS'] = PVAL_NONSYN

    return df_model

def gene_pvalue_sel_nb(df_model):
    """ Calculate dNdS selection p-values using a conservative NB model
        (NB model integrates over uncertainty in the rate estimate)
    """
    PVAL_SYN, PVAL_MIS, PVAL_TRUNC, PVAL_NONSYN = [], [], [], []
    for i, row in df_model.iterrows():
        p_syn, p_mis, p_trunc, p_nonsyn = _llr_test_nb(row)
        PVAL_SYN.append(p_syn)
        PVAL_MIS.append(p_mis)
        PVAL_TRUNC.append(p_trunc)
        PVAL_NONSYN.append(p_nonsyn)

    df_model['PVAL_SYN_SEL_NB'] = PVAL_SYN
    df_model['PVAL_MIS_SEL_NB'] = PVAL_MIS
    # df_model['PVAL_NONS_SEL_NB'] = PVAL_NONS
    # df_model['PVAL_SPL_SEL_NB'] = PVAL_NONS
    df_model['PVAL_TRUNC_SEL_NB'] = PVAL_TRUNC
    df_model['PVAL_NONSYN_SEL_NB'] = PVAL_NONSYN

    return df_model

def gene_pvalue_indel_by_transfer(df_model):
    ## Length of genes
    df_cds = pd.read_table(
        pkg_resources.resource_filename('DIGDriver', 'data/dndscv_gene_cds.bed.gz'),
        names=['CHROM', 'START', 'END', 'GENE'],
        low_memory=False
    )
    df_cds['LENGTH'] = df_cds.END - df_cds.START
    df_cds_l = df_cds.pivot_table(index='GENE', values='LENGTH', aggfunc=np.sum)
    df_model = df_model.merge(df_cds_l['LENGTH'], left_index=True, right_index=True, how='left')

    ## Probability of indels within each gene under uniform distribution
    df_model['Pi_INDEL'] = df_model.LENGTH / (df_model.R_SIZE)

    ## Non CGC genes for scaling factor
    f_panel = 'data/genes_CGC_ALL.txt'
    df_genes = pd.read_table(pkg_resources.resource_stream('DIGDriver', f_panel), names=['GENE'])
    all_cosmic = df_genes.GENE.to_list() + ['CDKN2A.p14arf', 'CDKN2A.p16INK4a']
    df_model_null = df_model[~df_model.index.isin(all_cosmic)]

    ## Expected Uniform indel rate
    EXP_INDEL_UNIF = (df_model_null.Pi_INDEL * df_model_null.ALPHA * df_model_null.THETA).sum()
    OBS_INDEL = df_model_null.OBS_INDEL.sum()
    t_indel = OBS_INDEL / EXP_INDEL_UNIF
    df_model['THETA_INDEL'] = df_model.THETA * t_indel
    df_model['EXP_INDEL'] = df_model.ALPHA * df_model.THETA_INDEL * df_model.Pi_INDEL

    df_model['PVAL_INDEL_BURDEN'] = [nb_model.nb_pvalue_greater_midp(row.OBS_INDEL, row.ALPHA, 1 / (row.THETA_INDEL*row.Pi_INDEL + 1)) for i, row in df_model.iterrows()]

    return df_model

def gene_pvalue_indel(df_model):
    f_panel = 'data/genes_CGC_ALL.txt'
    df_genes = pd.read_table(pkg_resources.resource_stream('DIGDriver', f_panel), names=['GENE'])
    all_cosmic = df_genes.GENE.to_list() + ['CDKN2A.p14arf', 'CDKN2A.p16INK4a']
    df_model_null = df_model[~df_model.index.isin(all_cosmic)]

    ## Expected Uniform indel rate
    EXP_INDEL_UNIF = (df_model_null.Pi_INDEL * df_model_null.ALPHA_INDEL * df_model_null.THETA_INDEL).sum()
    OBS_INDEL = df_model_null.OBS_INDEL.sum()
    t_indel = OBS_INDEL / EXP_INDEL_UNIF
    df_model['THETA_INDEL'] = df_model.THETA_INDEL * t_indel
    df_model['EXP_INDEL'] = df_model.ALPHA_INDEL * df_model.THETA_INDEL * df_model.Pi_INDEL

    # df_model['PVAL_INDEL_BURDEN'] = [nb_model.nb_pvalue_greater_midp(row.OBS_INDEL, row.ALPHA_INDEL, 1 / (row.THETA_INDEL*row.Pi_INDEL + 1)) for i, row in df_model.iterrows()]
    df_model['PVAL_INDEL_BURDEN'] = nb_model.nb_pvalue_greater_midp(
        df_model.OBS_INDEL,
        df_model.ALPHA_INDEL,
        1 / (df_model.THETA_INDEL * df_model.Pi_INDEL + 1)
    )

    return df_model

def element_pvalue_indel(df_model, t_indel):
    # EXP_INDEL_UNIF = (df_model.Pi_INDEL * df_model.ALPHA_INDEL * df_model.THETA_INDEL).sum()
    # OBS_INDEL = df_model.OBS_INDEL.sum()
    # t_indel_bck = OBS_INDEL / EXP_INDEL_UNIF
    # print(t_indel_bck)

    df_model['THETA_INDEL'] = df_model.THETA_INDEL * t_indel
    df_model['EXP_INDEL'] = df_model.ALPHA_INDEL * df_model.THETA_INDEL * df_model.Pi_INDEL

    # df_model['PVAL_INDEL_BURDEN'] = [nb_model.nb_pvalue_greater_midp(row.OBS_INDEL, row.ALPHA_INDEL, 1 / (row.THETA_INDEL*row.Pi_INDEL + 1)) for i, row in df_model.iterrows()]
    df_model['PVAL_INDEL_BURDEN'] = nb_model.nb_pvalue_greater_midp(
        df_model.OBS_INDEL,
        df_model.ALPHA_INDEL,
        1 / (df_model.THETA_INDEL * df_model.Pi_INDEL + 1)
    )

    return df_model

def gene_pvalue_sel_gamma(df_model):
    """ Calculate dNdS selection p-values using a more aggressive gamma-poisson model
    """
    PVAL_SYN, PVAL_MIS, PVAL_NONS, PVAL_NONSYN = [], [], [], []
    for i, row in df_model.iterrows():
        p_syn, p_mis, p_nons, p_nonsyn = _llr_test_gamma_poiss(row)
        PVAL_SYN.append(p_syn)
        PVAL_MIS.append(p_mis)
        PVAL_NONS.append(p_nons)
        PVAL_NONSYN.append(p_nonsyn)

    df_model['PVAL_SYN_SEL_PG'] = PVAL_SYN
    df_model['PVAL_MIS_SEL_PG'] = PVAL_MIS
    df_model['PVAL_NONS_SEL_PG'] = PVAL_NONS
    df_model['PVAL_NONSYN_SEL_PG'] = PVAL_NONSYN

    return df_model

def annotate_known_genes(df, key='GENE'):
    """ Annotate known driver genes based on existing databases
    """
    ## TODO: Remove hard-coded paths

    ## 1. Load databases
    df_cgc = pd.read_excel('/data/cb/maxas/data/projects/cancer_mutations/DRIVER_DBs/COSMIC_CGC_allMon_Oct_12_18_34_22_2020.xlsx')
    df_oncokb = pd.read_table('/data/cb/maxas/data/projects/cancer_mutations/DRIVER_DBs/OncoKB_cancerGeneList.txt')
    df_bailey = pd.read_excel('/data/cb/maxas/data/projects/cancer_mutations/DRIVER_DBs/Bailey_2018_supplementary_tables.xlsx', sheet_name='Table S1', skiprows=3)
    df_pcawg = pd.read_excel('/data/cb/maxas/data/projects/cancer_mutations/DRIVER_DBs/PCAWG_drivers.xlsx', sheet_name='TableS1_compendium_mutational_d')

    ## 2. Annotate
    df['COSMIC_CGC'] = df[key].isin(df_cgc['Gene Symbol'])
    df['OncoKB'] = df[key].isin(df_oncokb['Hugo Symbol'])
    df['BAILEY'] = df[key].isin(df_bailey['Gene'])
    df['PCAWG'] = df[key].isin(df_pcawg['Gene'])
    df['STATUS'] = df[['COSMIC_CGC', 'OncoKB', 'BAILEY', 'PCAWG']].sum(axis=1)

    return df

# def run_gene_model(f_mut, f_h5_genemodel, f_h5_pretrain, f_cds):
# def run_gene_model(f_mut, f_h5_genemodel, f_cds, pval_burden_nb=True, pval_burden_dnds=True, pval_sel=True, scale_by_sample=False):
def run_gene_model(f_mut, f_h5_genemodel, scale_by_sample=False, pval_burden_nb=True, pval_burden_dnds=True, pval_sel=True,
    max_muts_per_sample=3e9, max_muts_per_gene_per_sample=3e9, scale_factor=None, scale_by_expectation=True, cgc_genes=False):
    """ Run a gene transfer model
    """
    ## 1. Transfer model parameters from pretrained model to new CDS cohort
    df_pretrain = load_pretrained_model(f_h5_genemodel, restrict_cols=True)
    df_mut      = read_mutations_cds(f_mut)

    if cgc_genes:
        f_panel = 'data/genes_{}.txt'.format(cgc_genes)
        print(f_panel)
        df_genes = pd.read_table(pkg_resources.resource_stream('DIGDriver', f_panel), names=['GENE'])
        genes = df_genes.GENE.values
        df_pretrain = df_pretrain[df_pretrain.index.isin(genes)]
        df_mut = df_mut[df_mut.GENE.isin(genes)]

    df_mut = mutation_tools.filter_hypermut_samples(df_mut, max_muts_per_sample)
    df_cnt = mutation_tools.mutations_per_gene(df_mut, max_muts_per_gene_per_sample=max_muts_per_gene_per_sample)
    # df_mut_NOSPL     = read_mutations_cds(f_mut, f_cds)

    if scale_by_expectation:
        # print('scaling by expected number of mutations')
        # exp_mut = (df_pretrain.MU * (df_pretrain.Pi_SYN + df_pretrain.Pi_MIS + df_pretrain.Pi_TRUNC)).sum()
        # cj = len(df_mut) / exp_mut
        print('scaling by expected synonymous mutations (excluding TP53)')
        exp_mut = (df_pretrain[df_pretrain.index != 'TP53'].MU * df_pretrain[df_pretrain.index != 'TP53'].Pi_SYN).sum()
        cj = len(df_mut[(df_mut.GENE != 'TP53') & (df_mut.ANNOT == 'Synonymous')]) / exp_mut
    elif scale_factor:
        cj = scale_factor
    elif scale_by_sample:
        # cj = scale_factor_by_samples(f_h5_genemodel, df_mut)
        cj = calc_scale_factor(df_mut, f_h5_genemodel, scale_type='sample')
    else:  ## Scale by number of mutations
        # cj = scale_factor_by_cds(f_h5_genemodel, df_mut)
        cj = calc_scale_factor(df_mut, f_h5_genemodel, scale_type='exome')

    print("\tScaling factor is: {}".format(cj))

    # print(cj)
    df_model = transfer_gene_model(df_mut, df_cnt, df_pretrain, cj)

    ## 2. Calculate expected values for new cohort
    df_model = gene_expected_muts_nb(df_model)

    # if pval_burden_dnds or pval_sel:
    #     df_model = gene_expected_muts_dnds(df_model)

    ## 3. Calculate p-values
    ## Burden p-values
    if pval_burden_nb:
        print("\tCalculating burden p-values")
        df_model = gene_pvalue_burden_nb(df_model)

        ## Sample burden p-value
        df_model = gene_pvalue_burden_nb_by_sample(df_model)

    # if pval_burden_dnds:
    #     print("\tCalculating dnds-adjusted burden p-values")
    #     df_model = gene_pvalue_burden_dnds(df_model)


    # ## Selection p-values
    # if pval_sel:
    #     print("\tCalculating selection p-values")
    #     df_model = gene_pvalue_sel_nb(df_model)

    if df_model.OBS_INDEL.sum() != 0:
        print("\tCalculating indel burden p-values")
        df_model = gene_pvalue_indel(df_model)

        ## Combine SNV and indel p-values
        x2 = -2 * (np.log(df_model.PVAL_TRUNC_BURDEN) + np.log(df_model.PVAL_INDEL_BURDEN))
        df_model['PVAL_MUT_BURDEN'] = scipy.stats.chi2.sf(x2, df=4)
        # df_model = gene_pvalue_indel_by_transfer(df_model)
        # df_model['PVAL_LOF_BURDEN'] = [
        #     scipy.stats.combine_pvalues([row.PVAL_TRUNC_BURDEN, row.PVAL_INDEL_BURDEN],
        #         method='fisher'
        #     )[1]
        #     for i, row in df_model.iterrows()
        # ]

    ## 4. Annotate with gene driver databases
    # df_model['GENE'] = [s.split('.')[0] for s in df_model.index]
    # df_model = annotate_known_genes(df_model, key='GENE')

    return df_model

def run_target_model(f_mut, f_h5_genemodel, scale_by_sample=False, panel="MSK_341",
    max_muts_per_sample=3e9, max_muts_per_gene_per_sample=3e9, drop_synonymous=True, cgc_genes=False, scale_factor=None):
    """ Analyze MSK-IMPACT genes with a pretrained gene model
    """
    ## 0. Load genes in panel
    print(panel)
    f_panel = 'data/genes_{}.txt'.format(panel)
    df_genes = pd.read_table(pkg_resources.resource_stream('DIGDriver', f_panel), names=['GENE'])
    genes1 = df_genes.GENE.values
    if cgc_genes:
        f_panel = 'data/genes_{}.txt'.format(cgc_genes)
        print(f_panel)
        # f_panel = 'data/genes_CGC.txt'
        # f_panel = 'data/genes_CGC_ONCOGENES.txt'
    df_genes = pd.read_table(pkg_resources.resource_stream('DIGDriver', f_panel), names=['GENE'])
    genes = df_genes.GENE.values

    ## 1. Transfer gene model
    df_mut      = read_mutations_cds(f_mut)
    df_mut = df_mut[df_mut.GENE.isin(genes)]
    if drop_synonymous:
        df_mut = df_mut[df_mut.ANNOT != 'Synonymous']

    df_mut, sample_blacklist = mutation_tools.filter_hypermut_samples(df_mut, max_muts_per_sample, return_blacklist=True)
    df_cnt = mutation_tools.mutations_per_gene(df_mut, max_muts_per_gene_per_sample=max_muts_per_gene_per_sample)

    N_MUT = len(df_mut[(df_mut.ANNOT != 'Synonymous') & (df_mut.ANNOT != 'Essential_Splice') & (df_mut.ANNOT != "Noncoding")])
    N_SAMPLE = len(df_mut.SAMPLE.unique())
    # N_MUT_SAMPLE = df_mut_dedup.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts().sum()

    df_pretrain =  load_pretrained_model(f_h5_genemodel)
    df_pretrain = df_pretrain.loc[df_pretrain.index.isin(genes), :]
    print(len(df_pretrain))

    # df_mut = mutation_tools.read_mutation_file(f_mut)
    # df_mut = df_mut[df_mut.ANNOT != 'Noncoding']
    # df_mut = df_mut[df_mut.GENE.isin(genes)]
    #
    df_mut_dedup = mutation_tools.read_mutation_file(f_mut, drop_duplicates=True)
    df_mut_dedup = df_mut_dedup[~df_mut_dedup.SAMPLE.isin(sample_blacklist)]
    # df_mut_dedup = df_mut_dedup[df_mut_dedup.SAMPLE.isin(df_mut.SAMPLE)]
    # df_mut_dedup = mutation_tools.drop_duplicate_mutations(df_mut)
    df_mut_dedup = df_mut_dedup[(df_mut_dedup.ANNOT != 'Noncoding') & \
                                (df_mut_dedup.ANNOT != 'Synonymous') & \
                                (df_mut_dedup.ANNOT != 'Essential_Splice')]
    print(f_mut, df_mut_dedup.shape)
    df_mut_dedup = df_mut_dedup[df_mut_dedup.GENE.isin(genes1)]

    N_MUT = len(df_mut_dedup)
    N_SAMPLE = len(df_mut_dedup.SAMPLE.unique())
    N_MUT_SAMPLE = df_mut_dedup.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts().sum()

    # df_mut_pre = mutation_tools.read_mutation_file(f_mut_pre, drop_duplicates=True)
    # df_mut_pre = df_mut_pre[(df_mut_pre.ANNOT != 'Noncoding') & (df_mut_pre.ANNOT != 'Synonymous') & (df_mut_pre.ANNOT != 'Essential_Splice')]
    # df_mut_pre_targ = df_mut_pre[df_mut_pre.GENE.isin(genes)]

    with h5py.File(f_h5_genemodel, 'r') as h5:
        # N_SAMPLE_PRE = h5.attrs['N_SAMPLES']
        N_MUT_MSK = h5.attrs['N_MUT_{}'.format(panel)]
        N_MUT_SAMPLE_MSK = h5.attrs['N_MUT_SAMPLE_{}'.format(panel)]
        N_SAMPLE_MSK = h5.attrs['N_SAMPLE_{}'.format(panel)]

    cj = N_MUT / N_MUT_MSK
    # cj = len(df_mut) / N_MUT_MSK
    if scale_factor:
        cj = scale_factor
    elif scale_by_sample:
        # N_SAMPLE = len(df_mut_dedup.SAMPLE.unique())
        # cj = N_SAMPLE / N_SAMPLE_PRE
        print(N_SAMPLE, N_SAMPLE_MSK)
        cj = N_SAMPLE / N_SAMPLE_MSK
        # cj = N_MUT_SAMPLE / N_MUT_SAMPLE_MSK
    else:
       cj = N_MUT / N_MUT_MSK

    print("\tScaling factor is: {}".format(cj))

    # print(cj_mut)
    # print(cj_sample)
    # print(len(df_mut_dedup) / N_MUT_MSK)

    df_model = transfer_gene_model(df_mut, df_cnt, df_pretrain, cj)
    df_model = df_model.loc[df_model.index.isin(genes), :]

    ## 2. Calculate expected values
    df_model = gene_expected_muts_nb(df_model)

    ## 3. Caclulate p-values
    df_model = gene_pvalue_burden_nb(df_model)
    df_model = gene_pvalue_burden_nb_by_sample(df_model)

    return df_model

def run_element_region_model(f_mut, f_bed, f_h5_pretrain, pretrain_key, scale_factor=None, scale_factor_indel=None, scale_type="genome",
    scale_by_expectation=True, max_muts_per_sample=3e9, max_muts_per_elt_per_sample=3e9, skip_pvals=False):
    """ Run a model based on an arbitrary, user-defined set of regions

    Args:
        f_mut: path to mutation file
        f_bed: path to bed file of annotations
        f_h5_pretrain: path to model pretrained on f_bed
        pretrain_key: key under which pretrained model is stored in f_h5_pretrain
        scale_type: how to calculate the scaling factor
    """

    ## 1. Transfer element model
    df_pretrain =  load_pretrained_model(f_h5_pretrain, key=pretrain_key, restrict_cols=True)

    print('Tabulating mutations')
    df_mut_tab, blacklist = mutation_tools.tabulate_mutations_in_element(f_mut, f_bed, bed12=True, drop_duplicates=True,
        max_muts_per_sample=max_muts_per_sample, max_muts_per_elt_per_sample=max_muts_per_elt_per_sample, return_blacklist=True
    )

    if scale_by_expectation:
        # obs_mut = df_mut_tab.OBS_MUT.sum()
        # exp_mut = (df_pretrain.MU * df_pretrain.Pi_SUM).sum()
        # cj = obs_mut / exp_mut
        # print('scaling by expected synonymous mutations (excluding TP53)')
        # df_gene = load_pretrained_model(f_h5_pretrain)
        # df_mut = mutation_tools.read_mutation_file(f_mut, drop_duplicates=False)
        print('scaling by expected number of mutations')
        df_gene = load_pretrained_model(f_h5_pretrain)
        # f_bed = pkg_resources.resource_filename('DIGDriver', 'data/genes.MARTINCORENA.bed')
        df_mut = read_mutations_cds(f_mut)
        # df_mut = mutation_tools.restrict_mutations_by_bed_efficient(f_mut, f_bed, drop_duplicates=False, bed12=True)
        df_mut = df_mut[~df_mut.SAMPLE.isin(blacklist)]
        # exp_mut = (df_gene.MU * (df_gene.Pi_SYN + df_gene.Pi_MIS + df_gene.Pi_TRUNC)).sum()
        # cj = len(df_mut) / exp_mut
        df_syn = df_mut[(df_mut.ANNOT == 'Synonymous') & (df_mut.GENE != 'TP53')].drop_duplicates()
        exp_syn = (df_gene[df_gene.index != 'TP53'].MU * df_gene[df_gene.index != 'TP53'].Pi_SYN).sum()
        cj = len(df_syn) / exp_syn
        # cj = len(df_mut[(df_mut.GENE != 'TP53') & (df_mut.ANNOT == 'Synonymous')]) / exp_mut

        ## INDEL scaling factor
        f_panel = 'data/genes_CGC_ALL.txt'
        genes = pd.read_table(pkg_resources.resource_stream('DIGDriver', f_panel), names=['GENE'])
        all_cosmic = genes.GENE.to_list() + ['CDKN2A.p14arf', 'CDKN2A.p16INK4a']
        df_gene_null = df_gene[~df_gene.index.isin(all_cosmic)]
        df_mut_null = df_mut[~df_mut.index.isin(all_cosmic)]
        EXP_INDEL_UNIF = (df_gene_null.Pi_INDEL * df_gene_null.ALPHA_INDEL * df_gene_null.THETA_INDEL).sum()
        OBS_INDEL = len(df_mut_null[df_mut_null.ANNOT == 'INDEL'])
        cj_indel = OBS_INDEL / EXP_INDEL_UNIF

    elif scale_type == 'PCAWG_cds':
        assert (pretrain_key == 'PCAWG_cds'), \
            "ERROR: can only scale by PCAWG_cds if the loaded reference model is PCAWG_cds. Specify <KEY> as \"PCAWG_cds\" and rerun."

        f_panel = 'data/genes_CGC_ALL.txt'
        genes = pd.read_table(pkg_resources.resource_stream('DIGDriver', f_panel), names=['GENE'])
        all_cosmic = genes.GENE.to_list() + ['CDKN2A.p14arf', 'CDKN2A.p16INK4a']

        df_pretrain['GENE'] = [elt.split('::')[2] for elt in df_pretrain.index]
        df_pretrain_null = df_pretrain[~df_pretrain.GENE.isin(all_cosmic)]
        exp_snv = (df_pretrain_null.MU * df_pretrain_null.Pi_SUM).sum()
        exp_ind = (df_pretrain_null.MU_INDEL * df_pretrain_null.Pi_INDEL).sum()

        df_mut_tab['GENE'] = [elt.split('::')[2] for elt in df_mut_tab.index]
        df_tab_null = df_mut_tab[~df_mut_tab.GENE.isin(all_cosmic)]
        obs_snv = df_tab_null.OBS_SNV.sum()
        obs_ind = df_tab_null.OBS_INDEL.sum()

        cj = obs_snv / exp_snv
        cj_indel = obs_ind / exp_ind

        # df_gene = load_pretrained_model(f_h5_pretrain, 'PCAWG_cds')
        # print('Bootstrapping the scaling factors...')
        # cj_snv_lst = []
        # cj_indel_lst = []
        # for i in range(100):
        #     df_pre_tmp = df_pretrain.sample(frac=1, replace=True)
        #     df_tab_tmp = df_mut_tab.loc[df_mut_tab.index.isin(df_pre_tmp.index)]

        #     exp_snv = (df_pre_tmp.MU * df_pre_tmp.Pi_SUM).sum()
        #     exp_indel = (df_pre_tmp.MU_INDEL * df_pre_tmp.Pi_INDEL).sum()

        #     obs_snv = df_tab_tmp.OBS_SNV.sum()
        #     obs_indel = df_tab_tmp.OBS_INDEL.sum()

        #     cj_snv_lst.append(obs_snv/exp_snv)
        #     cj_indel_lst.append(obs_indel/exp_indel)

        # cj = np.median(cj_snv_lst)
        # cj_indel = np.median(cj_indel_lst)

    elif scale_factor:
        cj = scale_factor
        cj_indel = scale_factor_indel
    else:
        print('Calculating scale factor')
        cj, cj_indel = calc_scale_factor_efficient(f_mut, f_h5_pretrain, scale_type=scale_type)
    # cj = calc_scale_factor(df_mut, f_h5_pretrain, scale_type=scale_type)
    print("\tScale factor is: {}".format(cj))
    print("\tINDEL scale factor is: {}".format(cj_indel))

    df_model = transfer_element_model_with_indels(df_mut_tab, df_pretrain, cj)

    print('Calculating statistics')
    df_model = element_expected_muts_nb(df_model)

    if not skip_pvals:
        # df_model = element_pvalue_burden_nb(df_model)
        # df_model = element_pvalue_burden_nb_by_sample(df_model)
        df_model = element_pvalue_burden_nb(df_model)
        df_model = element_pvalue_burden_nb_by_sample(df_model)

        if df_model.OBS_INDEL.sum() != 0:
            print("\tCalculating indel burden p-values")
            df_model = element_pvalue_indel(df_model, cj_indel)

            ## Combine SNV and indel p-values
            x2 = -2 * (np.log(df_model.PVAL_SNV_BURDEN) + np.log(df_model.PVAL_INDEL_BURDEN))
            df_model['PVAL_MUT_BURDEN'] = scipy.stats.chi2.sf(x2, df=4)
            # df_model['PVAL_MUT_BURDEN'] = [
            #     scipy.stats.combine_pvalues([row.PVAL_SNV_BURDEN, row.PVAL_INDEL_BURDEN], 
            #         method='fisher'
            #     )[1] 
            #     for i, row in df_model.iterrows()
            # ]

    return df_model


def run_sites_region_model(f_mut, f_sites, f_h5_pretrain, pretrain_key, scale_factor=None, scale_type="genome", scale_by_expectation=True):
    """ Run a model based on an arbitrary, user-defined set of regions and a arbitrary set of sites of interest within those regions

    Args:
        f_mut: path to mutation file
        f_bed: path to bed12 file of annotations
        f_sites: path to sites file
        f_h5_pretrain: path to model pretrained on f_bed
        pretrain_key: key under which pretrained model is stored in f_h5_pretrain
        scale_type: how to calculate the scaling factor
    """


    ## 1. Transfer element model
    df_pretrain =  load_pretrained_model(f_h5_pretrain, key=pretrain_key, restrict_cols=True)

    if scale_by_expectation:
        print('scaling by expected synonymous mutations (excluding TP53)')
        df_gene = load_pretrained_model(f_h5_pretrain)
        df_mut = mutation_tools.read_mutation_file(f_mut, drop_duplicates=False)
        exp_mut = (df_gene[df_gene.index != 'TP53'].MU * df_gene[df_gene.index != 'TP53'].Pi_SYN).sum()
        cj = len(df_mut[(df_mut.GENE != 'TP53') & (df_mut.ANNOT == 'Synonymous')]) / exp_mut
        # print('scaling by expected number of mutations')
        # df_gene = load_pretrained_model(f_h5_pretrain)
        # df_mut = read_mutations_cds(f_mut)
        # exp_mut = (df_gene.MU * (df_gene.Pi_SYN + df_gene.Pi_MIS + df_gene.Pi_TRUNC)).sum()
        # cj = len(df_mut) / exp_mut
    elif scale_factor:
        cj = scale_factor
    elif scale_type == 'MSK_230':
        print('Scaling by samples in MSK 230 gene subset.')
        panel = 'MSK_230'
        f_panel = 'data/genes_{}.txt'.format(panel)
        df_genes = pd.read_table(pkg_resources.resource_stream('DIGDriver', f_panel), names=['GENE'])
        genes = df_genes.GENE.values
        df_mut_dedup = mutation_tools.read_mutation_file(f_mut, drop_duplicates=True)
        # df_mut_dedup = df_mut_dedup[df_mut_dedup.SAMPLE.isin(df_mut.SAMPLE)]
        df_mut_dedup = df_mut_dedup[(df_mut_dedup.ANNOT != 'Noncoding') & \
                                    (df_mut_dedup.ANNOT != 'Synonymous') & \
                                    (df_mut_dedup.ANNOT != 'Essential_Splice')]
        print(f_mut, df_mut_dedup.shape)
        df_mut_dedup = df_mut_dedup[df_mut_dedup.GENE.isin(genes)]

        N_MUT = len(df_mut_dedup)
        N_SAMPLE = len(df_mut_dedup.SAMPLE.unique())
        N_MUT_SAMPLE = df_mut_dedup.groupby(['GENE', 'SAMPLE']).size().reset_index(name='CNT').GENE.value_counts().sum()

        with h5py.File(f_h5_pretrain, 'r') as h5:
            N_MUT_MSK = h5.attrs['N_MUT_{}'.format(panel)]
            N_SAMPLE_MSK = h5.attrs['N_SAMPLE_{}'.format(panel)]

        # cj = N_MUT / N_MUT_MSK
        print(N_SAMPLE, N_SAMPLE_MSK)
        cj = N_SAMPLE / N_SAMPLE_MSK

    else:
        print('Calculating scale factor')
        cj = calc_scale_factor_efficient(f_mut, f_h5_pretrain, scale_type=scale_type)
    print("\tScale factor is: {}".format(cj))

    print('Tabulating mutations')

    df_mut_tab = mutation_tools.tabulate_sites_in_element(f_sites, f_mut)

    df_model = transfer_element_model(df_mut_tab, df_pretrain, cj, use_chrom = False)

    print('Calculating statistics')
    df_model = element_expected_muts_nb(df_model)
    df_model = element_pvalue_burden_nb(df_model)
    df_model = element_pvalue_burden_nb_by_sample(df_model)

    return df_model


def _llr_test_nb(row):
    ## Calculate likelihood under no selection for any variants
    ll0 = _ll_nb(row.OBS_SYN, row.ALPHA,  row.THETA * row.Pi_SYN  * row.MRFOLD) + \
          _ll_nb(row.OBS_MIS, row.ALPHA,  row.THETA * row.Pi_MIS  * row.MRFOLD) + \
          _ll_nb(row.OBS_TRUNC, row.ALPHA, row.THETA * row.Pi_TRUNC * row.MRFOLD)
          # _ll_nb(row.OBS_NONS, row.ALPHA, row.THETA * row.Pi_NONS * row.MRFOLD) + \
          # _ll_nb(row.OBS_SPL, row.ALPHA, row.THETA * row.Pi_SPL * row.MRFOLD)

    ## Calculate likelihood under selection for SYN variants
    ll1 = _ll_nb(row.OBS_SYN, row.ALPHA,  row.OBS_SYN / row.ALPHA) + \
          _ll_nb(row.OBS_MIS, row.ALPHA,  row.THETA * row.Pi_MIS  * row.MRFOLD) + \
          _ll_nb(row.OBS_TRUNC, row.ALPHA, row.THETA * row.Pi_TRUNC * row.MRFOLD)
          # _ll_nb(row.OBS_NONS, row.ALPHA, row.THETA * row.Pi_NONS * row.MRFOLD) + \
          # _ll_nb(row.OBS_SPL, row.ALPHA, row.THETA * row.Pi_SPL * row.MRFOLD)

    ## Caclulate likelihood under selection for MIS variants
    ll2 = _ll_nb(row.OBS_SYN, row.ALPHA, row.THETA * row.Pi_SYN * row.MRFOLD) + \
          _ll_nb(row.OBS_MIS, row.ALPHA, row.OBS_MIS / row.ALPHA) + \
          _ll_nb(row.OBS_TRUNC, row.ALPHA, row.THETA * row.Pi_TRUNC * row.MRFOLD)
          # _ll_nb(row.OBS_NONS, row.ALPHA, row.THETA * row.Pi_NONS * row.MRFOLD) + \
          # _ll_nb(row.OBS_SPL, row.ALPHA, row.THETA * row.Pi_SPL * row.MRFOLD)

    ## Caclulate likelihood under selection for TRUNC variants
    ll3 = _ll_nb(row.OBS_SYN, row.ALPHA, row.THETA * row.Pi_SYN * row.MRFOLD) + \
          _ll_nb(row.OBS_MIS, row.ALPHA,  row.THETA * row.Pi_MIS  * row.MRFOLD) + \
          _ll_nb(row.OBS_TRUNC, row.ALPHA, row.OBS_TRUNC / row.ALPHA)
          # _ll_nb(row.OBS_NONS, row.ALPHA, row.OBS_NONS / row.ALPHA) + \
          # _ll_nb(row.OBS_SPLCE, row.ALPHA, row.OBS_SPL / row.ALPHA)

    ## Caclulate likelihood under selection for both MIS and TRUNC variants
    ll4 = _ll_nb(row.OBS_SYN, row.ALPHA, row.THETA * row.Pi_SYN * row.MRFOLD) + \
          _ll_nb(row.OBS_MIS, row.ALPHA, row.OBS_MIS / row.ALPHA) + \
          _ll_nb(row.OBS_TRUNC, row.ALPHA, row.OBS_TRUNC / row.ALPHA)
          # _ll_nb(row.OBS_NONS, row.ALPHA, row.OBS_NONS / row.ALPHA)

    ## Likelihood ratio p-values
    p_syn  = scipy.stats.chi2.sf(-2 * (ll0 - ll1), df=1)
    p_mis  = scipy.stats.chi2.sf(-2 * (ll0 - ll2), df=1)
    p_trunc = scipy.stats.chi2.sf(-2 * (ll0 - ll3), df=1)
    p_nsyn = scipy.stats.chi2.sf(-2 * (ll0 - ll4), df=2)

    return p_syn, p_mis, p_trunc, p_nsyn

def _llr_test_gamma_poiss(row):
    ## Calculate likelihood under no selection for any variants
    ll0 = _ll_pois(row.OBS_SYN, row.ALPHA * row.THETA * row.Pi_SYN  * row.MRFOLD) + \
          _ll_pois(row.OBS_MIS, row.ALPHA * row.THETA * row.Pi_MIS  * row.MRFOLD) + \
          _ll_pois(row.OBS_NONS, row.ALPHA * row.THETA * row.Pi_NONS * row.MRFOLD) +\
          _ll_gamma(row.T_SYN, row.ALPHA, row.THETA * row.Pi_SYN * row.MRFOLD)

    ## Calculate likelihood under selection for SYN variants
    ll1 = _ll_pois(row.OBS_SYN, row.OBS_SYN) + \
          _ll_pois(row.OBS_MIS, row.ALPHA * row.THETA * row.Pi_MIS  * row.MRFOLD) + \
          _ll_pois(row.OBS_NONS, row.ALPHA * row.THETA * row.Pi_NONS * row.MRFOLD) +\
          _ll_gamma(row.T_SYN, row.ALPHA, row.THETA * row.Pi_SYN * row.MRFOLD)

    ## Caclulate likelihood under selection for MIS variants
    ll2 = _ll_pois(row.OBS_SYN, row.ALPHA * row.THETA * row.Pi_SYN  * row.MRFOLD) + \
          _ll_pois(row.OBS_MIS, row.OBS_MIS) + \
          _ll_pois(row.OBS_NONS, row.ALPHA * row.THETA * row.Pi_NONS * row.MRFOLD) +\
          _ll_gamma(row.T_SYN, row.ALPHA, row.THETA * row.Pi_SYN * row.MRFOLD)

    ## Caclulate likelihood under selection for both NONS variants
    ll3 = _ll_pois(row.OBS_SYN, row.ALPHA * row.THETA * row.Pi_SYN  * row.MRFOLD) + \
          _ll_pois(row.OBS_MIS, row.ALPHA * row.THETA * row.Pi_MIS  * row.MRFOLD) + \
          _ll_pois(row.OBS_NONS, row.OBS_NONS) +\
          _ll_gamma(row.T_SYN, row.ALPHA, row.THETA * row.Pi_SYN * row.MRFOLD)

    ## Caclulate likelihood under selection for both MIS and NONS variants
    ll4 = _ll_pois(row.OBS_SYN, row.ALPHA * row.THETA * row.Pi_SYN  * row.MRFOLD) + \
          _ll_pois(row.OBS_MIS, row.OBS_MIS) + \
          _ll_pois(row.OBS_NONS, row.OBS_NONS) +\
          _ll_gamma(row.T_SYN, row.ALPHA, row.THETA * row.Pi_SYN * row.MRFOLD)

    ## Likelihood ratio p-values
    p_syn  = scipy.stats.chi2.sf(-2 * (ll0 - ll1), df=1)
    p_mis  = scipy.stats.chi2.sf(-2 * (ll0 - ll2), df=1)
    p_nons = scipy.stats.chi2.sf(-2 * (ll0 - ll3), df=1)
    p_nsyn = scipy.stats.chi2.sf(-2 * (ll0 - ll4), df=2)

    return p_syn, p_mis, p_nons, p_nsyn

def _ll_nb(k, alpha, theta):
    p = 1 / (1 + theta)
    return scipy.stats.nbinom.logpmf(k, alpha, p)

def _ll_pois(k, lam):
    return scipy.stats.poisson.logpmf(k, lam)

def _ll_gamma(lam, alpha, theta):
    return scipy.stats.gamma.logpdf(lam, alpha, scale=theta)

def _mle_t(n_neut, exp_rel_neut, alpha, theta):
    """ Maximum likelihood estimator for dNdS rate of neutral mutations
    """
    tml = (n_neut + alpha - 1) / (exp_rel_neut + (1/theta))

    if alpha <= 1:
        tml = max(alpha * theta, tml)

    return tml

def _mrfold_factor(opt_t, exp_syn):
    """ dNdS mutation rate correction factor
    """
    return max(1e-10, opt_t / exp_syn)


def selection_coefficient(df_model, mut_type, pvalue=True):
    obs_key = 'OBS_{}'.format(mut_type)
    exp_key = 'EXP_{}'.format(mut_type)
    sel_key = 'SEL_{}'.format(mut_type)

    df_model[sel_key] = (df_model[obs_key] + 1e-16) / (df_model[exp_key] + 1e-16)

    if pvalue:
        pi_key = 'Pi_{}'.format(mut_type)
        pval_key = 'PVAL_{}_SEL'.format(mut_type)
        ll0 = scipy.stats.nbinom.logpmf(df_model[obs_key], df_model.ALPHA, 1 / (1 + df_model.THETA * df_model[pi_key]))
        ll1 = scipy.stats.nbinom.logpmf(df_model[obs_key], df_model.ALPHA, 1 / (1 + df_model.THETA * df_model[pi_key] * df_model[sel_key]))
        df_model[pval_key] = scipy.stats.chi2.sf(-2 * (ll0 - ll1), df=1)
