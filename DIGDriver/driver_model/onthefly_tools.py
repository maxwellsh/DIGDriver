import pandas as pd
import numpy as np
import pysam
import multiprocessing as mp
import pybedtools
import pkg_resources
import h5py
import scipy
import tempfile
import os

from DIGDriver.sequence_model import genic_driver_tools
from DIGDriver.sequence_model import sequence_tools
from DIGDriver.sequence_model import nb_model
from DIGDriver.driver_model import transfer_tools
from DIGDriver.data_tools import mutation_tools


def region_str_to_params(region_str):
    col_split = region_str.split(":")
    chrom = col_split[0].lstrip("chr")
    #chrom = col_split[0]
    pos_split = col_split[1].split("-")
    start = int(pos_split[0])
    end = int(pos_split[1])
    return chrom, start, end

def DIG_onthefly(f_pretrained, f_mut, f_fasta, f_elts_bed=None, region_str=None,
scale_factor=None, scale_factor_indel=None, scale_type="genome", scale_by_expectation=True,
 max_muts_per_sample=3e9, max_muts_per_elt_per_sample=3e9, skip_pvals=False):
    assert f_elts_bed or region_str, "ERROR: you must provide --f-bed or --region_str."

    if region_str:
        temp_file, temp_name = tempfile.mkstemp()

        CHROM,START,END = region_str_to_params(region_str)
        os.write(temp_file, "{}\t{}\t{}\tUserELT\t0\t+\t0\t0\t.\t1\t{},\t0,".format(CHROM,START,END,END-START).encode())
        os.close(temp_file)
        f_elts_bed = temp_name

    print('Tabulating mutations')
    df_mut_tab, blacklist = mutation_tools.tabulate_mutations_in_element(f_mut, f_elts_bed, bed12=True, drop_duplicates=True, all_elements = True,
        max_muts_per_sample=max_muts_per_sample, max_muts_per_elt_per_sample=max_muts_per_elt_per_sample, return_blacklist=True
    )
    if scale_by_expectation:
        print('scaling by expected number of mutations')
        df_gene = transfer_tools.load_pretrained_model(f_pretrained)
        df_mut = transfer_tools.read_mutations_cds(f_mut)
        df_mut = df_mut[~df_mut.SAMPLE.isin(blacklist)]
        df_syn = df_mut[(df_mut.ANNOT == 'Synonymous') & (df_mut.GENE != 'TP53')].drop_duplicates()
        exp_syn = (df_gene[df_gene.index != 'TP53'].MU * df_gene[df_gene.index != 'TP53'].Pi_SYN).sum()
        cj = len(df_syn) / exp_syn

        ## INDEL scaling factor
        f_panel = 'data/genes_CGC_ALL.txt'
        genes = pd.read_table(pkg_resources.resource_stream('DIGDriver', f_panel), names=['GENE'])
        all_cosmic = genes.GENE.to_list() + ['CDKN2A.p14arf', 'CDKN2A.p16INK4a']
        df_gene_null = df_gene[~df_gene.index.isin(all_cosmic)]
        df_mut_null = df_mut[~df_mut.index.isin(all_cosmic)]
        EXP_INDEL_UNIF = (df_gene_null.Pi_INDEL * df_gene_null.ALPHA_INDEL * df_gene_null.THETA_INDEL).sum()
        OBS_INDEL = len(df_mut_null[df_mut_null.ANNOT == 'INDEL'])
        cj_indel = OBS_INDEL / EXP_INDEL_UNIF
    elif scale_factor:
        cj = scale_factor
        cj_indel = scale_factor_indel
    else:
        print('Calculating scale factor')
        cj, cj_indel = transfer_tools.calc_scale_factor_efficient(f_mut, f_pretrained, scale_type=scale_type)

    L_contexts = sequence_tools.precount_region_contexts_parallel(
        f_elts_bed, f_fasta, 10, 10000, sub_elts = True, n_up=1, n_down=1)


    all_windows_df = pd.read_hdf(f_pretrained, 'region_params')
    window = all_windows_df.iloc[0][2]-all_windows_df.iloc[0][1]
    window_key = 'window_{}'.format(window)

    df_mut = pd.read_hdf(f_pretrained, key='sequence_model_192')
    mut_model_idx = [r[1] + '>' + r[1][0] + r[0][2] + r[1][2] for r in zip(df_mut.MUT_TYPE, df_mut.CONTEXT)]
    subst_idx = sorted(mut_model_idx)
    revc_subst_idx = [sequence_tools.reverse_complement(sub.split('>')[0]) + '>' + sequence_tools.reverse_complement(sub.split('>')[\
    -1]) for sub in subst_idx]
    revc_dic = dict(zip(subst_idx, revc_subst_idx))

    d_pr = pd.DataFrame(df_mut.FREQ.values, mut_model_idx)
    d_pr = d_pr.sort_index()[0].values

    df_elts = mutation_tools.bed12_boundaries(f_elts_bed)


    elt_lst = []
    mu_lst = []
    sigma_lst = []
    R_obs_lst = []
    alpha_lst = []
    theta_lst = []
    p_mut_lst = []
    flag_lst = []

    mu_ind_lst = []
    sigma_ind_lst = []
    R_size_lst = []
    elt_len_lst = []
    alpha_ind_lst = []
    theta_ind_lst = []
    p_ind_lst = []
    R_ind_lst=[]

    for _, row in df_elts.iterrows():

        chrom = row['CHROM']
        elt = row['ELT']
        strand = row['STRAND']
        block_starts = row['BLOCK_STARTS']
        block_ends = row['BLOCK_ENDS']
        elts_as_intervals = np.vstack((block_starts, block_ends))
        overlaps = genic_driver_tools.get_ideal_overlaps(chrom, elts_as_intervals, window)

        chrom_lst, start_lst, end_lst = ['chr' + str(r[0]) for r in overlaps], [r[1] for r in overlaps], [r[2] for r in overlaps]
        region_df = sequence_tools.count_contexts_by_regions(f_fasta, chrom_lst, start_lst, end_lst, n_up=1, n_down=1)
        region_counts = np.array([np.repeat(region, 3) for region in region_df.values]).sum(axis=0)

#         #if negative strand, take the reverse complement of the region counts
        if strand == '-1' or strand == '-':
            region_counts = np.array([r[1] for r in sorted(enumerate(region_counts), key=lambda k: revc_dic[subst_idx[k[0]]])])

        L = np.zeros((192))
        for start, end in zip(block_starts, block_ends):
            L += L_contexts.loc['chr{}:{}-{}'.format(chrom, start,end)].values

        prob_sum = region_counts * d_pr

        t_pi = d_pr / prob_sum.sum()

        p_mut = (t_pi * L).sum()

        p_mut_lst.append(p_mut)
        mu, sigma, R_obs, FLAG = genic_driver_tools.get_region_params_direct(all_windows_df, overlaps, window)
        alpha, theta = nb_model.normal_params_to_gamma(mu, sigma)
        theta = theta * cj

        flag_lst.append(FLAG)
        R_size_lst.append(int(region_counts.sum() / 3))  ## length of region containing gene

        elt_len_lst.append(int(np.sum(L) / 3))
        p_ind_lst.append(elt_len_lst[-1] / R_size_lst[-1])


        mu_ind,sigma_ind,R_ind = mu, sigma, R_obs
        alpha_ind, theta_ind = nb_model.normal_params_to_gamma(mu_ind, sigma_ind)
        theta_ind = theta_ind * cj_indel

        alpha_ind_lst.append(alpha_ind)
        theta_ind_lst.append(theta_ind)
        mu_ind_lst.append(mu_ind)
        sigma_ind_lst.append(sigma_ind)

        R_ind_lst.append(R_ind)
        elt_lst.append(elt)
        mu_lst.append(mu)
        sigma_lst.append(sigma)
        R_obs_lst.append(R_obs)
        alpha_lst.append(alpha)
        theta_lst.append(theta)


    pretrain_df = pd.DataFrame({'ELT_SIZE':elt_len_lst, 'FLAG': flag_lst, 'R_SIZE':R_size_lst, 'R_OBS':R_obs_lst, 'R_INDEL':R_ind_lst,
                     'MU':mu_lst, 'SIGMA':sigma_lst, 'ALPHA':alpha_lst, 'THETA':theta_lst,
                     'MU_INDEL': mu_ind_lst, 'SIGMA_INDEL':sigma_ind_lst, 'ALPHA_INDEL':alpha_ind_lst, 'THETA_INDEL':theta_ind_lst,
                     'Pi_SUM':p_mut_lst , 'Pi_INDEL':p_ind_lst
        }, index = elt_lst)

    df_model = df_mut_tab.merge(pretrain_df, left_on ='ELT', right_index=True)
    df_model = transfer_tools.element_expected_muts_nb(df_model)

    df_model = transfer_tools.element_expected_muts_nb(df_model)

    if not skip_pvals:
        df_model = transfer_tools.element_pvalue_burden_nb(df_model)
        df_model = transfer_tools.element_pvalue_burden_nb_by_sample(df_model)
        df_model = transfer_tools.element_pvalue_indel(df_model, cj_indel)
        df_model['PVAL_MUT_BURDEN'] = [
                    scipy.stats.combine_pvalues([row.PVAL_SNV_BURDEN, row.PVAL_INDEL_BURDEN],
                        method='fisher'
                    )[1]
                    for i, row in df_model.iterrows()
                ]
    if region_str:
        os.remove(temp_name)
    return df_model
