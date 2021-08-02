import pandas as pd
import numpy as np
import scipy.stats
import scipy.special
import pysam
import h5py
import multiprocessing as mp
import itertools as it
import math
import os

from DIGDriver.sequence_model import sequence_tools
from DIGDriver.sequence_model import nb_model

DNA53 = 'NTCGA'
DNA35 = 'NAGCT'
trans = DNA53.maketrans(DNA53, DNA35)

DNA = 'ACGT'
NUC = 'CT'

prod_items = [DNA] + [NUC] + [DNA]
keys = [''.join(tup) for tup in it.product(*prod_items)]

def reverse_complement(seq):
    return seq[::-1].translate(trans)


##called by genic_model_parallel
# def genic_model(df_obs, genes_lst, f_pretrained_str, f_genic_str, f_genome_counts, mapp):
def genic_model(genes_lst, f_pretrained_str, f_genic_str, counts_key, indels_direct):
    df_out = pd.DataFrame()
    f_genic = h5py.File(f_genic_str, 'r')
    subst_idx = f_genic['substitution_idx'][:].astype(str)
    cds_counts = pd.read_hdf(f_genic_str, counts_key)

    all_windows_df = pd.read_hdf(f_pretrained_str, 'region_params')
    window = all_windows_df.iloc[0][2]-all_windows_df.iloc[0][1]
    df_mut = pd.read_hdf(f_pretrained_str, key='sequence_model_192')
    mut_probs_idx = [r[1] + '>' + r[1][0] + r[0][2] + r[1][2] for r in zip(df_mut.MUT_TYPE, df_mut.CONTEXT)]
    d_pr = pd.DataFrame(df_mut.FREQ.values, mut_probs_idx)

    if indels_direct:
        all_indels_df = pd.read_hdf(f_pretrained_str, 'region_params_indels')

    chrom_lst = []
    genes_used = []

    # mis_obs_lst = []
    # nons_obs_lst = []
    # stop_loss_obs_lst = []
    # silent_obs_lst = []
    # splice_obs_lst = []
    # trunc_obs_lst = []

    p_silent_lst = []
    p_mis_lst = []
    p_nons_lst = []
    p_splice_lst = []
    p_trunc_lst = []
    p_ind_lst = []

    mu_lst = []
    s_lst = []

    mu_ind_lst = []
    s_ind_lst = []

    # pvals_mis = []
    # pvals_nons = []
    # pvals_silent = []
    # pvals_splice = []
    # pvals_trunc = []

    R_size_lst = []
    R_obs_lst = []
    R_ind_lst = []
    flag_lst = []
    gene_len_lst = []
    # exp_mis_lst = []
    # exp_nons_lst = []
    # exp_silent_lst = []
    # exp_splice_lst = []
    # exp_trunc_lst = []

    for gene in genes_lst:

        chrom = f_genic['chr'][gene][:][0].decode("utf-8")
        if chrom == 'X' or chrom =='Y':
            continue

        # obs_counts = df_obs.loc[gene]
        # obs_mis = obs_counts['Missense']
        # obs_nons = obs_counts['Nonsense']
        # obs_stop_loss = obs_counts['Stop_loss']
        # obs_silent = obs_counts['Synonymous']
        # obs_splice = obs_counts['Essential_Splice']
        # obs_trunc = obs_nons + obs_splice

        # mis_obs_lst.append(obs_mis)
        # nons_obs_lst.append(obs_nons)
        # stop_loss_obs_lst.append(obs_stop_loss)
        # silent_obs_lst.append(obs_silent)
        # splice_obs_lst.append(obs_splice)
        # trunc_obs_lst.append(obs_trunc)

        genes_used.append(gene)
        chrom_lst.append(chrom)
        intervals = f_genic['cds_intervals'][gene][:]
        L = pd.DataFrame(f_genic['L_data'][gene][:].T, index = subst_idx, columns=[0,1,2,3])
        context_counts = pd.DataFrame(cds_counts.loc[gene].T.values, index=cds_counts.columns)

        #replacing genic_seq_model with precounted cds
        prob_sum = context_counts * d_pr
        t_pi = d_pr / prob_sum[0].sum()
        t_pi = pd.concat([t_pi] * (4), axis=1, ignore_index=True)

        pi_sums = t_pi * L
        p_silent = pi_sums[0].sum()
        p_mis = pi_sums[1].sum()
        p_nons = pi_sums[2].sum()
        p_splice = pi_sums[3].sum()
        p_trunc = p_nons + p_splice

        p_mis_lst.append(p_mis)
        p_nons_lst.append(p_nons)
        p_silent_lst.append(p_silent)
        p_splice_lst.append(p_splice)
        p_trunc_lst.append(p_trunc)

        mu,sigma,R_obs,flag = get_region_params(all_windows_df, chrom, intervals,  window)
        mu_lst.append(mu)
        s_lst.append(sigma)
        flag_lst.append(flag)

        # pval_mis = calc_pvalue(mu, sigma, p_mis, obs_mis)
        # pvals_mis.append(pval_mis)
        # pval_nons = calc_pvalue(mu, sigma, p_nons, obs_nons)
        # pvals_nons.append(pval_nons)
        # pval_silent = calc_pvalue(mu, sigma, p_silent, obs_silent)
        # pvals_silent.append(pval_silent)
        # pval_splice = calc_pvalue(mu, sigma, p_splice, obs_splice)
        # pvals_splice.append(pval_splice)
        # pval_trunc = calc_pvalue(mu, sigma, p_trunc, obs_trunc)
        # pvals_trunc.append(pval_trunc)

        R_size_lst.append(int(cds_counts.loc[gene, :].sum() / 3))  ## length of region containing gene
        R_obs_lst.append(R_obs)
        # alpha, theta = nb_model.normal_params_to_gamma(mu,sigma)
        # exp_mis_lst.append(alpha*theta*p_mis)
        # exp_nons_lst.append(alpha*theta*p_nons)
        # exp_silent_lst.append(alpha*theta*p_silent)
        # exp_splice_lst.append(alpha*theta*p_splice)
        # exp_trunc_lst.append(alpha*theta*p_trunc)

        ## Deal with indel model
        # gene_len_lst.append(int(np.sum(L.values) / 3))
        gene_len_lst.append(np.sum(intervals[1, :] - intervals[0, :] + 1))
        p_ind_lst.append(gene_len_lst[-1] / R_size_lst[-1])

        if indels_direct:
            mu_ind,sigma_ind,R_ind, _ = get_region_params(all_indels_df, chrom, intervals, window)
        else:
            mu_ind, sigma_ind, R_ind = mu, sigma, R_obs

        mu_ind_lst.append(mu_ind)
        s_ind_lst.append(sigma_ind)
        R_ind_lst.append(R_ind)

    df_out['CHROM'] = chrom_lst
    df_out['GENE'] = genes_used
    # df_out['OBS_MIS'] = mis_obs_lst
    # df_out['OBS_NONS'] = nons_obs_lst
    # df_out['OBS_SILENT']= silent_obs_lst
    # df_out['OBS_SPLICE']= splice_obs_lst
    # df_out['OBS_TRUNC']= trunc_obs_lst
    # df_out['EXP_MIS'] = exp_mis_lst
    # df_out['EXP_NONS'] = exp_nons_lst
    # df_out['EXP_SILENT'] = exp_silent_lst
    # df_out['EXP_SPLICE']= exp_splice_lst
    # df_out['EXP_TRUNC']= exp_trunc_lst
    df_out['GENE_LENGTH'] = gene_len_lst
    df_out['R_SIZE'] = R_size_lst
    df_out['R_OBS'] = R_obs_lst
    df_out['R_INDEL'] = R_ind_lst
    df_out['MU'] = mu_lst
    df_out['SIGMA'] = s_lst
    df_out['MU_INDEL'] = mu_ind_lst
    df_out['SIGMA_INDEL'] = s_ind_lst
    df_out['FLAG'] = flag_lst
    # df_out['PVAL_MIS'] = pvals_mis
    # df_out['PVAL_NONS'] = pvals_nons
    # df_out['PVAL_SILENT'] = pvals_silent
    # df_out['PVAL_SPLICE'] = pvals_splice
    # df_out['PVAL_TRUNC'] = pvals_trunc
    df_out['P_MIS'] = p_mis_lst
    df_out['P_NONS'] = p_nons_lst
    df_out['P_SILENT'] = p_silent_lst
    df_out['P_SPLICE'] = p_splice_lst
    df_out['P_TRUNC'] = p_trunc_lst
    df_out['P_INDEL'] = p_ind_lst
    f_genic.close()
    return df_out

# def genic_model_parallel(mut_obs_df, f_pretrained_str, f_genic_str, f_genome_counts, mapp, N_procs):
def genic_model_parallel(f_pretrained_str, f_genic_str, N_procs, counts_key="window_10kb/counts", indels_direct=False):
    ## Parallel chunk parameters:
    f_genic = h5py.File(f_genic_str, 'r')
    all_genes = list(f_genic['cds_intervals'].keys())
    f_genic.close()
    chunksize = int(np.ceil(len(all_genes) / N_procs))
    res = []
    pool = mp.Pool(N_procs)
    for i in np.arange(0, len(all_genes), chunksize):
        gene_chunk = all_genes[i:i+chunksize]

        r = pool.apply_async(genic_model, (gene_chunk, f_pretrained_str, f_genic_str, counts_key, indels_direct))
        # r = pool.apply_async(genic_model, (mut_obs_df, gene_chunk, f_pretrained_str, f_genic_str, f_genome_counts, mapp))
        res.append(r)

    pool.close()
    pool.join()

    res_lst = [r.get() for r in res]
    complete = pd.concat(res_lst)
    return complete


# finds estimated region parameters (mu, sigma) for a given gene
# inputs : df - gp results df with chr locs and region parameters
#          chrom - gene chrom
#          intervals - 2d numpy array of start, end positions of cds regions
# output : average mu and sigma values for the non-duplicated overlapping regions
#          or returns -1,-1 if no regions are overlapped
def get_region_params(df, chrom, intervals, window):
    ideal = get_ideal_overlaps(chrom, intervals, window)
    mu_sum = 0
    var_sum = 0
    R_obs_sum = 0
    FLAG = False
    ideal = [trip_to_str(r) for r in ideal]
    for r in ideal:
        mu_sum += df.loc[r, 'Y_PRED']
        var_sum += df.loc[r, 'STD']**2
        R_obs_sum += df.loc[r, 'Y_TRUE']
        FLAG += df.loc[r, 'FLAG']

    mu = mu_sum
    sigma = np.sqrt(var_sum)
    return mu, sigma, R_obs_sum, FLAG

# finds estimated region parameters (mu, sigma) for a given gene
#using overlaps directly
# inputs : df - gp results df with chr locs and region parameters
#
# output : average mu and sigma values for the non-duplicated overlapping regions
#          or returns -1,-1 if no regions are overlapped
def get_region_params_direct(df, overlaps, window):
    mu_sum = 0
    var_sum = 0
    R_obs_sum = 0
    FLAG = False
    ideal = [trip_to_str(r) for r in overlaps]
    for r in ideal:
        mu_sum += df.loc[r, 'Y_PRED']
        var_sum += df.loc[r, 'STD']**2
        R_obs_sum += df.loc[r, 'Y_TRUE']
        FLAG += df.loc[r, 'FLAG']

    mu = mu_sum
    sigma = np.sqrt(var_sum)
    return mu, sigma, R_obs_sum, FLAG


def get_ideal_overlaps(chrom, intervals, window):
    region_lst = []
    for i in intervals.T:
        low = math.floor(i[0].min() / window) * window
        high = math.ceil(i[1].max() / window) * window
        borders = np.arange(low, high +window, window)
        for i in range(len(borders)-1):
            region_lst.append((chrom,int(borders[i]), int(borders[i+1])))
    return list(set(region_lst))


def trip_to_str(trip):
    return 'chr{}:{}-{}'.format(trip[0], trip[1], trip[2])

def get_elt_ideal_overlaps(chrom, start, end, window):
    region_lst = []
    low = math.floor(start / window) * window
    high = math.ceil(end / window) * window
    borders = np.arange(low, high + window, window)
    for i in range(len(borders)-1):
        region_lst.append((int(chrom),borders[i], borders[i+1]))
    return list(set(region_lst))


# def nonc_model(df_nonc_obs, f_pretrained, f_nonc_data, save_key, f_sites = False):
def nonc_model(elt_lst, f_pretrained, f_nonc_data, save_key, indels_direct):
    # if f_sites:
    #     df_nonc_obs = df_nonc_obs.copy().astype({'ELT':str, 'OBS_SAMPLES':int, 'OBS_MUT':int})
    # else:
    #     df_nonc_obs = df_nonc_obs.copy().astype({'CHROM':int, 'ELT':str, 'STRAND':str,
    #     'BLOCK_STARTS':object, 'BLOCK_ENDS':object, 'OBS_SAMPLES':int,'OBS_MUT':int})

    all_windows_df = pd.read_hdf(f_pretrained, 'region_params')
    window = all_windows_df.iloc[0][2]-all_windows_df.iloc[0][1]
    window_key = 'window_{}'.format(window)
    # if window < 1000:
    #     print("Warning: Model is not intended for use with windows < 1kb")
    #     window_key = '{}bp'.format(window)
    # else:
    #     window_key = '{}kb'.format(int(window/1000))

    if indels_direct:
        all_indels_df = pd.read_hdf(f_pretrained, 'region_params_indels')

    nonc_data = h5py.File(f_nonc_data, 'r')

    df_mut = pd.read_hdf(f_pretrained, key='sequence_model_192')
    mut_model_idx = [r[1] + '>' + r[1][0] + r[0][2] + r[1][2] for r in zip(df_mut.MUT_TYPE, df_mut.CONTEXT)]

    d_pr = pd.DataFrame(df_mut.FREQ.values, mut_model_idx)
    d_pr = d_pr.sort_index()[0].values

    p_mut_lst = []
    p_ind_lst = []

    mu_lst = []
    s_lst = []

    mu_ind_lst = []
    s_ind_lst = []

    # pvals_lst = []
    R_obs_lst = []
    R_size_lst = []
    R_ind_lst = []
    flag_lst = []
    elt_len_lst = []
    # exp_lst = []
    # exp_samples_lst = []
    # pval_samples_lst = []

    # for _, row in df_nonc_obs.iterrows():
    for elt in elt_lst:
        # obs_samp = row.OBS_SAMPLES
        # obs_mut = row.OBS_MUT
        # elt = row.ELT
        # if f_sites:
        #     region_counts = nonc_data['{}/sites_data/{}/{}/region_counts'.format(window_key, save_key, elt)][:]
        #     L = nonc_data['{}/sites_data/{}/{}/L_counts'.format(window_key, save_key, elt)][:]
        #     overlaps = nonc_data['{}/sites_data/{}/{}'.format(window_key, save_key, elt)].attrs['overlaps']

        # else:
        region_counts = nonc_data['{}/{}/{}/region_counts'.format(window_key, save_key, elt)][:]
        L = nonc_data['{}/{}/{}/L_counts'.format(window_key, save_key, elt)][:]
        overlaps = nonc_data['{}/{}/{}'.format(window_key, save_key, elt)].attrs['overlaps']

        prob_sum = region_counts * d_pr
        if prob_sum.sum() == 0:
            print(elt, overlaps)
        t_pi = d_pr / prob_sum.sum()

        p_mut = (t_pi * L).sum()
        p_mut_lst.append(p_mut)

        mu, sigma, R_obs, FLAG = get_region_params_direct(all_windows_df, overlaps, window)
        mu_lst.append(mu)
        s_lst.append(sigma)
        flag_lst.append(FLAG)
        # pval_mut = calc_pvalue(mu, sigma, p_mut, obs_mut)
        # pvals_lst.append(pval_mut)
        R_size_lst.append(int(region_counts.sum() / 3))  ## length of region containing gene
        R_obs_lst.append(R_obs)
        # alpha, theta = nb_model.normal_params_to_gamma(mu,sigma)
        # exp_lst.append(alpha*theta*p_mut)

        elt_len_lst.append(int(np.sum(L) / 3))
        p_ind_lst.append(elt_len_lst[-1] / R_size_lst[-1])

        if indels_direct:
            mu_ind,sigma_ind,R_ind, _ = get_region_params_direct(all_indels_df, overlaps, window)
        else:
            mu_ind, sigma_ind, R_ind = mu, sigma, R_obs

        mu_ind_lst.append(mu_ind)
        s_ind_lst.append(sigma_ind)
        R_ind_lst.append(R_ind)

        # if obs_mut == 0:
        #     samples_scaling = 1
        # else:
        #     samples_scaling = obs_samp / obs_mut

        # pval_sample = calc_pvalue(mu, sigma, samples_scaling * p_mut, obs_samp)
        # pval_samples_lst.append(pval_sample)
        # exp_samples_lst.append(alpha*theta*p_mut*samples_scaling)

    nonc_data.close()
    # q_lst = nb_model.get_q_vals(np.array(pvals_lst))

    df_nonc = pd.DataFrame({
        'ELT': elt_lst,
        'ELT_SIZE': elt_len_lst,
        'FLAG': flag_lst,
        'R_SIZE': R_size_lst,
        'R_OBS': R_obs_lst,
        'R_INDEL': R_ind_lst,
        'MU': mu_lst,
        'SIGMA': s_lst,
        'MU_INDEL': mu_ind_lst,
        'SIGMA_INDEL': s_ind_lst,
        'P_SUM': p_mut_lst,
        'P_INDEL': p_ind_lst
    })
    # df_nonc_obs['EXP_MUTS'] = exp_lst
    # df_nonc_obs['EXP_SAMPLES'] = exp_samples_lst
    # df_nonc_obs['R_OBS'] = R_obs_lst
    # df_nonc_obs['MU'] = mu_lst
    # df_nonc_obs['SIGMA'] = s_lst
    # df_nonc_obs['PVAL'] = pvals_lst
    # df_nonc_obs['PVAL_SAMPLES'] = pval_samples_lst
    # df_nonc_obs['P_SUM'] = p_mut_lst #sum of pi values
    # df_nonc_obs['QVAL'] = q_lst #fdr corrected q value
    # if not f_sites:
    #     df_nonc_obs = df_nonc_obs.drop(['BLOCK_STARTS', 'BLOCK_ENDS'], axis=1)
    # return df_nonc_obs

    return df_nonc

# def nonc_model_parallel(df_nonc_obs, f_pretrained, f_nonc_data, nonc_L_key, N_procs, f_sites = False):
def nonc_model_parallel(f_pretrained, f_nonc_data, nonc_L_key, N_procs, indels_direct=False):

    with h5py.File(f_pretrained, 'r') as h5_pre:
        window = h5_pre['idx'][0, 2] - h5_pre['idx'][0, 1]
        window_key = 'window_{}'.format(window)

    with h5py.File(f_nonc_data, 'r') as h5_nc:
        elt_lst = list(h5_nc['{}/{}'.format(window_key, nonc_L_key)].keys())

    ## Parallel chunk parameters:
    chunksize = int(np.ceil(len(elt_lst) / N_procs))
    # chunksize = int(np.ceil(len(df_nonc_obs) / N_procs))
    res = []
    pool = mp.Pool(N_procs)
    for i in np.arange(0, len(elt_lst), chunksize):
        # df = df_nonc_obs.iloc[i:i+chunksize]
        elt_chunk = elt_lst[i:i+chunksize]

        r = pool.apply_async(nonc_model,(elt_chunk, f_pretrained, f_nonc_data, nonc_L_key, indels_direct))
        # r = pool.apply_async(nonc_model,(df, f_pretrained, f_nonc_data, nonc_L_key, f_sites))
        res.append(r)

    pool.close()
    pool.join()

    res_lst = [r.get() for r in res]
    complete = pd.concat(res_lst)
    return complete

def nonc_model_region_parallel(f_bed, f_pretrained, f_nonc_data, nonc_L_key, N_procs, f_sites = None):
    """ Pretrain a noncoding model based on regions defined in a bed file

    Context counts for each (sub)element should have been precounted and
    stored in f_nonc_data under key nonc_L_key
    """
    ### Read in regions from bed file
    print('Parsing regions bed file')
    df_nonc = pd.read_table(f_bed, names=['CHROM', 'START', 'END', "ELT", "SCORE", "STRAND", 'thickStart', 'thickEnd', 'rgb', 'blockCount', 'blockSizes', 'blockStarts'], low_memory=False)
    df_nonc.CHROM = df_nonc.CHROM.astype(str)
    df_nonc = df_nonc[df_nonc.CHROM.isin([str(c) for c in range(1, 23)])]
    df_nonc.CHROM = df_nonc.CHROM.astype(int)

    def _get_starts(row):
        str_starts = row.blockStarts
        if str_starts.endswith(','):
            str_starts = str_starts[:-1]

        return [int(x)+row.START for x in str_starts.split(",")]

    def _get_ends(row):
        str_sizes = row.blockSizes
        if str_sizes.endswith(','):
            str_sizes = str_sizes[:-1]

        sizes = [int(x) for x in str_sizes.split(',')]

        return [START + SIZE for START, SIZE in zip(row.BLOCK_STARTS, sizes)]

    df_nonc['BLOCK_STARTS'] = df_nonc.apply(_get_starts, axis=1)
    df_nonc['BLOCK_ENDS'] = df_nonc.apply(_get_ends, axis=1)

    df_nonc = df_nonc[['CHROM', 'ELT', 'STRAND', 'BLOCK_STARTS', 'BLOCK_ENDS']]

    # print(df_nonc[df_nonc.ELT == 'TMEM240'])
    # return

    ## Parallel chunk parameters:
    print('Pretraining model')
    chunksize = int(np.ceil(len(df_nonc) / N_procs))
    res = []
    pool = mp.Pool(N_procs)
    for i in np.arange(0, len(df_nonc), chunksize):
        df = df_nonc.iloc[i:i+chunksize]

        r = pool.apply_async(nonc_model_region,(df, f_pretrained, f_nonc_data, nonc_L_key, f_sites))
        res.append(r)

    pool.close()
    pool.join()

    res_lst = [r.get() for r in res]
    complete = pd.concat(res_lst)
    return complete

def nonc_model_region(df_nonc, f_pretrained, f_nonc_data, nonc_L_key, f_sites = None, return_intermediates=False):
    df_nonc = df_nonc.copy().astype({'CHROM':int, 'ELT':str, 'STRAND':str, 'BLOCK_STARTS':object, 'BLOCK_ENDS':object,})
    L_counts = pd.read_hdf(f_nonc_data, nonc_L_key)

    all_windows_df = pd.read_hdf(f_pretrained, 'region_params')
    window = all_windows_df.iloc[0][2]-all_windows_df.iloc[0][1]

    nonc_data = h5py.File(f_nonc_data, 'r')
    idx = nonc_data['full_window_si_index'][:]
    idx_dict = dict(zip(map(tuple, idx), range(len(idx))))

    df_mut = pd.read_hdf(f_pretrained, key='sequence_model_192')
    mut_model_idx = [r[1] + '>' + r[1][0] + r[0][2] + r[1][2] for r in zip(df_mut.MUT_TYPE, df_mut.CONTEXT)]
    subst_idx = sorted(mut_model_idx)
    revc_subst_idx = [sequence_tools.reverse_complement(sub.split('>')[0]) + '>' + sequence_tools.reverse_complement(sub.split('>')[-1]) for sub in subst_idx]
    revc_dic = dict(zip(subst_idx, revc_subst_idx))
    d_pr = pd.DataFrame(df_mut.FREQ.values, mut_model_idx)
    d_pr = d_pr.sort_index()[0].values

    keys = set(list(subst_idx))
    d = {key: 0 for key in sorted(keys)}

    p_mut_lst = []
    mu_lst = []
    s_lst = []
    pvals_lst = []
    R_obs_lst = []
    exp_lst = []
    exp_samples_lst = []
    pval_samples_lst = []
    L_lst = []
    t_pi_lst = []

    for _, row in df_nonc.iterrows():
        chrom = row.CHROM
        strand = row.STRAND

        block_starts = row[3]
        block_ends = row[4]
        elts_as_intervals = np.vstack((block_starts, block_ends))
        overlaps = get_ideal_overlaps(chrom, elts_as_intervals, window)
        region_counts = np.array([np.repeat(nonc_data['full_window_si_values'][idx_dict[region], :], 3) for region in overlaps]).sum(axis=0)

        # if negative strand, take the reverse complement of the region counts
        if strand == '-1' or strand == '-':
            region_counts = [r[1] for r in sorted(enumerate(region_counts), key=lambda k: revc_dic[subst_idx[k[0]]])]
        L = np.zeros((192))

        for start, end in zip(block_starts, block_ends):
            L += L_counts.loc['chr{}:{}-{}'.format(chrom, start,end)].values

        prob_sum = region_counts * d_pr
        t_pi = d_pr / prob_sum.sum()

        p_mut = (t_pi * L).sum()

        p_mut_lst.append(p_mut)
        mu,sigma,R_obs = get_region_params(all_windows_df, chrom, elts_as_intervals, window)

        mu_lst.append(mu)
        s_lst.append(sigma)
        R_obs_lst.append(R_obs)

        if return_intermediates:
            L_lst.append(L)
            t_pi_lst.append(t_pi)

    df_nonc['R_OBS'] = R_obs_lst
    df_nonc['MU'] = mu_lst
    df_nonc['SIGMA'] = s_lst
    df_nonc['P_SUM'] = p_mut_lst #sum of pi values

    if return_intermediates:
        idx = sorted(mut_model_idx)
        df_L = pd.DataFrame(L_lst, columns=idx, index=df_nonc.ELT)
        df_pi = pd.DataFrame(t_pi_lst, columns=idx, index=df_nonc.ELT)
        return df_nonc.drop(['BLOCK_STARTS', 'BLOCK_ENDS'], axis=1), df_L, df_pi
    else:
        return df_nonc.drop(['BLOCK_STARTS', 'BLOCK_ENDS'], axis=1)


def tiled_nonc_model(elt_lst, f_pretrained, f_nonc_data, save_key):

    all_windows_df = pd.read_hdf(f_pretrained, 'region_params')
    window = all_windows_df.iloc[0][2]-all_windows_df.iloc[0][1]
    window_key = 'window_{}'.format(window)

    nonc_data = h5py.File(f_nonc_data, 'r')

    L_table = pd.read_hdf(f_nonc_data, "{}/L_counts".format(save_key))

    idx = nonc_data['{}/full_window_si_index'.format(window_key)][:]
    idx_dict = dict(zip(map(tuple, idx), range(len(idx))))

    df_mut = pd.read_hdf(f_pretrained, key='sequence_model_192')
    mut_model_idx = [r[1] + '>' + r[1][0] + r[0][2] + r[1][2] for r in zip(df_mut.MUT_TYPE, df_mut.CONTEXT)]

    d_pr = pd.DataFrame(df_mut.FREQ.values, mut_model_idx)
    d_pr = d_pr.sort_index()[0].values

    p_mut_lst = []
    mu_lst = []
    s_lst = []
    R_obs_lst = []

    elt_len_lst = []
    flag_lst = []
    R_size_lst = []
    R_ind_lst = []
    mu_ind_lst = []
    s_ind_lst = []
    p_ind_lst = []

    for elt in elt_lst:

        pos = elt.split(":")[1]
        chrom = int(elt.split(":")[0].lstrip("chr"))
        start = int(pos.split("-")[0])
        region_start = int(np.floor(start / 10000) * 10000)

        L = L_table.loc[elt]
        #L = nonc_data['{}/{}/{}/L_counts'.format(window_key, save_key, elt)][:]
        #overlaps = nonc_data['{}/{}/{}'.format(window_key, save_key, elt)].attrs['overlaps']
        region = (chrom, region_start, region_start+window)
        region_counts = np.repeat(nonc_data['{}/full_window_si_values'.format(
            window_key)][idx_dict[region], :], 3)

        overlaps = [region]

        prob_sum = region_counts * d_pr
        t_pi = d_pr / prob_sum.sum()

        p_mut = (t_pi * L).sum()

        p_mut_lst.append(p_mut)
        mu,sigma,R_obs,FLAG = get_region_params_direct(all_windows_df, overlaps, window)

        flag_lst.append(FLAG)
        R_size_lst.append(int(region_counts.sum() / 3))
        elt_len_lst.append(int(np.sum(L) / 3))
        p_ind_lst.append(elt_len_lst[-1] / R_size_lst[-1])

        mu_ind, sigma_ind, R_ind = mu, sigma, R_obs

        mu_ind_lst.append(mu_ind)
        s_ind_lst.append(sigma_ind)
        R_ind_lst.append(R_ind)

        mu_lst.append(mu)
        s_lst.append(sigma)

        R_obs_lst.append(R_obs)

    nonc_data.close()

    #for compatibility with region files
    elt_lst = [_index_transform(i) for i in elt_lst]
    df_nonc = pd.DataFrame({
        'ELT': elt_lst,
        'ELT_SIZE': elt_len_lst,
        'FLAG': flag_lst,
        'R_SIZE': R_size_lst,
        'R_OBS': R_obs_lst,
        'R_INDEL': R_ind_lst,
        'MU': mu_lst,
        'SIGMA': s_lst,
        'MU_INDEL': mu_ind_lst,
        'SIGMA_INDEL': s_ind_lst,
        'P_SUM': p_mut_lst,
        'P_INDEL': p_ind_lst
    })

    return df_nonc

def tiled_model_parallel(f_pretrained, f_nonc_data, save_key, N_procs):

    with h5py.File(f_pretrained, 'r') as h5_pre:
        window = h5_pre['idx'][0, 2] - h5_pre['idx'][0, 1]
        window_key = 'window_{}'.format(window)

    elt_table = pd.read_hdf(f_nonc_data, "{}/L_counts".format(save_key))
    elt_lst = elt_table.index

    ## Parallel chunk parameters:
    chunksize = int(np.ceil(len(elt_lst) / N_procs))
    # chunksize = int(np.ceil(len(df_nonc_obs) / N_procs))
    res = []
    pool = mp.Pool(N_procs)
    for i in np.arange(0, len(elt_lst), chunksize):
        # df = df_nonc_obs.iloc[i:i+chunksize]
        elt_chunk = elt_lst[i:i+chunksize]

        r = pool.apply_async(tiled_nonc_model,(elt_chunk, f_pretrained, f_nonc_data, save_key))
        # r = pool.apply_async(nonc_model,(df, f_pretrained, f_nonc_data, nonc_L_key, f_sites))
        res.append(r)

    pool.close()
    pool.join()

    res_lst = [r.get() for r in res]
    complete = pd.concat(res_lst)
    return complete

def _index_transform(s):
    chrom = int(s.split(":")[0].lstrip("chr"))
    start = int(s.split(":")[-1].split('-')[0])
    end = int(s.split(":")[-1].split('-')[1])
    return "region_{}_{}_{}".format(chrom,start, end)
