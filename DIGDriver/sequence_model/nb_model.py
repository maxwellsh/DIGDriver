import pandas as pd
import numpy as np
import scipy.stats
import scipy.special
import pysam
import h5py
import multiprocessing as mp
import statsmodels.stats.multitest

from DIGDriver.sequence_model import sequence_tools
from DIGDriver.sequence_model import gp_tools

def tabix_to_dataframe(tbx, chrom, start, end):
    """ Fetch a regions from a tabix file of mutations
        and return as dataframe
    """
    res = [row.split("\t") for row in tbx.fetch(chrom, start, end)]

    if not res:
        cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'ID']
    elif len(res[0]) == 6:
        cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'ID']
    elif len(res[0]) == 7:
        cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'ID', 'ANNOT']
    elif len(res[0]) == 8:
        cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'ID', 'MUT', 'CONTEXT']
    else:
        cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'ID', 'ANNOT', 'MUT', 'CONTEXT']

    df = pd.DataFrame(res, columns=cols)
    df = df.astype(dict(START=int, END=int))

    return df

def mutation_freq_conditional(S_mut, S_gen, N):
    """ Calculate the conditonal probability of a mutation given it's trinucleotide context
        (Normalizes mutation frequencies in a region by trinucloetide frequencies
        in that region)

        Each trinucloetide sequence is treated as a four-sided die: one side for each possible mutation
        and one side for no mutation. We want to estimate the probability of rolling each side given
        the trinucleotide context.

        We observe N * #{trinuc in genome} rolls of each trinucleotide die.
        We observe #{b | trinuc} stored in S_mut
        Pr(b | trinuc) = #{b | trinuc} / N * #{trinuc}
    """
    # K = S_mut.sum()  ## Total number of mutations
    S_mut_norm = S_mut.copy().astype(float)
    for tup in S_mut.index:
        S_mut_norm[tup] = S_mut[tup] / (N * S_gen[tup[1]])

    return S_mut_norm

def mutation_freq_joint(S_mut, S_gen, N):
    """ Calculate the joint probability of observing a mutation in a particular trinucleotide context
        Pr(b, tinuc) = Pr(b | trinuc) * Pr(trinuc)

        Empirically: Pr(b, trinuc) = #{b | trinuc} / N * #{all trinucs}

        Each trinucloetide sequence is treated as a four-sided die: one side for each possible mutation
        and one side for no mutation. We want to estimate the probability of rolling each side given
        the trinucleotide context.

        We observe N * #{trinuc in genome} rolls of each trinucleotide die.
        We observe #{b | trinuc} stored in S_mut
        Pr(b | trinuc) = #{b | trinuc} / N * #{trinuc}
    """
    # K = S_mut.sum()  ## Total number of mutations
    S_mut_norm = S_mut.copy().astype(float)
    for tup in S_mut.index:
        S_mut_norm[tup] = S_mut[tup] / (N * S_gen[tup[1]])

    return S_mut_norm

def train_sequence_model(train_idx, f_model, N, key_prefix=None):
    """ Train a trinucleotide sequence model based on precalculated mutational frequencies
        and trinucleotide occurences across the genome
    """

    train_idx_str = ['chr{}:{}-{}'.format(row[0], row[1], row[2]) for row in train_idx]

    key_mut = 'mutation_counts'
    if key_prefix:
        key_mut = key_prefix + "_" + key_mut

    df_mut = pd.read_hdf(f_model, key=key_mut)
    df_gen = pd.read_hdf(f_model, key='genome_counts')

    S_mut_train = df_mut.loc[train_idx_str, :].sum(axis=0) ## mutation context counts in train set
    S_gen_train = df_gen.loc[train_idx_str, :].sum(axis=0) ## trinucloetide counts in train set

    ## Probabilities stratified by mutation type
    Pr_mut_train = mutation_freq_conditional(S_mut_train, S_gen_train, N)

    ## Probabilities by trinucleotide context
    keys = set([tup[1] for tup in Pr_mut_train.index])
    d = {key: 0 for key in keys}
    for key in d:
        d[key] = sum([Pr_mut_train[tup] for tup in Pr_mut_train.index if tup[1]==key])

    S_pr = pd.Series(d)

    # return Pr_mut_train, S_pr
    return Pr_mut_train, d

def expected_mutations_by_context(train_idx, test_idx, f_model, N=1, key_prefix=None):
    """ Calculate the expected number of mutations in a train-test split
        based only on nucleotide sequence context
    """
    _, d_mut = train_sequence_model(train_idx, f_model, N, key_prefix=key_prefix)
    s_mut = pd.Series(d_mut)

    df_gen = pd.read_hdf(f_model, key='genome_counts')
    df_exp = (df_gen * s_mut).sum(axis=1)

    train_idx_str = ['chr{}:{}-{}'.format(row[0], row[1], row[2]) for row in train_idx]
    test_idx_str = ['chr{}:{}-{}'.format(row[0], row[1], row[2]) for row in test_idx]

    exp_train = df_exp.loc[train_idx_str]
    exp_test = df_exp.loc[test_idx_str]

    return exp_train, exp_test


def apply_nb_to_region(CHROM, START, END, mu, sigma, S_probs, tabix, fasta, n_up=2, n_down=2, binsize=1, collapse=False):

    chrom = "chr{}".format(CHROM)
    probs, pos_lst = sequence_tools.base_probabilities_by_region(fasta, S_probs, chrom, START, END,
                                                                 n_up=n_up, n_down=n_down, normed=True,
                                                                 collapse=collapse
                                                                )
    # probs, pos_lst, trinucs = sequence_tools.base_probabilities_by_region(fasta, S_probs, chrom, START, END)
    # print(probs)

    df = tabix_to_dataframe(tabix, str(CHROM), START, END)
    mut_counts = df.START.value_counts()

    alpha, theta = normal_params_to_gamma(mu, sigma)
    # expR = alpha * theta

    pvals = []
    poss = []
    obss = []
    exps = []
    pt_lst = []

    if binsize == 1:
        for pos, pt in zip(pos_lst, probs):
            k = 0
            if pos in mut_counts.index:
                k = mut_counts[pos]

            p = 1 / (pt * theta + 1)
            pvals.append(nb_pvalue_exact(k, alpha, p))
            # pvals.append(nb_pvalue_approx(k, alpha, p))
            poss.append(pos)
            obss.append(k)
            exps.append(pt * mu)
            pt_lst.append(pt)

    else:
        for i in range(0, len(pos_lst), binsize):
            pt = np.sum(probs[i:i+binsize])

            k = 0
            for pos in pos_lst[i:i+binsize]:
                if pos in mut_counts.index:
                    k += mut_counts[pos]

            p = 1 / (pt * theta + 1)
            pvals.append(nb_pvalue_exact(k, alpha, p))
            # pvals.append(nb_pvalue_approx(k, alpha, p))
            pos = np.mean(pos_lst[i:i+binsize])
            poss.append(pos)
            obss.append(k)
            exps.append(pt * mu)
            pt_lst.append(pt)

    pvals = np.array(pvals, dtype=float)
    poss = np.array(poss)
    obss = np.array(obss)
    exps = np.array(exps)
    # chroms = np.array([CHROM] * len(poss))

    return pvals, poss, obss, exps, pt_lst

def nb_model(d_pr, idx, mu_lst, sigma_lst, f_tabix, f_fasta, n_up=2, n_down=2, binsize=50, collapse=False):
    tabix = pysam.TabixFile(f_tabix)
    fasta = pysam.FastaFile(f_fasta)

    # alpha_lst, theta_lst = normal_params_to_gamma(mu_lst, sigma_lst)

    pvals_lst = []
    poss_lst = []
    obss_lst = []
    exps_lst = []
    chrom_lst = []
    reg_lst = []
    mus_lst = []
    sigmas_lst = []
    pts_lst = []

    for row, mu, sigma in zip(idx, mu_lst, sigma_lst):
        pvals, poss, obss, exps, pts = apply_nb_to_region(row[0], row[1], row[2],
                                                     mu, sigma, d_pr, tabix, fasta,
                                                     n_up=n_up, n_down=n_down,
                                                     binsize=binsize, collapse=collapse
                                  )
        pvals_lst.append(pvals)
        poss_lst.append(poss)
        obss_lst.append(obss)
        exps_lst.append(exps)
        pts_lst.append(pts)
        chrom_lst.append(np.array([row[0]]*len(poss)))
        reg_lst.append(np.array(["{}:{}-{}".format(row[0], row[1], row[2])]*len(poss)))
        mus_lst.append(np.array([mu]*len(poss)))
        sigmas_lst.append(np.array([sigma]*len(poss)))

    all_pvals = np.array([pval for pvals in pvals_lst for pval in pvals]).reshape(-1, 1)
    all_pos = np.array([pos for poss in poss_lst for pos in poss]).reshape(-1, 1)
    all_obs = np.array([obs for obss in obss_lst for obs in obss]).reshape(-1, 1)
    all_exp = np.array([exp for exps in exps_lst for exp in exps]).reshape(-1, 1)
    all_pt = np.array([pt for pts in pts_lst for pt in pts]).reshape(-1, 1)
    all_chroms = np.array([chrom for chroms in chrom_lst for chrom in chroms]).reshape(-1, 1)
    all_regs = np.array([reg for regs in reg_lst for reg in regs]).reshape(-1, 1)
    all_mu = np.array([mu for mus in mus_lst for mu in mus]).reshape(-1, 1)
    all_std = np.array([std for stds in sigmas_lst for std in stds]).reshape(-1, 1)

    nd = np.hstack([all_chroms, all_pos, all_obs, all_exp, all_pvals, all_pt, all_mu, all_std])
    df = pd.DataFrame(nd, columns=['CHROM', 'POS', 'OBS', 'EXP', 'PVAL', 'Pi', 'MU', 'SIGMA'])
    df['REGION'] = all_regs

    return df
    # return pvals_lst, poss_lst, obss_lst, exps_lst, chrom_lst, reg_lst, mus_lst, sigmas_lst

def normal_params_to_gamma(mu, sigma):
    alpha = mu**2 / sigma**2
    theta = sigma**2 / mu

    return alpha, theta

def nb_pvalue_greater(k, alpha, p):
    """ Calculate an UPPER TAIL p-value for a negative binomial distribution
    """
    if k == 0:
        pval = 1.

    else:
        pval = scipy.special.betainc(k, alpha, 1-p) # + \

        # Approximate p-value if betainc returns zero
        if pval == 0:
            pval = scipy.stats.nbinom.pmf(k, alpha, p)

    return pval

def nb_pvalue_greater_midp_DEPRECATED(k, alpha, p):
    """ Calculate an UPPER TAIL p-value for a negative binomial distribution
        with a midp correction
    """
    if k == 0:
        pval = 1 - 0.5 * scipy.stats.nbinom.pmf(k, alpha, p)

    else:
        pval = 0.5 * scipy.stats.nbinom.pmf(k, alpha, p) + \
               scipy.special.betainc(k+1, alpha, 1-p)

    return pval

def nb_pvalue_greater_midp(k, alpha, p):
    """ Calculate an UPPER TAIL p-value for a negative binomial distribution
        with a midp correction
    """
    pval = 0.5 * scipy.stats.nbinom.pmf(k, alpha, p) + \
           scipy.special.betainc(k+1, alpha, 1-p)

    return pval

def nb_pvalue_less(k, alpha, p):
    """ Calculate a LOWER TAIL p-value for a negative binomial distribution
    """
    pval = scipy.special.betainc(alpha, k+1, p)

def nb_pvalue_less_midp(k, alpha, p):
    """ Calculate a LOWER TAIL p-value for a negative binomial distribution
        with a midp correction
    """
    if k == 0:
        pval = 0.5 * scipy.stats.nbinom.pmf(k, alpha, p)

    else:
        pval = 0.5 * scipy.stats.nbinom.pmf(k, alpha, p) + \
               scipy.special.betainc(alpha, k, p)

    return pval

def nb_pvalue_exact(k, alpha, p, mu=None):
    """ Calculate an UPPER TAIL or LOWER TAIL p-value for a negative binomial distribution
        conditional on whether k is greater or less than the expectation
    """
    if not mu:
        mu = alpha * (1-p) / p

    if k < mu:
        pval = scipy.special.betainc(alpha, k+1, p)

    else:
        pval = scipy.special.betainc(k, alpha, 1-p)

        if pval == 0:
            pval = scipy.stats.nbinom.pmf(k, alpha, p)

    return pval

def nb_pvalue_midp(k, alpha, p, mu=None):
    """ Calculate an UPPER TAIL or LOWER TAIL p-value for a negative binomial distribution
        conditional on whether k is greater or less than the expectation
        with a midp correction
    """
    if not mu:
        mu = alpha * (1-p) / p

    if k < mu:
        if k > 0:
            pval = 0.5 * scipy.stats.nbinom.pmf(k, alpha, p) + \
                   scipy.special.betainc(alpha, k, p)
        else:
            pval = 0.5 * scipy.stats.nbinom.pmf(k, alpha, p)
    else:
        pval = 0.5 * scipy.stats.nbinom.pmf(k, alpha, p) + \
               scipy.special.betainc(k+1, alpha, 1-p)

        # if pval == 0:
        #     pval = scipy.stats.nbinom.pmf(k, alpha, p)[0]

    return pval

#using statsmodels for now
def get_q_vals(pvals_lst):
    _, q_vals = statsmodels.stats.multitest.fdrcorrection(pvals_lst, alpha=0.05, method='indep', is_sorted=False)
    return q_vals
