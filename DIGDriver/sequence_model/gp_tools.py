##entire module seems to be deprecated. superceeeded by region_model_tools

import pandas as pd
import numpy as np
import scipy.stats
import h5py
import seaborn as sns
import matplotlib.pyplot as plt

##deprecated. gp loading now done in region_model_tools. may be useful in notebooks?
##TODELETE
def load_ensemble(f, cancer=None, split='test'):
    ## Load data
    data_pred = h5py.File(f, 'r')
    if cancer:
        dset = data_pred[cancer]
    else:
        dset = data_pred

    try:
        runs = [key for key in dset[split].keys() if key.isdigit()] ## NOTE: bad way to find integers used as keys
        train_idx = dset['train']['chr_locs'][:]
        y_true = dset[split]['y_true'][:].reshape(-1, 1)
        idx = dset[split]['chr_locs'][:]
        gp_mean_lst = [dset[split][str(i)]['mean'][:] for i in runs]
        gp_std_lst = [dset[split][str(i)]['std'][:] for i in runs]

    except:
        reruns = len([key for key in dset[split].keys() if key.startswith('gp_mean')])
        train_idx = dset['train']['idxs'][:]
        y_true = dset[split]['true'][0, :].reshape(-1, 1)
        idx = dset[split]['idxs'][:]
        gp_mean_lst = [dset[split]['gp_mean_{:02d}'.format(run)][:] for run in range(1, reruns-1)]
        gp_std_lst = [dset[split]['gp_std_{:02d}'.format(run)][:] for run in range(1, reruns-1)]

    gp_mean_nd = np.vstack(gp_mean_lst)
    gp_mean = np.median(gp_mean_nd, axis=0).reshape(-1, 1)

    gp_std_nd = np.vstack(gp_std_lst)
    gp_std = np.median(gp_std_nd, axis=0).reshape(-1, 1)

    data_pred.close()

    return train_idx, y_true, idx, gp_mean, gp_std

##deprecated. gp loading now done in region_model_tools. may be useful in notebooks?
##TODELETE
def load_run(f, run, cancer=None, split='test'):
    hf = h5py.File(f, 'r')
    if cancer:
        dset = hf[cancer]
    else:
        dset = hf

    try:
        train_idx = dset['train']['chr_locs'][:]
        test_Y = dset[split]['y_true'][:].reshape(-1, 1)
        test_idx = dset[split]['chr_locs'][:]
        test_Yhat = dset[split]['{}'.format(run)]['mean'][:].reshape(-1, 1)
        test_std = dset[split]['{}'.format(run)]['std'][:].reshape(-1, 1)
    except:
        train_idx = dset['train']['idxs'][:]
        test_Y = dset[split]['true'][0, :].reshape(-1, 1)
        test_idx = dset[split]['idxs'][:]
        test_Yhat = dset[split]['gp_mean_{:02d}'.format(run)][:].reshape(-1, 1)
        test_std = dset[split]['gp_std_{:02d}'.format(run)][:].reshape(-1, 1)

    hf.close()
    return train_idx, test_Y, test_idx, test_Yhat, test_std

##deprecated. gp loading now done in region_model_tools. may be useful in notebooks?
##TODELETE
def load_fold(f, cancer=None, run=None, split='test', reruns=10):
    if run == None:
        run = pick_gp_by_calibration(f, cancer=cancer, dataset=split)

    if run=='ensemble':
        train_idx, test_Y, test_idx, test_Yhat, test_std = load_ensemble(f, cancer=cancer, split=split)

    else:
        train_idx, test_Y, test_idx, test_Yhat, test_std = load_run(f, run, cancer=cancer, split=split)

    vals = np.hstack([test_idx, test_Y, test_Yhat, test_std])
    df = pd.DataFrame(vals, columns=['CHROM', 'START', 'END', 'Y_TRUE', 'Y_PRED', 'STD'])

    return df

def plot_qq_log(pvals, label='', ax=None, rasterized=False, color=None):
    if not ax:
        f, ax = plt.subplots(1, 1)
    exp = -np.log10(np.arange(1, len(pvals) + 1) / len(pvals))
    pvals_log10_sort = -np.log10(np.sort(pvals))

    if not color:
        color = sns.color_palette()[0]

    ax.plot(exp, pvals_log10_sort, '.', label=label, rasterized=rasterized, color=color)
    ax.plot(exp, exp, 'k-')
    # ax.plot(exp, exp, 'r-')

    if label:
        ax.legend()

def plot_qq(pvals, label='', ax=None, rasterized=False):
    if not ax:
        f, ax = plt.subplots(1, 1)
    exp  = (np.arange(1, len(pvals) + 1) / len(pvals))
    pvals_sort = np.sort(pvals)

    ax.plot(exp, pvals_sort, '.', label=label, rasterized=rasterized)
    ax.plot(exp, exp, 'r-')

    if label:
        ax.legend()


def calibration_score_by_pvals(pvals):
    alpha = [0.05, 0.01, 0.001, 0.0001]
    alpha_emp = [len(pvals[pvals < a]) / len(pvals) for a in alpha]

    return sum([(a-ap)**2 for a, ap in zip(alpha, alpha_emp)])


# def merge_windows(df, start, end, new_size):
def merge_windows(df, idx_new):
    # bins = np.concatenate([np.arange(start, end, new_size), [end]])

    Y_merge = np.array([df[(df.CHROM==row[0]) & (df.START >= row[1]) & (df.START < row[2])].Y_TRUE.sum() \
                           for row in idx_new])
    Yhat_merge = np.array([df[(df.CHROM==row[0]) & (df.START >= row[1]) & (df.START < row[2])].Y_PRED.sum() \
                           for row in idx_new])
    std_merge = np.array([np.sqrt((df[(df.CHROM==row[0]) & (df.START >= row[1]) & (df.START < row[2])].STD**2).sum()) \
                           for row in idx_new])

    # Y_merge = np.array([df[(df.START >= v1) & (df.START < v2)].Y_TRUE.sum() \
    #                          for v1, v2 in zip(bins[:-1], bins[1:])])
    # Yhat_merge = np.array([df[(df.START >= v1) & (df.START < v2)].Y_PRED.sum() \
    #                             for v1, v2 in zip(bins[:-1], bins[1:])])
    # std_merge = np.array([np.sqrt((df[(df.START >= v1) & (df.START < v2)].STD**2).sum()) \
    #                            for v1, v2 in zip(bins[:-1], bins[1:])])

    a_merge = np.hstack([idx_new,
                         Y_merge.reshape(-1, 1),
                         Yhat_merge.reshape(-1, 1),
                         std_merge.reshape(-1, 1)
                         ]
                        )
    # a_merge = np.hstack([bins[:-1].reshape(-1, 1),
    #                      bins[1:].reshape(-1, 1),
    #                      Y_merge.reshape(-1, 1),
    #                      Yhat_merge.reshape(-1, 1),
    #                      std_merge.reshape(-1, 1)
    #                      ]
    #                     )

    df_merge = pd.DataFrame(a_merge, columns=['CHROM', 'START', 'END', 'Y_TRUE', 'Y_PRED', 'STD'])
    # df_merge = pd.DataFrame(a_merge, columns=['START', 'END', 'Y_TRUE', 'Y_PRED', 'STD'])
    # df_merge.insert(0, 'CHROM', df.CHROM.iloc[0])

    return df_merge
