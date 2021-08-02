import h5py
import pandas as pd
import numpy as np
import scipy
import scipy.stats

def _load_fold_avg_DEPRECATED(f, cancer, test_idx=[], key='held-out'):
    """ Low-level loading of a single fold, removing outlier runs
    """
    hf = h5py.File(f, 'r')
    dset = hf[cancer]

    if not key in dset.keys():
        #print("WARNING: {} is not a key in the dataset. Defaulting to 'test'".format(key))
        key = 'test'

    runs = [int(key) for key in dset[key].keys() if key.isdigit()]
    test_Y = dset[key]['y_true'][:].reshape(-1, 1)
    if not len(test_idx):
        test_idx = dset[key]['chr_locs'][:]

    # print(test_idx.shape)
    test_Yhat_lst = []
    test_std_lst = []
    r2_lst = []
    run_lst = []
    for run in runs:
        y_hat = dset[key]['{}'.format(run)]['mean'][:].reshape(-1, 1)
        #gets rid of runs with all means predicted the same (casuses nan pearsonr)
        # if (y_hat-y_hat.mean()).sum() == 0:
        #     continue
        r2 = scipy.stats.pearsonr(test_Y.squeeze(), y_hat.squeeze())[0]**2

        if np.isnan(r2):
            continue

        r2_lst.append(r2)
        test_Yhat_lst.append(y_hat)
        test_std_lst.append(dset[key]['{}'.format(run)]['std'][:].reshape(-1, 1))
        run_lst.append(run)
        # print(r2_lst[-1])

    hf.close()
    r2s = np.array(r2_lst)
    # print(r2s)
    med = np.median(r2s)
    mad = np.median(np.abs(r2s - med))

    # idx = np.array(run_lst)[r2s > (med - 2*mad)]
    idx = np.where(r2s > (np.max(r2s) - 2*mad))
    if not len(idx[0]):
        idx = np.arange(len(test_Yhat_lst))

    test_Yhat = np.array(test_Yhat_lst)[idx].mean(axis = 0)
    test_std = np.array(test_std_lst)[idx].mean(axis = 0)
    vals = np.hstack([test_idx, test_Y, test_Yhat, test_std])
    df = pd.DataFrame(vals, 
            columns=['CHROM', 'START', 'END', 'Y_TRUE', 'Y_PRED', 'STD']
    )
    # print(df[0:5])

    return df

def _load_fold_avg(f, cancer, key='held-out', fold=None):
    """ Low-level loading of a single fold
    """
    h5 = h5py.File(f, 'r')
    out_h5 = h5[cancer]
    # if not fold:
    #     fold = int(f.split('.h5')[0].split('_')[-1])

    assert key in out_h5, 'Cannot compute pretrained model with no saved held-out set. Existing feilds are: {}'.format(out_h5.keys())
    ds = out_h5['held-out']

    runs = [key for key in ds.keys() if key.isdigit()]
    # test_Y = dset[key]['y_true'][:].reshape(-1, 1)
    # test_idx = dset[key]['chr_locs'][:]

    chr_locs = ds['chr_locs'][:]
    mapps = ds['mappability'][:].reshape(-1, 1)
    quants = ds['quantiles'][:].reshape(-1, 1)
    y_true = ds['y_true'][:].reshape(-1, 1)
    mean_lst = []
    std_lst = []

    for i in runs:
        mean_lst.append(ds[i]['mean'][:])
        std_lst.append(ds[i]['std'][:])

    means = np.array(mean_lst).mean(axis=0).reshape(-1, 1)
    stds = np.array(std_lst).mean(axis=0).reshape(-1, 1)

    vals = np.hstack([chr_locs, y_true, means, stds, mapps, quants])
    df = pd.DataFrame(vals, 
            columns=['CHROM', 'START', 'END', 'Y_TRUE', 'Y_PRED', 'STD', 'MAPP', 'QUANT']
    )
    # df['FOLD'] = fold
    # print(df[0:5])

    return df

def kfold_supmap_results(kfold_path, cancer_str, key='held-out', drop_pos_cols=False, sort=True):
    """ Load kfold results for regions above the user-defined mappability threshold
    """
    fold_files = sorted(kfold_path.glob("gp_results_fold*.h5"))
    df_lst = [_load_fold_avg(str(fold), cancer=cancer_str, key=key) for fold in fold_files]
    df = pd.concat(df_lst).astype({'CHROM':int, 
                                   'START':int, 
                                   'END':int, 
                                   'Y_TRUE':int, 
                                   'Y_PRED':float, 
                                   'STD':float,
                                   'MAPP': float,
                                   'QUANT': float})
    # window = int(df.iloc[0]['END'] - df.iloc[0]['START'])
    df['FLAG'] = False
    df['Region'] = ['chr{}:{}-{}'.format(row[0], row[1], row[2]) \
                    for row in zip(df.CHROM, df.START, df.END)]

    if sort:
        df = df.sort_values(by=['CHROM', 'START'])

    if drop_pos_cols:
        df = df.drop(['CHROM', 'START', 'END'], axis = 1)

    df.set_index('Region', inplace=True)

    return df

def kfold_submap_results(kfold_path, cancer_str, key='held-out', drop_pos_cols=False, sort=True):
    """ Load kfold results for regions below mappabiliy threshold
    """
    fold_files = sorted(kfold_path.glob("sub_mapp_results_fold*.h5"))
    df_lst = [_load_fold_avg(str(fold), cancer=cancer_str, key=key) for fold in fold_files]

    a_mean = np.array([df.Y_PRED.values for df in df_lst])
    mean = np.mean(a_mean, axis=0)

    a_std = np.array([df.STD.values for df in df_lst])
    std = np.mean(a_std, axis=0)

    df = pd.DataFrame({'CHROM': df_lst[0].CHROM.values,
                       'START': df_lst[0].START.values,
                       'END': df_lst[0].END.values,
                       'Y_TRUE': df_lst[0].Y_TRUE.values,
                       'Y_PRED': mean,
                       'STD': std,
                       'MAPP': df_lst[0].MAPP.values,
                       'QUANT': df_lst[0].QUANT.values,
                       }
                      ).astype({'CHROM':int, 'START':int, 'END':int, 'Y_TRUE':int, 
                                'Y_PRED':float, 'STD':float, 'MAPP':float, 'QUANT':float})

    # window = int(df.iloc[0]['END'] - df.iloc[0]['START'])
    df['FLAG'] = True
    df['Region'] = ['chr{}:{}-{}'.format(row[0], row[1], row[2]) \
                    for row in zip(df.CHROM, df.START, df.END)]

    if sort:
        df = df.sort_values(by=['CHROM', 'START'])

    if drop_pos_cols:
        df = df.drop(['CHROM', 'START', 'END'], axis = 1)

    df.set_index('Region', inplace=True)

    return df #, window

def kfold_results(kfold_path, cohort_name, key='held-out'):
    """ Load kfold results and remove outlier runs
    """
    try:
        df_sup = kfold_supmap_results(kfold_path, cohort_name, key=key)
        df_sub = kfold_submap_results(kfold_path, cohort_name, key=key)
    except:
    # except KeyError as e:
        raise Exception('ERROR: failed to load kfold {}. You should rerun the CNN+GP kfold.'.format(kfold_path))
        # print('FAIL: {}'.format(kfold_path))
        # print('\nERROR: uh oh there was an error loading the kfold results.')
        # print('This probably means a CNN+GP run crashed (it happens).')
        # print('Rerunning the CNN+GP kfold should fix the problem')

    # print(scipy.stats.pearsonr(df_sup.Y_TRUE, df_sup.Y_PRED)[0]**2)
    # print(scipy.stats.pearsonr(df_sub.Y_TRUE, df_sub.Y_PRED)[0]**2)

    df = pd.concat([df_sup, df_sub]).sort_values(by=['CHROM', 'START'])
    df_dedup = df.drop_duplicates(['CHROM', 'START', 'END'])
    assert len(df) == len(df_dedup), \
        "Oh snap! There are duplicate entries in the folds. You should rerun this kfold."

    print(scipy.stats.pearsonr(df[~df.FLAG].Y_TRUE, df[~df.FLAG].Y_PRED)[0]**2)

    return df
