import numpy as np
import pandas as pd
import bbi
import pysam
##only mappabilty_by_idx called from top level
def load_chromsizes(f_bw):
    chroms = bbi.chromsizes(f_bw)
    chroms.pop('chrM')
    chroms.pop('chrX')
    chroms.pop('chrY')

    return chroms

def mappability_by_window(f_mapp, window, overlap=0):
    chroms = load_chromsizes(f_mapp)

    mapp_lst = []
    for chr_id, chr_size in chroms.items():
        print(chr_id, end=' ')
        i = 0
        while i + window < chr_size:
            # print(i)
            mapp = bbi.fetch(f_mapp, chr_id, i, i + window, bins=1)[0]
            mapp_lst.append([chr_id, i, i+window, mapp])
            i += window - overlap

    return pd.DataFrame(np.array(mapp_lst),
                        columns=['CHROM', 'START', 'END', 'MAPP'])

def mappability_by_idx(f_mapp, idx):

    mapp_lst = []
    chr_prev = ''
    for row in idx:
        chr_id = 'chr{}'.format(row[0])
        start = row[1]
        end = row[2]

        if chr_id != chr_prev:
            print(chr_id)

        mapp = bbi.fetch(f_mapp, chr_id, start, end, bins=1)[0]
        mapp_lst.append([row[0], start, end, mapp])
        chr_prev = chr_id

    return mapp_lst

def P_bases_by_window(f_fasta, window, overlap=0):
    fasta = pysam.FastaFile(f_fasta)
    sizes = fasta.lengths
    chroms = fasta.references

    mapp_lst = []
    for chr_id, chr_size in zip(chroms, sizes):
        print(chr_id, end=' ')
        i = 0
        while i + window < chr_size:
            seq = fasta.fetch(chr_id, i, i + window)
            mapp = seq.count('P') / window
            mapp_lst.append([chr_id, i, i+window, mapp])
            i += window - overlap

    return pd.DataFrame(np.array(mapp_lst),
                        columns=['CHROM', 'START', 'END', 'MAPP'])
