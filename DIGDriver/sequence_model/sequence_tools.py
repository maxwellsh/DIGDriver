import pandas as pd
import numpy as np
import json
import itertools as it
import pysam
import multiprocessing as mp
import subprocess as sp
import h5py
import pybedtools

from DIGDriver.data_tools import mutation_tools
from DIGDriver.sequence_model import genic_driver_tools

DNA53 = 'NTCGA'
DNA35 = 'NAGCT'
trans = DNA53.maketrans(DNA53, DNA35)

def reverse_complement(seq):
    return seq[::-1].translate(trans)

def fetch_sequence(fasta, CHROM, START, END, n_up=2, n_down=2):
    """ Fetch a sequence expanded by one bp on either end
        to allow for trinucleotide counting of all positions
    """
    if START == 0:
        START = n_up

    seq = fasta.fetch(CHROM, START-n_up, END+n_down).upper()
    return seq, START-n_up, END+n_down

def mk_context_sequences(n_up=2, n_down=2, collapse=False):
    DNA = 'ACGT'
    NUC = 'ACGT'
    if collapse:
        NUC = 'CT'

    prod_items = [DNA]*n_up + [NUC] + [DNA]*n_down

    keys = [''.join(tup) for tup in it.product(*prod_items)]
    return {key: 0 for key in keys}

def seq_to_context(seq, baseix=2, collapse=False):
    """ Convert any sequence into
        its unique nucleotide context

    kwarg   baseix: the index of the base around which the context is constructed
    """
    if 'N' in seq:
        return ''

    if collapse:
        if seq[baseix] == 'G' or seq[baseix] == 'A':
            return reverse_complement(seq)

    return seq

def type_mutation(REF, ALT, collapse=False):
    if collapse:
        if REF == 'G' or REF == 'A':
            REF = REF.translate(trans)
            ALT = ALT.translate(trans)

    return "{}>{}".format(REF, ALT)

def count_sequence_context(seq, n_up=2, n_down=2, nuc_dict=None, collapse=False):
    """ Count the nucleotides context present in a sequence
    """
    if not nuc_dict:
        nuc_dict = mk_context_sequences(n_up=n_up, n_down=n_down, collapse=collapse)

    for i in range(n_up, len(seq)-n_down):
        substr = seq_to_context(seq[i-n_up:i+n_down+1], baseix=n_up, collapse=collapse)
        if not substr:
            continue

        nuc_dict[substr] += 1

    return nuc_dict

def count_contexts_by_regions(f_fasta, chrom_lst, start_lst, end_lst, n_up=2, n_down=2, collapse=False):
    """ Sequence context counts within a set of regions
    """

    fasta = pysam.FastaFile(f_fasta)
    # print(set(chrom_lst), end = " ")

    idx_lst = []
    dict_lst = []
    for CHROM, START, END in zip(chrom_lst, start_lst, end_lst):
        seq, _, _ = fetch_sequence(fasta, CHROM, START, END, n_up=n_up, n_down=n_down)
        dict_lst.append(count_sequence_context(seq, n_up=n_up, n_down=n_down, collapse=collapse))
        idx_lst.append("{}:{}-{}".format(CHROM, START, END))

    return pd.DataFrame(dict_lst, index=idx_lst)

def count_contexts_in_bed(f_fasta, df_bed, n_up=1, n_down=1, N_proc=1, N_chunk=10, collapse=False):
    """ Count nucleotide contexts within regions of a bed-like dataframe
    """
    chrom_lst = ['chr{}'.format(val) for val in df_bed.iloc[:, 0].values]
    start_lst = df_bed.iloc[:, 1].values
    end_lst = df_bed.iloc[:, 2].values

    chunk_size = int(len(chrom_lst) / N_chunk)
    chunk_idx = list(range(0, len(chrom_lst), chunk_size)) + [len(chrom_lst)]

    res = []
    pool = mp.Pool(N_proc)
    for idx_start, idx_end in zip(chunk_idx[:-1], chunk_idx[1:]):
        chrom_tmp = chrom_lst[idx_start:idx_end]
        start_tmp = start_lst[idx_start:idx_end]
        end_tmp   =   end_lst[idx_start:idx_end]

        res.append(pool.apply_async(count_contexts_by_regions, (f_fasta, chrom_tmp, start_tmp, end_tmp),
                                                          dict(n_up=n_up, n_down=n_down, collapse=collapse)
                                   )
                  )
        # res.append(count_trinuc_regions(fasta, chrom_tmp, start_tmp, end_tmp))

    pool.close()
    pool.join()

    df_lst = [r.get() for r in res]
    # df_lst = res

    df_cnt = pd.concat(df_lst)
    # df = df_pos.merge(df_cnt, left_index=True, right_index=True)

    return df_cnt

def mutation_contexts_by_chrom(f_fasta, df, n_up=2, n_down=2, collapse=False):
    fasta = pysam.FastaFile(f_fasta)
    CHROM = str(df.CHROM.iloc[0])
    if not CHROM.startswith('chr'):
        CHROM = "chr{}".format(CHROM)

    seq = fasta.fetch(CHROM).upper()

    cntxt_lst = []
    muttype_lst = []
    prev_start = -1
    prev_alt = ''
    prev_ref = ''

    for START, REF, ALT in zip(df.START.values, df.REF.values, df.ALT.values):
        if seq[START] != REF:
            print('WARNING: REF {} does not match sequence {} at {}. Mutation will be removed.'.format(REF, seq[START], START))
            mut = type_mutation(REF, ALT, collapse=collapse)
            substr = ""

        elif START == prev_start:
            substr = cntxt_lst[-1]

            if ALT == prev_alt:
                mut = muttype_lst[-1]
            else:
                mut = type_mutation(REF, ALT, collapse=collapse)

        else:
            substr = seq_to_context(
                        seq[START-n_up:START+n_down+1], baseix=n_up, collapse=collapse
                     )

            if ALT == prev_alt and REF == prev_ref:
                mut = muttype_lst[-1]
            else:
                mut = type_mutation(REF, ALT, collapse=collapse)

        muttype_lst.append(mut)
        cntxt_lst.append(substr)

        prev_start = START
        prev_ref   = REF
        prev_alt   = ALT

    df.insert(df.shape[1], 'MUT_TYPE', muttype_lst)
    df.insert(df.shape[1], 'CONTEXT', cntxt_lst)

    return df[df.CONTEXT != ""]

def add_context_to_mutations(f_fasta, df_mut, n_up=2, n_down=2, N_proc=1, collapse=False):
    """ Add sequence context annotations to mutations
    """
    # cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'ID']
    # df_mut = pd.read_csv(f_mut, sep="\t", names=cols)

    df_indel = df_mut[df_mut.ANNOT.str.contains('INDEL')]
    df_mut = df_mut[~df_mut.ANNOT.str.contains('INDEL')]
    # print(len(df_mut), len(df_indel))

    if len(df_mut) > 0:
        res = []
        pool = mp.Pool(N_proc)
        for chrom, df in df_mut.groupby('CHROM'):
            if 'MT' in str(chrom):
                continue

            res.append(pool.apply_async(
                            mutation_contexts_by_chrom,
                            (f_fasta, df), dict(n_up=n_up, n_down=n_down, collapse=collapse)
                       )
                      )
            # res.append(trinuc_mutation_by_chrom(f_fasta, df))

        pool.close()
        pool.join()

        df_lst = [r.get() for r in res]
        df_out = pd.concat(df_lst)

    ## Deal with INDELS:
    if len(df_indel) > 0:
        print('Adding context to indels')
        df_indel = df_indel.rename({'ANNOT': 'MUT_TYPE'}, axis=1)
        df_indel.insert(df_indel.shape[1]-1, 'ANNOT', 'INDEL')
        df_indel.insert(df_indel.shape[1], 'CONTEXT', '.')

        if len(df_mut) > 0:
            df_out = pd.concat([df_out, df_indel]).sort_values(['CHROM', 'START', 'END'])
        else:
            df_out = df_indel.sort_values(['CHROM', 'START', 'END'])

    return df_out

def bgzip(filename):
    """Call bgzip to compress a file."""
    sp.run(['bgzip', '-f', filename])

def tabix_index(filename, preset="bed", skip=1, comment="#"):
    """Call tabix to create an index for a bgzip-compressed file."""
    sp.run(['tabix', '-p', preset, '-S {}'.format(skip), filename])

def mk_mutation_context(n_up=1, n_down=1, collapse=False, return_df=False):
    DNA = 'ACGT'
    prod_items_T = [DNA]*n_up + ['T'] + [DNA]*n_down
    prod_items_C = [DNA]*n_up + ['C'] + [DNA]*n_down

    keys_T = [''.join(tup) for tup in it.product(*prod_items_T)]
    keys_C = [''.join(tup) for tup in it.product(*prod_items_C)]

    muts_T = ['T>A', 'T>G', 'T>C']
    muts_C = ['C>A', 'C>G', 'C>T']

    if collapse:
        tups = [tup for tup in it.product(muts_C, keys_C)] + \
               [tup for tup in it.product(muts_T, keys_T)]

    else:
        prod_items_A = [DNA]*n_up + ['A'] + [DNA]*n_down
        prod_items_G = [DNA]*n_up + ['G'] + [DNA]*n_down

        keys_A = [''.join(tup) for tup in it.product(*prod_items_A)]
        keys_G = [''.join(tup) for tup in it.product(*prod_items_G)]

        muts_A = ['A>T', 'A>C', 'A>G']
        muts_G = ['G>T', 'G>C', 'G>A']

        tups = [tup for tup in it.product(muts_A, keys_A)] + \
               [tup for tup in it.product(muts_C, keys_C)] + \
               [tup for tup in it.product(muts_G, keys_G)] + \
               [tup for tup in it.product(muts_T, keys_T)]

    # multi_idx = pd.MultiIndex.from_tuples(tups, sortorder=1)
    # # print(multi_idx)
    # return pd.Series([0]*96, index=multi_idx)

    d = {tup: 0 for tup in tups}
    df = pd.DataFrame(tups, columns=['MUT_TYPE', 'CONTEXT'])
    # df['COUNT'] = 0
    # for mut in muts_C + muts_T:
    #     if mut.startswith('C'):
    #         d[mut] = {key: 0 for key in keys_C}
    #     else:
    #         d[mut] = {key: 0 for key in keys_T}

    if return_df:
        return df

    return d

    # return tups

def mk_trans_idx(n_up=1, n_down=1, collapse=False):
    """ Make a list of all possible transitions of the form ATG>AGG
    """
    d = mk_mutation_context(n_up=n_up, n_down=n_down, collapse=collapse)
    keys = list(d.keys())
    trans_idx = sorted([k[1] + '>' + k[1][:n_up] + k[0][2] + k[1][n_up+1:] for k in keys])

    return trans_idx


def base_probabilities_by_region(fasta, S_prob, CHROM, START, END, n_up=2, n_down=2, normed=True, collapse=False):
    """ Get the probability of mutation at every position across a region
    """
    seq, start, end = fetch_sequence(fasta, CHROM, START, END, n_up=n_up, n_down=n_down)

    probs = []
    poss = []
    # trinucs = []
    for i in range(n_up, len(seq)-n_down):
        poss.append(start+i)
        substr = seq_to_context(seq[i-n_up:i+n_down+1], baseix=n_up, collapse=collapse)
        # trinucs.append(substr)
        if not substr:
            probs.append(0)
            continue

        probs.append(S_prob[substr])

    probs = np.array(probs)
    poss = np.array(poss)
    # trinucs = np.array(trinucs)

    if normed:
        probs = probs / np.sum(probs)

    return probs, poss
    # return probs, poss, trinucs

##duplicated function name and use of pass?
def train_sequence_model(regions, df_mut, genome_counts, n_up=1, n_down=1, key_prefix=None):
    """ Train a trinucleotide sequence model based on precalculated
        trinucleotide frequencies and mutation counts
    """
    pass
    # 1. Restrict df_mut to whitelist regions
    df_bed = pd.DataFrame(regions, columns=['CHROM', 'START', 'END'])
    # df_mut_white = df_mut
    df_mut_white = mutation_tools.restrict_mutations_by_bed(df_mut, df_bed, unique=True, remove_X=False)
    df_mut_white.columns = df_mut.columns

    # print(len(df_mut))
    # print(len(df_mut_white))

    # 2. Calculate trinucleotide mutation counts
    df_ct_empty = mk_mutation_context(n_up=n_up, n_down=n_down, collapse=False, return_df=True)
    df_ct = df_mut_white.groupby(['MUT_TYPE', 'CONTEXT']).size().reset_index(name='COUNT')
    df_ct = df_ct_empty.merge(df_ct, on=['MUT_TYPE', 'CONTEXT'], how='left')
    df_ct.loc[df_ct.COUNT.isna(), 'COUNT'] = 0

    # 3. Pass mutation and genome counts to mutation_freq_conditional
    df_freq_mut = mutation_freq_conditional(df_ct, genome_counts)
    # df_freq_mut.index = [(row.MUT_TYPE, row.CONTEXT) for i, row in df_freq_mut.iterrows()]
    # print(df_freq_mut)
    # for i, row in df_freq.iterrows():
    #     print(row.MUT_TYPE, row.CONTEXT, row.FREQ)

    # 4. Summarize into 64 trinucleotide contexts
    df_freq_context = df_freq_mut.pivot_table('FREQ', index=['CONTEXT'], aggfunc=np.sum)
    # print(df_freq_context)
    # for cont, row in df_freq_context.iterrows():
    #     print(cont, row.FREQ)

    return df_freq_mut, df_freq_context

def mutation_freq_conditional(df_freq, S_gen):
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
    # S_mut_norm = S_mut.copy().astype(float)
    df_freq["FREQ"] = [row.COUNT / S_gen[row.CONTEXT] for i, row in df_freq.iterrows()]

    return df_freq

def si_count_pretrain(gene_lst, f_genic_str, f_fasta, window):
    """
    For a given list of gene names, find the context counts for all 192 context transitions
    (in trinuc_i>trinuc_f format).
    """
    fasta = pysam.FastaFile(f_fasta)
    with h5py.File(f_genic_str, 'r') as f_genic:
        trans_idx = f_genic['substitution_idx'][:].astype(str)
        si_accum = pd.DataFrame(index = trans_idx)
        si_accum.sort_index(inplace=True)
        for gene in gene_lst:
            chrom = f_genic['chr'][gene][:][0].decode("utf-8")
            strd = f_genic['strands'][gene][:][0]
            intervals = f_genic['cds_intervals'][gene][:]
            regions_overlapped = [genic_driver_tools.trip_to_str(r) for r in genic_driver_tools.get_ideal_overlaps(chrom, intervals, window)]
            s_i = si_by_regions(fasta, trans_idx, regions_overlapped, strand=strd)
            si_accum[gene] = s_i.sort_index()[0]
        out = si_accum.T
    return out


def si_by_regions(fasta, trans_idx, regions, strand=1, n_up=1, n_down=1, normed=True):
    """ Get the probability of mutation at every position across the region that a gene overlaps
    """

    keys = set(list(trans_idx))
    d = {key: 0 for key in keys}
    all_nucs = set('ATCG')
    for r in regions:
        splt1 = r.split('-')
        end = int(splt1[1])
        splt2 = splt1[0].split(':')
        start = int(splt2[1])
        chrom = splt2[0]
        seq, _, _ = fetch_sequence(fasta, chrom, start, end, n_up=n_up, n_down=n_down)
        if strand == -1 or strand == '-':
            seq = reverse_complement(seq)

        for i in range(n_up, len(seq)-n_down):
            substr = seq_to_context(seq[i-n_up:i+n_down+1], baseix=n_up)
            # trinucs.append(substr)
            if not substr:
                continue

            alts = ''.join(list(all_nucs.difference(substr[n_up])))
            prod_items = [substr[:n_up]] + [alts] + [substr[n_down + 1:]]
            trans = [''.join(tup) for tup in it.product(*prod_items)]
            trans = [substr + '>' + t for t in trans]
            for t in trans:
                d[t] += 1
    return pd.DataFrame(d.values(), index = list(d.keys()))

#top level parallel runner for context counting
def si_count_parallel(f_genic_str, f_fasta, window, n_procs):
    """
    parallel runner for fast si count pretraining
    """

    with h5py.File(f_genic_str, 'r') as f_genic:
        all_genes = list(f_genic['cds_intervals'].keys())
    chunksize = int(np.ceil(len(all_genes) / n_procs))
    res = []
    pool = mp.Pool(n_procs)
    for i in np.arange(0, len(all_genes), chunksize):
        gene_chunk = all_genes[i:i+chunksize]

        r = pool.apply_async(si_count_pretrain, (gene_chunk, f_genic_str, f_fasta, window))
        res.append(r)

    pool.close()
    pool.join()

    res_lst = [r.get() for r in res]
    results = pd.concat(res_lst)
    return results

def initialize_nonc_data(f_nonc_data_str, f_genome_counts, window, n_up=1, n_down=1):
    ## extract window size key from window parameter
    window_key = 'window_{}'.format(window)
    # if window < 1000:
    #     print("Warning: Model is not intended for use with windows < 1kb")
    #     window_key = '{}bp'.format(window)
    # else:
    #     window_key = '{}kb'.format(int(window/1000))

    with h5py.File(f_nonc_data_str, 'a') as f_nonc_data:
        keys = f_nonc_data.keys()

        ## add substitution index to file root if it doesnt already exist
        if not 'substitution_idx' in keys:
            trans_idx = mk_trans_idx(n_up=n_up, n_down=n_down, collapse=False)
            b_trans_idx = np.array([t.encode('ascii') for t in trans_idx])
            f_nonc_data.create_dataset('substitution_idx', data = b_trans_idx)
        ## otherwise load subsitution index
        else:
            trans_idx = f_nonc_data['substitution_idx'][:].astype(str)
        ## add genome counts to file under window key if they do not already exist
        if not ('{}/full_window_si_index'.format(window_key) in keys and '{}/full_window_si_values'.format(window_key) in keys):
            with h5py.File(f_genome_counts, 'r') as f_genome:
                idx = f_genome['idx'][:]
            genome_df = pd.read_hdf(f_genome_counts, 'all_window_genome_counts')
            assert int(genome_df.index[0].split('-')[-1]) == window ## ensure sure correct genome count window size
            f_nonc_data.create_dataset('{}/full_window_si_values'.format(window_key), data = genome_df.values, dtype = int)
            f_nonc_data.create_dataset('{}/full_window_si_index'.format(window_key), data=idx)


def precount_region_contexts_parallel(f_nonc_bed, f_fasta, n_procs, window, sub_elts = True, n_up=1, n_down=1):
    """ counts contexts within each sub element. Assumes bed 12 input

    MS NOTES:
        * Why are there duplicated rows that need to be removed in the final line?
        --> this is to ensure that only one row is returned when we are reading in these values by index
            within the nonc_model
    """
#    with h5py.File(f_nonc_data_str, 'r') as f_nonc_data:
#        trans_idx = f_nonc_data['substitution_idx'][:].astype(str)

    trans_idx = mk_trans_idx(n_up=1, n_down=1, collapse=False)

    if sub_elts:
        bed12_file = pybedtools.BedTool(f_nonc_bed)
        df6 = bed12_file.bed6().to_dataframe(header=None, names=['CHROM', 'START', 'END', 'ELT', 'SCORE', 'STRAND'], low_memory=False)
        if 'chr' in str(df6.CHROM[0]):
            df6['CHROM'] = df6['CHROM'].map(lambda x: x.lstrip('chr'))

        chunksize = int(np.ceil(len(df6) / n_procs))
        res = []
        pool = mp.Pool(n_procs)
        all_regions = list(zip(df6.CHROM, df6.START, df6.END, df6.STRAND))
    else:
        bed12_file = pd.read_csv(f_nonc_bed, sep='\t', header=None, names=None, low_memory=False)
        if 'chr' in str(bed12_file[0][0]):
            bed12_file[0] = bed12_file[0].map(lambda x: x.lstrip('chr'))
        chunksize = int(np.ceil(len(bed12_file) / n_procs))
        res = []
        pool = mp.Pool(n_procs)
        all_regions = list(zip(bed12_file[0], bed12_file[1], bed12_file[2], bed12_file[5]))

    for i in np.arange(0, len(all_regions), chunksize):
        region_chunk = all_regions[i:i+chunksize]

        r = pool.apply_async(nonc_elt_context_count, (region_chunk, trans_idx, f_fasta))
        res.append(r)

    pool.close()
    pool.join()

    res_lst = [r.get() for r in res]
    results = pd.concat(res_lst)

    return results.loc[~results.index.duplicated()]

def nonc_elt_context_count(regions, trans_idx, f_fasta, n_up=1, n_down=1):
    """ Get the number of contexts within each element
    """

    fasta = pysam.FastaFile(f_fasta)

    keys = set(list(trans_idx))

    dic = {key: 0 for key in keys}
    all_nucs = set('ATCG')
    si_accum = pd.DataFrame(index = trans_idx)
    si_accum.sort_index(inplace=True)

    # fasta = pysam.FastaFile(f_fasta)
    # print(set(chrom_lst), end = " ")

    idx_lst = []
    dict_lst = []
    #  for CHROM, START, END in zip(chrom_lst, start_lst, end_lst):
    for region in regions:
        chrom = 'chr' + str(region[0])
        start = region[1]
        end = region[2]
        strand = region[3]
        seq, _, _ = fetch_sequence(fasta, chrom, start, end, n_up=n_up, n_down=n_down)
        if strand == '-' or strand == -1:
            seq = reverse_complement(seq)

        dict_lst.append(count_sequence_context(seq, n_up=n_up, n_down=n_down, collapse=False))
        idx_lst.append("{}:{}-{}".format(chrom, start, end))

    ## Expand to full mutation context
    df_64 = pd.DataFrame(dict_lst, index=idx_lst)
    df_192 = pd.DataFrame(0., columns=sorted(keys), index=df_64.index)

    for key in sorted(keys):
        key_64 = key.split('>')[0]
        df_192.loc[:, key] = df_64.loc[:, key_64]

    return df_192

    # for region in regions:
    #     d = dic.copy()
    #     chrom = 'chr' + str(region[0])
    #     start = region[1]
    #     end = region[2]
    #     strand = region[3]
    #     seq, _, _ = fetch_sequence(fasta, chrom, start, end, n_up=n_up, n_down=n_down)

    #     if strand == '-' or strand == -1:
    #         seq = reverse_complement(seq)
    #
    #     for i in range(n_up, len(seq)-n_down):
    #         substr = seq_to_context(seq[i-n_up:i+n_down+1], baseix=n_up)
    #         # trinucs.append(substr)
    #         if not substr:
    #             continue

    #         alts = ''.join(list(all_nucs.difference(substr[n_up])))
    #         prod_items = [substr[:n_up]] + [alts] + [substr[n_down + 1:]]
    #         trans = [''.join(tup) for tup in it.product(*prod_items)]
    #         trans = [substr + '>' + t for t in trans]
    #         for t in trans:
    #             d[t] += 1
    #     # to_add = pd.DataFrame(d.values(), index = list(d.keys()))
    #     # region_str = genic_driver_tools.trip_to_str(region)
    #     # si_accum[region_str] = to_add.sort_index()[0]
    # # return si_accum.T

def preprocess_nonc(f_nonc_bed, f_nonc_data, f_pretrained, L_contexts, save_key, window):

    # if window < 1000:
    #     print("Warning: Model is not intended for use with windows < 1kb")
    #     window_key = '{}bp'.format(window)
    # else:
    #     window_key = '{}kb'.format(int(window/1000))
    window_key = 'window_{}'.format(window)

    nonc_data = h5py.File(f_nonc_data, 'a')

    idx = nonc_data['{}/full_window_si_index'.format(window_key)][:]
    idx_dict = dict(zip(map(tuple, idx), range(len(idx))))
    df_mut = pd.read_hdf(f_pretrained, key='sequence_model_192')
    mut_model_idx = [r[1] + '>' + r[1][0] + r[0][2] + r[1][2] for r in zip(df_mut.MUT_TYPE, df_mut.CONTEXT)]
    subst_idx = sorted(mut_model_idx)
    revc_subst_idx = [reverse_complement(sub.split('>')[0]) + '>' + reverse_complement(sub.split('>')[\
    -1]) for sub in subst_idx]
    revc_dic = dict(zip(subst_idx, revc_subst_idx))

    df_elts = mutation_tools.bed12_boundaries(f_nonc_bed)

    i = 0
    for _, row in df_elts.iterrows():
        # if i % 10000 == 0:
        #     print("{} of {}".format(i, len(df_elts)))

        chrom = row['CHROM']
        elt = row['ELT']
        strand = row['STRAND']
        block_starts = row['BLOCK_STARTS']
        block_ends = row['BLOCK_ENDS']
        elts_as_intervals = np.vstack((block_starts, block_ends))
        overlaps = genic_driver_tools.get_ideal_overlaps(chrom, elts_as_intervals, window)
        region_counts = np.array([np.repeat(nonc_data['{}/full_window_si_values'.format(
            window_key)][idx_dict[region], :], 3) for region in overlaps]).sum(axis=0)
        #if negative strand, take the reverse complement of the region counts
        if strand == '-1' or strand == '-':
            region_counts = [r[1] for r in sorted(enumerate(region_counts), key=lambda k: revc_dic[subst_idx[k[0]]])]
        L = np.zeros((192))
        for start, end in zip(block_starts, block_ends):
            L += L_contexts.loc['chr{}:{}-{}'.format(chrom, start,end)].values

        nonc_data.create_dataset('{}/{}/{}/L_counts'.format(window_key, save_key, elt), data=L)
        nonc_data.create_dataset('{}/{}/{}/region_counts'.format(window_key, save_key, elt), data=region_counts)
        nonc_data['{}/{}/{}'.format(window_key, save_key, elt)].attrs.create('overlaps', overlaps)
        i += 1

    nonc_data.close()

## Note that the f_pretrained here can be any cancer, as it is only used for the sequence model
def preprocess_sites(f_sites, f_nonc_data, f_pretrained, save_key, window):
    window_key = 'window_{}'.format(window)
    # if window < 1000:
    #     print("Warning: Model is not intended for use with windows < 1kb")
    #     window_key = '{}bp'.format(window)
    # else:
    #     window_key = '{}kb'.format(int(window/1000))

    nonc_data = h5py.File(f_nonc_data, 'a')

    idx = nonc_data['{}/full_window_si_index'.format(window_key)][:]
    idx_dict = dict(zip(map(tuple, idx), range(len(idx))))
    df_mut = pd.read_hdf(f_pretrained, key='sequence_model_192')
    mut_model_idx = [r[1] + '>' + r[1][0] + r[0][2] + r[1][2] for r in zip(df_mut.MUT_TYPE, df_mut.CONTEXT)]
    subst_idx = sorted(mut_model_idx)
    revc_subst_idx = [reverse_complement(sub.split('>')[0]) + '>' + reverse_complement(sub.split('>')[\
    -1]) for sub in subst_idx]
    revc_dic = dict(zip(subst_idx, revc_subst_idx))

    keys = set(list(subst_idx))
    d = {key: 0 for key in sorted(keys)}

    df_sites = mutation_tools.read_mutation_file(f_sites)
    df_sites = df_sites.drop(columns = ['GENE','ANNOT','REF','ALT']).rename(columns={'SAMPLE':'GENE'}).set_index('GENE')
    df_sites.loc[df_sites.CONTEXT.isna(), 'CONTEXT'] = 'nan'
    if not 'STRAND' in df_sites.columns:
        df_sites['STRAND'] = '.'

    def _overlaps_and_contexts(group):
        chrom = list(group['CHROM'])[0]
        starts = list(group['START'])
        ends = list(group['END'])
        strand = list(group['STRAND'])[0]
        intervals = np.array(list(zip(starts,ends))).T
        if strand == "-1" or strand == "-":
            contexts = [reverse_complement(r[1]) + '>' + reverse_complement(r[1][0] + r[0][2] + r[1][2]) \
                        for r in zip(list(group['MUT_TYPE']), list(group['CONTEXT']))]
        else:
            contexts = [r[1] + '>' + r[1][0] + r[0][2] + r[1][2] for r in zip(list(group['MUT_TYPE']), list(group['CONTEXT']))]
        overlaps = genic_driver_tools.get_ideal_overlaps(chrom, intervals, window)
        # overlaps = genic_driver_tools.get_ideal_overlaps(chrom, intervals, 10000)
        return overlaps, contexts, strand

    results = df_sites.groupby('GENE').apply(_overlaps_and_contexts)

    for gene, row in zip(results.index, results):
        region_counts = np.array([np.repeat(nonc_data['{}/full_window_si_values'.format(
            window_key)][idx_dict[region], :], 3) for region in row[0]]).sum(axis=0)
        #if negative strand, take the reverse complement of the region counts
        if row[2] == '-1' or row[2] == '-':
            region_counts = [r[1] for r in sorted(enumerate(region_counts), key=lambda k: revc_dic[subst_idx[k[0]]])]
        L_dict = d.copy()
        for trinuc in row[1]:
            if 'nan' in trinuc:
                continue
            L_dict[trinuc]+=1
        L = np.array(list(L_dict.values()))
        nonc_data.create_dataset('{}/{}/{}/L_counts'.format(window_key, save_key, gene), data=L)
        nonc_data.create_dataset('{}/{}/{}/region_counts'.format(window_key, save_key, gene), data=region_counts)
        nonc_data['{}/{}/{}'.format(window_key, save_key, gene)].attrs.create('overlaps', row[0])
        # nonc_data.create_dataset('{}/sites_data/{}/{}/L_counts'.format(window_key, save_key, gene), data=L)
        # nonc_data.create_dataset('{}/sites_data/{}/{}/region_counts'.format(window_key, save_key, gene), data=region_counts)
        # nonc_data['{}/sites_data/{}/{}'.format(window_key, save_key, gene)].attrs.create('overlaps', row[0])
    nonc_data.close()
    nonc_data.close()
