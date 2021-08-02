import pandas as pd
import numpy as np
import pybedtools
import csv
import gzip

##deprecated by restrict_mutations_by_bed_efficient
def restrict_mutations_by_bed(df_mut, df_bed, unique=True, remove_X=True, replace_cols=False):
    """ Restrict mutations to only those that overlap elements in a bed file.
    """
    # df_mut = pd.read_table(f_mut, header=None, low_memory=False)
    # df_bed = pd.read_table(f_bed, header=None, low_memory=False)
    if remove_X:
        df_mut = df_mut[df_mut.iloc[:, 0] != "X"]
        df_bed = df_bed[df_bed.iloc[:, 0] != "X"]

    bed_mut = pybedtools.BedTool.from_dataframe(df_mut)
    bed_bed = pybedtools.BedTool.from_dataframe(df_bed)

    bed_inter = bed_mut.intersect(bed_bed)
    df_inter = bed_inter.to_dataframe()

    if unique:
        df_inter = df_inter.drop_duplicates()

    if replace_cols:
        # df_inter.columns = ['CHROM', 'START', 'END', 'REF', 'ALT', 'SAMPLE', 'GENE', 'ANNOT', 'MUT', 'CONTEXT']
        df_inter.columns = df_mut.columns

    return df_inter

def restrict_mutations_by_bed_efficient(f_mut, f_bed, bed12=False, drop_duplicates=False, drop_sex=False, replace_cols=False):
    """ Restrict mutations to only those that overlap elements in a bed file.
    """
    bed_mut = pybedtools.BedTool(f_mut)
    bed_bed = pybedtools.BedTool(f_bed)
    if bed12:
        bed_bed = bed_bed.bed12tobed6()

    bed_inter = bed_mut.intersect(bed_bed, wa=True)
    df_mut = read_mutation_file(bed_inter.fn, drop_duplicates=drop_duplicates, drop_sex=drop_sex)

    return df_mut

def read_mutation_file(path, drop_sex=True, drop_duplicates=False, unique_indels=True):
    # cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'SAMPLE', 'ANNOT', 'MUT_TYPE', 'CONTEXT']
    try:
        with open(path) as f:
            reader = csv.reader(f, delimiter='\t', skipinitialspace=True)
            first_row = next(reader)
    except UnicodeDecodeError:  ## file is probably gzipped
        with gzip.open(path, 'rt') as f:
            reader = csv.reader(f, delimiter='\t', skipinitialspace=True)
            first_row = next(reader)

    num_cols = len(first_row)

    if num_cols == 5:
        cols = ['CHROM', 'POS', 'REF', 'ALT', 'SAMPLE']
        dtype = dict(CHROM=str, POS=int, REF=str, ALT=str, SAMPLE=str)
    elif num_cols == 6:
        cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'SAMPLE']
        dtype = dict(CHROM=str, START=int, END=int, REF=str, ALT=str, SAMPLE=str)
    elif num_cols == 7:
        cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'SAMPLE', 'ANNOT']
        dtype = dict(CHROM=str, START=int, END=int, REF=str, ALT=str, SAMPLE=str, ANNOT=str)
    elif num_cols == 8:
        cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'SAMPLE', 'GENE', 'ANNOT']
        dtype = dict(CHROM=str, START=int, END=int, REF=str, ALT=str, SAMPLE=str, GENE=str, ANNOT=str)
    elif num_cols == 9:
        cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'SAMPLE', 'ANNOT', 'MUT_TYPE', 'CONTEXT']
        dtype = dict(CHROM=str, START=int, END=int, REF=str, ALT=str, SAMPLE=str, ANNOT=str, MUT_TYPE=str, CONTEXT=str)
    elif num_cols == 10:
        cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'SAMPLE', 'GENE', 'ANNOT', 'MUT_TYPE', 'CONTEXT']
        dtype = dict(CHROM=str, START=int, END=int, REF=str, ALT=str, SAMPLE=str, GENE=str, ANNOT=str, MUT_TYPE=str, CONTEXT=str)
    elif num_cols == 11:
        cols = ['CHROM', 'START', 'END', 'REF', 'ALT', 'SAMPLE', 'GENE', 'ANNOT', 'MUT_TYPE', 'CONTEXT', 'STRAND']
        dtype = dict(CHROM=str, START=int, END=int, REF=str, ALT=str, SAMPLE=str, GENE=str, ANNOT=str, MUT_TYPE=str, CONTEXT=str, STRAND=str)

    df = pd.read_csv(path, sep="\t", low_memory=False, names=cols, dtype=dtype)

    # df.columns = cols

    # df.CHROM = df.CHROM.astype(str)

    if drop_sex:
        if set(df.CHROM.unique()) - set([str(i) for i in range(1, 23)]):
            print('Restricting to autosomes')
            df = df[df.CHROM.isin([str(i) for i in range(1, 23)])]

        df['CHROM'] = df.CHROM.astype(int)

    # else:
    #     dtype['CHROM'] = str

    # df = df.astype(dtype)

    if drop_duplicates:
        df = drop_duplicate_mutations(df)

    if unique_indels:
        df = get_unique_indels(df)

    return df


def drop_duplicate_mutations(df_mut):
    df_dedup = df_mut.drop_duplicates(['CHROM', 'START', 'END', 'REF', 'ALT', 'SAMPLE'])
    return df_dedup

def get_unique_indels(df_mut):
    df_indel = df_mut[df_mut.ANNOT == 'INDEL']
    df_snv = df_mut[df_mut.ANNOT != 'INDEL']

    df_indel = df_indel.drop_duplicates(subset=['CHROM', 'START', 'END', 'REF', 'ALT', 'GENE'])

    return pd.concat([df_snv, df_indel])

def tabulate_nonc_mutations_split(f_nonc_bed, f_mut):
    try:
        # df_nonc = pd.read_csv(f_nonc_bed, sep= '\t', header=None, low_memory = False,
        # names=None)
        df_nonc = pd.read_table(f_nonc_bed, names=['CHROM', 'START', 'END', "ELT", "SCORE", "STRAND", 'thickStart', 'thickEnd', 'rgb', 'blockCount', 'blockSizes', 'blockStarts'], low_memory=False)
        df_nonc.CHROM = df_nonc.CHROM.astype(str)
        df_nonc = df_nonc[df_nonc.CHROM.isin([str(c) for c in range(1, 23)])]
        df_nonc.CHROM = df_nonc.CHROM.astype(int)
    except:
        raise Exception("ERROR: failed to load {}. make sure the bed file is in the correct bed12 format".format(df_nonc))
    df_mut = read_mutation_file(f_mut, drop_duplicates=True)
    assert ('GENE' in df_mut.columns and 'ANNOT' in df_mut.columns and'MUT_TYPE' in df_mut.columns)
    # try:
    #     df_mut = pd.read_csv(f_mut,sep='\t', low_memory=False, index_col=False,
    #     names=['CHROM', 'START', 'END', 'REF', 'ALT', 'ID', 'GENE', 'ANNOT', 'MUT','CONTEXT'])
    # except:
    #     raise Exception("ERROR: failed to load {}. make sure mut file is properly processed".format(df_mut))

    if 'chr' in str(df_nonc.CHROM[0]):
        df_nonc.CHROM = df_nonc.CHROM.map(lambda x: x.lstrip('chr'))

    bed_mut = pybedtools.BedTool.from_dataframe(df_mut)
    bed_bed = pybedtools.BedTool.from_dataframe(df_nonc)
    bed_split = bed_bed.bed6()

    bed_inter = bed_split.intersect(bed_mut, wao=True)
    df_inter = bed_inter.to_dataframe(header=None, names=np.arange(bed_inter.field_count()))
    df_inter = df_inter.drop(columns = [4,6,7,8,9,10,12,13,14,15])
    # df_split = df_inter.groupby([0,1,2,3,5]).agg({11:lambda x:len(set(x).difference(set('.'))), 16:np.sum}).reset_index()
    # df_split.columns = ['CHROM', 'START', 'END', 'ELT', 'STRAND','OBS_SAMPLES','OBS_MUT']

    df_whole = pd.pivot_table(df_inter, values = [1,2,11,16], index = [0,3,5], aggfunc={16:np.sum, 1:lambda x: sorted(set(x)), 2:lambda x: sorted(set(x)),11:lambda x:len(set(x).difference(set('.')))}).reset_index()
    df_whole.columns = ['CHROM', 'ELT', 'STRAND','BLOCK_STARTS', 'BLOCK_ENDS', 'OBS_SAMPLES', 'OBS_MUT']
    return None, df_whole
    # return df_split, df_whole

def tabulate_mutations_in_element(f_mut, f_elt_bed, bed12=False, drop_duplicates=False, all_elements=False,
    max_muts_per_sample=1e9, max_muts_per_elt_per_sample=3e9, return_blacklist=False):
    df_cnt = tabulate_muts_per_sample_per_element(f_mut, f_elt_bed, bed12=bed12, drop_duplicates=drop_duplicates)
    df_cnt.rename({'SAMPLE': 'OBS_SAMPLES'}, axis=1, inplace=True)

    # Remove hypermutated samples and cap total mutations from a sample in an element
    # *if there are any samples
    if len(df_cnt) > 0:
        df_cnt_sample = df_cnt.pivot_table(index='OBS_SAMPLES', values='OBS_MUT', aggfunc=np.sum)
        blacklist = df_cnt_sample[df_cnt_sample.OBS_MUT > max_muts_per_sample].index
        df_cnt = df_cnt[~df_cnt.OBS_SAMPLES.isin(blacklist)]
    else:
        blacklist = []
    df_cnt.loc[df_cnt.OBS_SNV > max_muts_per_elt_per_sample, 'OBS_SNV'] = max_muts_per_elt_per_sample
    df_cnt.loc[df_cnt.OBS_INDEL > max_muts_per_elt_per_sample, 'OBS_INDEL'] = max_muts_per_elt_per_sample

    df_summary = df_cnt.pivot_table(index='ELT',
        values=['OBS_SAMPLES', 'OBS_SNV', 'OBS_INDEL'],
        aggfunc={'OBS_SAMPLES': len, 'OBS_SNV': np.sum, 'OBS_INDEL': np.sum})
    if len(df_summary) == 0:
        df_summary = pd.DataFrame({'OBS_SAMPLES':[], 'OBS_SNV':[], 'OBS_INDEL':[], 'ELT':[]})
        df_summary = df_summary.set_index('ELT')
    ## Add elements with zero counts if required
    if all_elements:
        df_bed = pd.read_csv(f_elt_bed, sep="\t", header=None).set_index(3)
        df_bed.index.rename('ELT', inplace=True)
        df_summary = df_bed.merge(df_summary, left_index=True, right_index=True, how='left')
        df_summary.loc[df_summary.OBS_SNV.isna(), 'OBS_SNV'] = 0
        df_summary.loc[df_summary.OBS_INDEL.isna(), 'OBS_INDEL'] = 0
        df_summary.loc[df_summary.OBS_SAMPLES.isna(), 'OBS_SAMPLES'] = 0

    if return_blacklist:
        return df_summary[['OBS_SAMPLES', 'OBS_SNV', 'OBS_INDEL']], blacklist
    else:
        return df_summary[['OBS_SAMPLES', 'OBS_SNV', 'OBS_INDEL']]

def tabulate_muts_per_sample_per_element(f_mut, f_elt_bed, bed12=False, drop_duplicates=False, unique_indels=True):
    ## Convert dataframes to bedtools objects
    bed_mut = pybedtools.BedTool(f_mut)
    bed_elt = pybedtools.BedTool(f_elt_bed)

    if bed12:
        bed_elt = bed_elt.bed12tobed6()

    ## Intersect bedtools objects
    bed_inter = bed_mut.intersect(bed_elt, wa=True, wb=True)
    if len(bed_inter) == 0:
        return  pd.DataFrame({'ELT':[], 'SAMPLE':[], 'OBS_SNV':[], 'OBS_INDEL':[], 'OBS_MUT':[]})

    n_field = bed_inter.field_count()
    df_inter = bed_inter.to_dataframe(header=None, names=range(n_field), low_memory=False)

    if drop_duplicates:
        df_inter = df_inter.drop_duplicates([0, 1, 2, 3, 4, 5, 13]) ## Drop mutations duplicated by genic annotations

    # Separate SNVs and INDELs
    df_inter.loc[df_inter[7] != 'INDEL', 7] = 'SNV'
    df_inter_snv = df_inter[df_inter[7] == 'SNV']
    df_inter_ind = df_inter[df_inter[7] == 'INDEL']

    # if unique_indels:
    #     df_inter_ind = df_inter_ind.drop_duplicates(subset=[0, 1, 2, 3, 4, 6])

    ## Aggregate over subelements of a single annotation
    df_cnt_snv = df_inter_snv.groupby([13, 5]).size().reset_index(name='OBS_SNV')
    df_cnt_ind = df_inter_ind.groupby([13, 5]).size().reset_index(name='OBS_INDEL')
    df_cnt = df_cnt_snv.merge(df_cnt_ind, how='outer')
    df_cnt.loc[df_cnt.OBS_SNV.isna(), 'OBS_SNV'] = 0
    df_cnt.loc[df_cnt.OBS_INDEL.isna(), 'OBS_INDEL'] = 0
    df_cnt['OBS_MUT'] = df_cnt.OBS_SNV + df_cnt.OBS_INDEL
    df_cnt = df_cnt[[13, 5, 'OBS_SNV', 'OBS_INDEL', 'OBS_MUT']]
    df_cnt.columns = ['ELT', 'SAMPLE', 'OBS_SNV', 'OBS_INDEL', 'OBS_MUT']
    # df_cnt = df_inter.groupby([13, 5]).size().reset_index(name='COUNT')
    # df_cnt.columns = ['ELT', 'SAMPLE', 'OBS_SNV']

    return df_cnt


def tabulate_nonc_mutations_at_sites(f_sites, f_mut, return_sites = False):
    """ tabulate mutations occuring at given sites.
        * sites file is assumed to be annotated with corresponding gene
        ** This function no longer filters by cds regions using -use_subelts flag, instead this
            is to be carried out by the user with the f_sites file prior to pretraining
    """
    df_sites = read_mutation_file(f_sites)
    df_sites.rename({"SAMPLE": "ELT"}, axis=1, inplace=True)
    assert ('GENE' in df_sites.columns and 'ANNOT' in df_sites.columns and 'MUT_TYPE' in df_sites.columns)
    if not 'STRAND' in df_sites.columns:
        print("WARNING: strand column not detected in sites file. Defaulting all sites to + strand.")
        df_sites['STRAND'] = "+"

    df_mut = read_mutation_file(f_mut, drop_duplicates = False)
    assert ('GENE' in df_mut.columns and 'ANNOT' in df_mut.columns and'MUT_TYPE' in df_mut.columns)

    if len(df_mut[df_mut.ANNOT == 'INDEL']):
        print('WARNING: INDELS found in mutation file. Dig sites model is only applicable to SNVs. INDELS will be dropped.')

    df_mut = df_mut[df_mut.ANNOT != 'INDEL']

    muts_atsites = df_mut.merge(df_sites, on =['CHROM', 'START', 'END', 'REF', 'ALT', 'GENE', 'ANNOT','MUT_TYPE', 'CONTEXT'], how='inner')
    # print(muts_atsites[0:5])
    ## use sites annotations as gene
    # muts_atsites = muts_atsites.drop(columns=['GENE']).rename(columns={'SAMPLE_x':'SAMPLE', 'SAMPLE_y':'GENE'})

    counts = muts_atsites.groupby(['ELT']).agg(
        {'SAMPLE': lambda x: len(set(x)), 'CHROM': lambda x: len(x)}).reset_index()
    counts.columns = ['ELT', 'OBS_SAMPLES', "OBS_SNV"]

    # ## Add ELTs with zero observed mutatios
    # ALL_SITES = set(df_sites.ELT.unique())
    # ZERO_SITES = ALL_SITES - set(counts.ELT.unique())
    # counts_zero = pd.DataFrame({'ELT': list(ZERO_SITES), 'OBS_SAMPLES': [0]*len(ZERO_SITES), 'OBS_MUT': [0]*len(ZERO_SITES)})
    # counts_aug = pd.concat([counts, counts_zero]).sort_values(by='ELT')
    counts_aug = counts

    if return_sites:
        #trim unecessary columns and set index for fast lookup
        df_sites = df_sites.drop(columns = ['GENE','ANNOT','REF','ALT']).rename(columns={'SAMPLE':'GENE'}).set_index('GENE')
        return counts_aug, df_sites
    else:
        return counts_aug

#wrapper for transfer learning model on sites
def tabulate_sites_in_element(f_sites, f_mut):
    df_res = tabulate_nonc_mutations_at_sites(f_sites, f_mut)
    df_res = df_res.set_index('ELT')
    return df_res[['OBS_SAMPLES', 'OBS_SNV']]

def _genic_fill_empty_cols(df_gene_counts):
    """
    ensures that every mutation type is tabulated
    """
    needed_cols = set(['Essential_Splice', 'Missense','Nonsense','Stop_loss','Synonymous'])
    cur_cols  = set(df_gene_counts.columns)
    dif_cols = needed_cols.difference(cur_cols)
    for c in dif_cols:
        df_gene_counts[c] = 0

def filter_hypermut_samples(df_mut, max_muts_per_sample, return_blacklist=False):
    """ Remove samples that have more mutations than the specified threshold
    """
    sample_cnt = df_mut.SAMPLE.value_counts()
    samples_blacklist = sample_cnt[sample_cnt > max_muts_per_sample].index.to_list()

    df_whitelist = df_mut[~df_mut.SAMPLE.isin(samples_blacklist)]

    if return_blacklist:
        return df_whitelist, samples_blacklist

    return df_whitelist

def filter_samples_by_stdev(df_mut, stdev_cutoff):
    """ Remove samples where # mutations > stdev_cutoff*stdev of cohort mutation counts
    """
    sample_cnt = df_mut.SAMPLE.value_counts()
    stdev = sample_cnt.std()
    print(stdev)
    samples_blacklist = sample_cnt[sample_cnt > stdev * stdev_cutoff].index.to_list()

    df_whitelist = df_mut[~df_mut.SAMPLE.isin(samples_blacklist)]

    return df_whitelist

def cap_muts_per_element_per_sample(df_mut_elt_samp, max_muts_per_elt_per_sample):
    """ Upper-bound number of mutations any one sample can contribute to an element

    Args:
        df_mut_elt_samp     dataframe produced by tabulate_muts_per_sample_per_element
    """
    mask = df_mut_elt_samp.OBS_MUT > max_muts_per_elt_per_sample
    df_mut_elt_samp.loc[mask, 'OBS_MUT'] = max_muts_per_elt_per_sample

    return df_mut_elt_samp

def mutations_per_gene(df_mut_cds, max_muts_per_gene_per_sample=3e9):
    """ Count the number of mutations per gene in a mutation dataframe
    """
    # df_counts = pd.crosstab(df_mut_cds.GENE, df_mut_cds.ANNOT)
    df_group = df_mut_cds.groupby(['GENE', 'SAMPLE', 'ANNOT']).size().reset_index(name='COUNT')
    df_group.loc[df_group.COUNT > max_muts_per_gene_per_sample, 'COUNT'] = max_muts_per_gene_per_sample
    df_pivot = df_group.pivot_table(index=['GENE', 'ANNOT'], values='COUNT', aggfunc=np.sum).reset_index()
    df_counts = df_pivot.pivot(index='GENE', columns='ANNOT', values='COUNT')
    df_counts[df_counts.isna()] = 0
    df_counts = df_counts.astype(int)
    df_counts.columns = df_counts.columns.to_list()

    if not "Missense" in df_counts.columns:
        df_counts['Missense'] = 0
    if not "Nonsense" in df_counts.columns:
        df_counts['Nonsense'] = 0
    if not "Synonymous" in df_counts.columns:
        df_counts['Synonymous'] = 0
    if not "Essential_Splice" in df_counts.columns:
        df_counts['Essential_Splice'] = 0
    if not "INDEL" in df_counts.columns:
        df_counts['INDEL'] = 0

    df_counts.rename({'Missense': 'OBS_MIS',
                      'Nonsense': 'OBS_NONS',
                      'Synonymous': 'OBS_SYN',
                      'Essential_Splice': 'OBS_SPL',
                      'INDEL': 'OBS_INDEL'
                     },
        axis=1, inplace=True
    )

    return df_counts

def mutations_by_element(f_mut, f_elt_bed, bed12=False, drop_duplicates=False):
    """ Get all mutations falling within a set of elements
    """
    bed_mut = pybedtools.BedTool(f_mut)
    bed_elt = pybedtools.BedTool(f_elt_bed)
    if bed12:
        bed_elt = bed_elt.bed12tobed6()

    ## Intersect bedtools objects
    bed_inter = bed_mut.intersect(bed_elt, wa=True, wb=True)
    df_hits = bed_inter.to_dataframe(header=None, names=range(18), low_memory=False)
    df_hits = df_hits.iloc[:, [0,1,2,3,4,5,6,7,8,9,13]]

    if drop_duplicates:
        df_hits = df_hits.drop_duplicates([0, 1, 2, 3, 4, 5, 13]) ## Drop mutations duplicated by genic annotations

    df_hits.columns = ['CHROM', 'START', 'END', 'REF', 'ALT', 'SAMPLE', 'GENE', 'ANNOT', 'TYPE', 'CONTEXT', 'ELT']

    return df_hits

def bed12_boundaries(f_bed):
    df_nonc = pd.read_table(f_bed, names=['CHROM', 'START', 'END', "ELT", "SCORE", "STRAND", 'thickStart', 'thickEnd', 'rgb', 'blockCount', 'blockSizes', 'blockStarts'], low_memory=False)
    df_nonc.CHROM = df_nonc.CHROM.astype(str)
    df_nonc.blockSizes = df_nonc.blockSizes.astype(str)
    df_nonc.blockStarts = df_nonc.blockStarts.astype(str)
    if 'chr' in str(df_nonc['CHROM'][0]):
        df_nonc['CHROM'] = df_nonc['CHROM'].map(lambda x: x.lstrip('chr'))
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

    return df_nonc
