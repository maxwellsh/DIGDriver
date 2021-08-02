#!/usr/bin/env python

import pandas as pd
import pkg_resources
import pathlib
import os
import argparse

from DIGDriver.data_tools import mutation_tools

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Filter hypermutated samples.')
    parser.add_argument('--suffix', default='annot.txt', help='suffix of Dig mutation files to filer')
    parser.add_argument('--max-muts-per-sample', default=3000, type=int, help='Maximum number of coding mutations allowed per sample. Samples with more coding mutations will be filtered.')
    args = parser.parse_args()

    if not os.path.isdir("filter_hypermut"):
        os.mkdir("filter_hypermut")

    paths = sorted(pathlib.Path('.').glob('*'+args.suffix))

    for f in paths:
        df = mutation_tools.read_mutation_file(str(f), drop_duplicates=True)
        df_mut = df[df.GENE != '.']
        _, sample_blacklist = mutation_tools.filter_hypermut_samples(df_mut, 
            max_muts_per_sample=3000, 
            return_blacklist=True
        )
        df_out = df[~df.SAMPLE.isin(sample_blacklist)]
        print(f.name, df.shape, df_out.shape)
        f_out = os.path.join("filter_hypermut", f.name.split('.annot.txt')[0] + ".no_hypermut.annot.txt")
        df_out.to_csv(f_out, header=False, index=False, sep="\t")
