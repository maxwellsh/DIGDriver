import os, sys
import pickle as pkl
import numpy as np
import pandas as pd
import argparse

def get_cmd_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--track-list', required=True, nargs='?', action='store', type=str, dest='track_lst_path',help= 'path to list of tracks being used')
    ap.add_argument('-o', '--out-dir', nargs='?', default = './', action='store', type=str, dest='out_dir',help= 'path to save track selection file')
    ap.add_argument('-stemcells',  action = 'store_true', help='Include Stem cells [ESC, ESC_derived, IPSC, Placental]')
    ap.add_argument('-general', action = 'store_true', help='Include general tracks [fibroblasts, stromal cells, adipose tissue]')
    ap.add_argument('-other', action = 'store_true', help='Include all other misc tracks')
    ap.add_argument('-lung', action = 'store_true', help='Include lung tracks')
    ap.add_argument('-breast', action = 'store_true', help='Include breast tracks')
    ap.add_argument('-blood', action = 'store_true', help='Include blood tracks')
    ap.add_argument('-skin', action = 'store_true', help='Include skin tracks')
    ap.add_argument('-liver', action = 'store_true', help='Include liver tracks')
    ap.add_argument('-stomach', action = 'store_true', help='Include stomach tracks (limited selection)')
    ap.add_argument('-GC', action = 'store_true', help='Include GC content track')
    ap.add_argument('-HiC', action = 'store_true', help='Include all HiC tracks')
    ap.add_argument('-repli_chip', action = 'store_true', help='Include all repli-seq tracks')
    ap.add_argument('-cons', action = 'store_true', help='Include conservation tracks (included in general)')
    ap.add_argument('-seq', action = 'store_true', help='Include sequence context tracks (included in general)')
    return ap.parse_args()

def main():
    args = get_cmd_arguments()
    meta = pd.read_csv(open('/scratch1/priebeo/neurIPS/new_tracks_meta.csv', 'r'))
    track_lst = pkl.load(open(args.track_lst_path, 'rb'))
    track_lst = np.array([t.split('/')[-1].split('.')[0] for t in track_lst])

    meta['track_pos'] = -1
    for i, l in enumerate(track_lst):
        meta.loc[meta['File accession'] == l, 'track_pos'] = i
    meta = meta.astype({'track_pos': int})
    meta = meta.set_index('track_pos')
    meta.sort_index(inplace = True)

    track_accumulator = set([])
    if args.stemcells:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['Anatomy'] == 'StemCells'].index)))
    if args.general:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['Anatomy'] == 'General'].index)))
    if args.other:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['Anatomy'] == 'Other'].index)))
    if args.lung:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['Anatomy'] == 'Lung'].index)))
    if args.breast:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['Anatomy'] == 'Breast'].index)))
    if args.blood:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['Anatomy'] == 'Blood'].index)))
    if args.skin:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['Anatomy'] == 'Skin'].index)))
    if args.liver:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['Anatomy'] == 'Liver'].index)))
    if args.stomach:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['Anatomy'] == 'Stomach'].index)))
    
    if args.GC:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['File accession'] == 'GC_content'].index)))
    if args.seq:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['File accession'] == 'hg19'].index)))
    if args.HiC:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['File accession'] == 'GC_content'].index)))
    if args.cons:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['Assay'] == 'conservation'].index)))
    if args.repli_chip:
        track_accumulator = track_accumulator.union(set(np.array(meta.loc[meta['Assay'] == 'Repli-chip'].index)))

    to_add = np.array(sorted(track_accumulator))
    if len(to_add[to_add < 0]) > 0:
        print('Some desired file not present in track list')
    to_add = to_add[to_add >= 0]
    print('adding {} tracks'.format(to_add.shape[0]))
    out_dir = os.path.join(args.out_dir, 'track_selection.txt')
    np.savetxt(out_dir, to_add, fmt='%i')

    print('Done!')

if __name__ == "__main__":
    main()
