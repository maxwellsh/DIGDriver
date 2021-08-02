import sys
import os
import argparse
import numpy as np 
import pandas as pd
from pathlib import Path
from multiprocessing.pool import Pool
import h5py

sys.path.append('../data_tools/')
sys.path.append('../region_model/')
sys.path.append('../sequence_model/')
import DataExtractor
import kfold_mutations_main
import SequenceModel
import GenicDriver


def parse_args(text=None):
    parser = argparse.ArgumentParser(description="Automation tool for running DIG pipeline")
    subparsers = parser.add_subparsers(help='DIG sub-commands')

    parser_a = subparsers.add_parser('runDIG', help='Run DIG model')

    ## required
    parser_a.add_argument('--out-dir', type=str, dest='out_dir', required = True, help='Base Directory of DIG run. All intermediate files will be saved relative to this location')
    parser_a.add_argument('--map-ref', type=str, dest='map_ref', help='path to mappability file')
    parser_a.add_argument('--window-size', type=int, default=10000, dest='window', help='desired window size for DIG model regions')
    parser_a.add_argument('--min-map', type=float, default=0.50, dest='min_mapp', help='minimum mappability for windows')
    parser_a.add_argument('--ref-file', type=str, dest='ref_file', help='path to reference hg19 genome')
    parser_a.add_argument('--mut-file', type=str, dest='mut_file', required = True, help='path to mutations file')
    parser_a.add_argument('--N-procs', type = int, dest='n_procs', default = 20, help= 'number of processes to run')

    ## partial runs
    parser_a.add_argument('--map-file', type = str, dest = 'map_file', help = 'map to precomputed mappability file')
    parser_a.add_argument('--epi-dir', type=str, dest='epi_dir', help='path to epigenomics files')
    parser_a.add_argument('--split_idx', type=str, dest='split_dir', help='path to split index dir')
    parser_a.add_argument('--epi-matrix_dir', type=str, dest='epi_matrix_dir', help='path to constructed epigenome matrix h5 file')
    parser_a.add_argument('--fmodel-dir', type=str, dest='fmodel_dir', help='path to constructed genome context frequency file')
    parser_a.add_argument('--gp-results-base', type=str, dest='gp_res', help='path to generic file name of gp results fold')

    ##optional arguments
    parser_a.add_argument('-c', '--cancer-key', type = str, dest = 'cancer_key', help = 'key name for cancer targets')
    parser_a.add_argument('-g', "--gpus", required=False, nargs='?', action='store', type=str, dest='gpus',
                    default='all', help='GPUs devices (all/comma separted list)')

    parser_a.set_defaults(func=run)
    
    if text:
        args = parser.parse_args(text.split())
    else:
        args = parser.parse_args()

    return args

# inputs are epi-genome tracks and mutation file



def run(args):
    if args.gp_res is None:
        if args.epi_matrix_dir is None:
            if args.epi_dir is None:
                print('Error: need to provide either a epi_track dir or a epi_matrix_dir')
                return
            else:
                map_file_name = "high_mapp_{}_{}_{}".format(args.min_mapp, args.window, 0)
                mapp_file_path = os.path.join(args.out_dir, map_file_name)
                if args.map_file is None:
                    print('Finding mappable windows...')
                    mapp_args = DataExtractor.parse_args('mappability {} --out-dir {} --window {} --overlap {} --min-map {}'.format(args.map_ref, args.out_dir, args.window, 0, args.min_mapp))
                    DataExtractor.mappability(mapp_args)
                    print('map file saved at: ' + mapp_file_path)

                print('creating split index...')

                if args.split_dir is None:
                    split_path = os.path.join(args.out_dir, 'splitIdx_{}'.format(args.window))
                    if not os.path.exists(split_path):
                        os.mkdir(split_path)
                    split_args = DataExtractor.parse_args('splitDataIdx --base-dir {} --out-dir {} --chunk-size {} --window {} --overlap {} --min-map {}'.format(args.out_dir, split_path, 10000, args.window, 0, args.min_mapp))
                    DataExtractor.split_data_idx(split_args)
                    print('splitIdx files saved at'+ split_path)
                else:
                    split_path = args.split_dir

                print('creating matrix chunks...')
                chunks_path = os.path.join(args.out_dir, 'matrix_chunks_{}'.format(args.window))
                print(chunks_path)
                if not os.path.exists(chunks_path):
                    os.mkdir(chunks_path)
                p = Pool(args.n_procs)
                path = Path(split_path).glob('**/*')
                files = [str(x) for x in path if x.is_file()]
                res = []
                for f in files:
                    res.append(p.apply_async(chunk_runner, (f, chunks_path, args.ref_file, args.epi_dir, args.mut_file, args.window, args.cancer_key)))
                p.close()
                p.join()
                _ = [r.get() for r in res]
                print('chunks saved')

                print('concatenating chunks...')
                concat_args = DataExtractor.parse_args('concatH5 {} --out-dir {}'.format(chunks_path, args.out_dir))
                DataExtractor.concatH5(concat_args)

                print('adding mappability track')
                epi_matrix_fname = os.path.join(args.out_dir, 'data_matrices' + '_{}_0_{}'.format(args.window, args.min_mapp) + '.h5')
                addMap_args = DataExtractor.parse_args('addMappability {} {}'.format(epi_matrix_fname, args.map_ref))
                DataExtractor.add_mappability(addMap_args)
                print('epi track done!')
        else:
            print('running NN model')
            epi_matrix_fname = args.epi_matrix_dir
        
        kfold_args = kfold_mutations_main.get_cmd_arguments('-c {} -d {} -o {} -m {} -g {}'.format(args.cancer_key, epi_matrix_fname, args.out_dir, args.min_mapp, args.gpus))
        kfold_mutations_main.main(kfold_args)
        print('finished NN model')
        directory = os.path.join(args.out_dir, 'kfold/{}'.format(args.cancer_key))
        date_dir = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
        gp_results_base = os.path.join(date_dir, 'gp_results_fold_{}.h5')
    else:
        gp_results_base = args.gp_res
        mapp_file_path = args.map_file
    # we assume that you either dont have anything, have the genome counts but not the mutation counts (or annotations) or have everything
    if args.fmodel_dir is None:
        f_model_path = os.path.join(args.out_dir, 'fmodel_{}_trinuc_192.h5'.format(args.window))
        genome_context_args = SequenceModel.parse_args('countGenomeContext {} {} {} {} --up {} --down {} --n-procs {}'.format(mapp_file_path, args.window, args.ref_file, f_model_path, 1, 1, args.n_procs))
        SequenceModel.countGenomeContext(genome_context_args)
    else:
        f_model_path = args.fmodel_dir

    fmodel = h5py.File(f_model_path, 'r')
    if args.cancer_key + '_mutation_counts' in fmodel.keys():
        run_canc = False
    else:
        run_canc = True
    fmodel.close()

    if run_canc:
        annot_name = os.path.basename(args.mut_file).split('txt.gz')[0] + 'trinuc.txt'
        annot_path = os.path.join(args.out_dir, annot_name)
        print(annot_path)
        annot_args = SequenceModel.parse_args('annotateMutationFile {} {} {} {} --n-procs {}'.format(args.mut_file, f_model_path, args.ref_file, annot_path, args.n_procs))
        SequenceModel.annotateMutationFile(annot_args)
        annot_path = annot_path + '.gz'
        
        count_contexts_args = SequenceModel.parse_args('countMutationContext {} {} {} {} {} --n-procs {} '.format(mapp_file_path, annot_path, f_model_path, args.window, args.cancer_key, args.n_procs))
        SequenceModel.countMutationContext(count_contexts_args)
    else:
        annot_path = args.mut_file
    
    #run models
    print('running models')
    submap_path = gp_results_base.split('gp_results')[0] + 'sub_mapp_results_fold_{}.h5'

#    for fold in range(5):
#        apply_seq_args = SequenceModel.parse_args('applySequenceModel {} {} {} {} {} --cancer {} --key-prefix {} --key {} --n-procs {} --bins {} --run ensemble'.format(gp_results_base.format(fold), f_model_path, annot_path, args.ref_file, args.window, args.cancer_key, args.cancer_key, args.cancer_key, args.n_procs, 50))
#        SequenceModel.applySequenceModel(apply_seq_args)

    results_path = os.path.join(args.out_dir, 'results')
    if not os.path.exists(results_path):
        os.mkdir(results_path)

#    concat_sequence_results(gp_results_base, args.cancer_key, os.path.join(results_path, 'hotspot_results_{}.h5'.format(args.cancer_key)))
    genic_out = os.path.join(results_path, 'genicDetect_{}_{}_{}.h5'.format(args.cancer_key, args.window, args.min_mapp))

    genic_args = GenicDriver.parse_args('genicDetectParallel {} {} {} {} -c {} -N {} -m {} -u {}'.format(annot_path, gp_results_base, f_model_path, genic_out, args.cancer_key, args.n_procs, args.min_mapp, submap_path))

    GenicDriver.genicDetectParallel(genic_args)

    nonc_out = os.path.join(results_path, 'noncDetect_{}_{}_{}.h5'.format(args.cancer_key, args.window, args.min_mapp))
    nonc_args = GenicDriver.parse_args('noncDetectParallel {} {} {} {} -c {} -N {} -m {} -u {} -t both'.format(annot_path, gp_results_base, f_model_path, nonc_out, args.cancer_key, args.n_procs, args.min_mapp, submap_path))
    GenicDriver.noncodingDetectParallel(nonc_args)
    
def main():
    args = parse_args()
    args.func(args)
    print('Done!')

def chunk_runner(f, chunks_path, ref_file, epi_dir, mut_file, window, cancer_key):
    chunk_args = DataExtractor.parse_args('createChunk {} --out-dir {} --ref-file {} --epi-dir {} --mut-file {} --window {} --bins {} --cancer-key {}'.format(f, chunks_path, ref_file, epi_dir, mut_file, window, 100, cancer_key))
    DataExtractor.create_chunk(chunk_args)

def concat_sequence_results(base_results, cancer, out_path):
    fout = h5py.File(out_path, 'a')
    #keys = [k for k in f[cancer]['test'].keys() if 'nb_model' in k]
    keys = ['nb_model_up1_down1_binsize50_run_ensemble']
    if len(keys) ==0:
        return -1
    for k in keys:
        print('working on {}'.format(k))
        df_lst = []
        for run in range(5):
            run_res = pd.read_hdf(base_results.format(run), key='{}/test/{}'.format(cancer, k))
            run_res = run_res.astype({'CHROM': 'int32', 'POS': 'float64', 'OBS': 'int32', 'EXP': 'float64','PVAL': 'float64','Pi': 'float64','MU': 'float64','SIGMA': 'float64','REGION': 'object'})
            df_lst.append(run_res)
        complete = pd.concat(df_lst)
        complete.to_hdf(out_path, key = k, format='fixed')
    fout.close()

if __name__ == '__main__':
    main()

    
