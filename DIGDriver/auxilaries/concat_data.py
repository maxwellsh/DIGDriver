import os
import sys
import json
import h5py
import numpy as np
import pickle as pkl



def main():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    if len(sys.argv) < 2:
        print('No input was given. Dataset directory must be given.')
    else:
        dir_path = sys.argv[1]

    idxs_relative_path = '../data_indices'
    data_lst = sorted([f for f in os.listdir(dir_path)])
    idx_lst = sorted([f for f in os.listdir(os.path.join(dir_path, idxs_relative_path))])
    data_arr = []
    idx_arr = []

    print('Loading all data and index files...')
    for data_file, idx_file  in zip(data_lst, idx_lst):
        hf = h5py.File(os.path.join(dir_path, data_file), 'r')
        data_arr.append(hf['x_data'][:])  # returns a numpy array as long as the dataset's ID is 'x_data'
        with open(os.path.join(dir_path, idxs_relative_path, idx_file), 'rb') as f:
            idx_arr.append(pkl.load(f))

    print('Saving indices file to ./all_indices.pkl...')
    with open('all_indices.pkl', 'wb') as f:
        pkl.dump(idx_arr, f)

    print('Saving data file to ./all_data.pkl...')
    h5f = h5py.File('all_data.h5', 'w')
    h5f.create_dataset('x_data', data=np.concatenate(data_arr))
    h5f.close()



if __name__ == '__main__':
    main()
