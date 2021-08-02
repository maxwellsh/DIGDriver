#!/usr/bin/env python

import os
import sys
import h5py

if len(sys.argv) <= 1:
    raise Exception('Expected at least 1 input argument but found {}'.format(len(sys.argv) - 1))

zipped_file_path = sys.argv[1]

print('Opening zipped h5 file...')
zipped_h5f = h5py.File(zipped_file_path, 'r')

split_path = zipped_file_path.split('/')
unzipped_file_path = os.path.join('/'.join(split_path[:-1]), 'unzipped_{}'.format(split_path[-1]))
unzipped_h5f = h5py.File(unzipped_file_path, 'w')

print('Loading unzipped data to {}...'.format(unzipped_file_path))
for k in zipped_h5f.keys():
    print('Unzipping {}'.format(k))
    unzipped_h5f[k] = zipped_h5f[k][:]

print('Done!')
