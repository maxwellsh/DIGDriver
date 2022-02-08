#!/bin/bash

MODEL="Pancan_SNV_MNV_INDEL.Pretrained.h5"
MUTS="Pancan_SNV_MNV_INDEL.ICGC.annot.txt.gz"

## Check that DigDriver.py is in path
[[ $(type -P "DigDriver.py") ]]  || 
    { echo "DigDriver.py is NOT in PATH. Please Ensure Dig is installed." 1>&2; exit 1; }

## Download files as necessary
[[ ! -f "$MODEL" ]] && { echo -e "Downloading $MODEL\n"; wget -nv --show-progress "http://cb.csail.mit.edu/cb/DIG/downloads/mutation_maps/$MODEL"; echo -e "\n"; }

[[ ! -f "$MUTS" ]] && { echo -e "Downloading $MUTS\n"; wget -nv --show-progress "http://cb.csail.mit.edu/cb/DIG/downloads/mutation_files/PCAWG/ICGC_only/$MUTS"; echo -e "\n"; }

## Run DigDriver
echo -e "Running DigDriver.py...\n"
DigDriver.py geneDriver \
    Pancan_SNV_MNV_INDEL.ICGC.annot.txt.gz \
    Pancan_SNV_MNV_INDEL.Pretrained.h5 \
    --outdir . \
    --outpfx Pancan_SNV_MNV_INDEL.genes 
