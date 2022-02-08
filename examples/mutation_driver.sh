#!/bin/bash

MODEL="Pancan_SNV_MNV_INDEL.Pretrained.h5"
MUTS="Pancan_SNV_MNV_INDEL.ICGC.annot.txt.gz"

## Annotation to be analyzed. Comment-in the desired annotation.
DRIVERS="grch37.spliceAI_CRYPTIC.noncoding.txt.gz"
NAME="spliceAI_cryptic_noncoding"

# DRIVERS="grch37.spliceAI_CRYPTIC.txt.gz"
# NAME="spliceAI_cryptic_all"

# DRIVERS="grch37.spliceAI_CANONICAL.txt.gz"
# NAME="spliceAI_canonical"

# DRIVERS="grch37.spliceAI_CRYPTIC.coding.txt.gz"
# NAME="spliceAI_cryptic_coding"

## Check that DigDriver.py is in path
[[ $(type -P "DigDriver.py") ]]  || 
    { echo "DigDriver.py is NOT in PATH. Please Ensure Dig is installed." 1>&2; exit 1; }

## Download files as necessary
[[ ! -f "$MODEL" ]] && { echo -e "Downloading $MODEL\n"; wget -nv --show-progress "http://cb.csail.mit.edu/cb/DIG/downloads/mutation_maps/$MODEL"; echo -e "\n"; }

[[ ! -f "$MUTS" ]] && { echo -e "Downloading $MUTS\n"; wget -nv --show-progress "http://cb.csail.mit.edu/cb/DIG/downloads/mutation_files/PCAWG/ICGC_only/$MUTS"; echo -e "\n"; }

[[ ! -f "$DRIVERS" ]] && { echo -e "Downloading $DRIVERS\n"; wget -nv --show-progress "http://cb.csail.mit.edu/cb/DIG/downloads/annotions/splicing/$DRIVERS"; echo -e "\n"; }

## Run DigDriver
echo -e "Running DigDriver.py...\n"
DigDriver.py elementDriver \
    Pancan_SNV_MNV_INDEL.ICGC.annot.txt.gz \
    Pancan_SNV_MNV_INDEL.Pretrained.h5 \
    $NAME \
    --f-sites $DRIVERS \
    --outpfx Pancan_SNV_MNV_INDEL.$NAME \
    --outdir .
