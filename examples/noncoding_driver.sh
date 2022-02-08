#!/bin/bash

MODEL="Pancan_SNV_MNV_INDEL.Pretrained.h5"
MUTS="Pancan_SNV_MNV_INDEL.ICGC.annot.txt.gz"

## Annotation to be analyzed. Comment-in the desired annotation.
REGION="grch37.PCAWG_noncoding.bed"
NAME="PCAWG_all_elts"

# REGION="grch37.canonical_5utr_with_splice.bed"
# NAME="utr5_w_splice"

# REGION="grch37.TP53_5UTR_exon1.bed"
# NAME="TP53_5UTR"

## Check that DigDriver.py is in path
[[ $(type -P "DigDriver.py") ]]  || 
    { echo "DigDriver.py is NOT in PATH. Please Ensure Dig is installed." 1>&2; exit 1; }

## Download files as necessary
[[ ! -f "$MODEL" ]] && { echo -e "Downloading $MODEL\n"; wget -nv --show-progress "http://cb.csail.mit.edu/cb/DIG/downloads/mutation_maps/$MODEL"; echo -e "\n"; }

[[ ! -f "$MUTS" ]] && { echo -e "Downloading $MUTS\n"; wget -nv --show-progress "http://cb.csail.mit.edu/cb/DIG/downloads/mutation_files/PCAWG/ICGC_only/$MUTS"; echo -e "\n"; }

[[ ! -f "$REGION" ]] && { echo -e "Downloading $REGION\n"; wget -nv --show-progress "http://cb.csail.mit.edu/cb/DIG/downloads/annotions/noncoding/$REGION"; echo -e "\n"; }

## Run DigDriver
echo -e "Running DigDriver.py...\n"
DigDriver.py elementDriver \
    Pancan_SNV_MNV_INDEL.ICGC.annot.txt.gz \
    Pancan_SNV_MNV_INDEL.Pretrained.h5 \
    $NAME \
    --f-bed $REGION \
    --outpfx Pancan_SNV_MNV_INDEL.$NAME \
    --outdir .
