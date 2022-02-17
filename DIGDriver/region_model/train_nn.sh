#!/bin/bash

## This is an **example** command to fit neural network predictions for 10kb regions of the PCAWG pan-cancer cohort.
## 
## NOTE: THE PROCESS REQUIRES INPUT DATA TOO LARGE TO BE INCLUDED IN THIS GITHUB REPO.
##       CONTACT THE AUTHORS TO ENSURE YOU HAVE THE NECESSARY INPUT FILES AND COMPUTE RESROUCES
##       IF YOU WANT TO CREATE MUTATION RATE MAPS FROM YOUR OWN WGS DATASETS.

python mutations_main.py -c Pancan_SNV
