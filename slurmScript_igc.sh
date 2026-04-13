#!/bin/bash

###########################
# Set up submit to ...
#SBATCH -J igc_network
#SBATCH -t 0
#SBATCH --ntasks-per-node=48
#SBATCH -N 1
###########################

cd /mnt/speicher/.wissmit/oliver/Data/python/IGC_model/network_module
python network_module.py