#!/bin/bash

####################################################################
# SETUP (note that “#SBATCH” is a command - in all other cases “#” comments out):
####################################################################

# NAME your job as you wish:
#SBATCH --job-name=IGC.NRW

# Number of tasks 
#SBATCH --ntasks=1                 

# all CPUs of node for task
#SBATCH --cpus-per-task=48

# TIMELIMIT of your job (HH:MM:SS) – “0” for an infinite run:
#SBATCH --time=24:00:00 

# MEMORY per CPU specification in [MB] (i.e. max.Memory/max.Cores=128GB/96): (not recommended) 
##SBATCH --mem-per-cpu=1344

# DO NOT CHANGE:
#SBATCH -N 1

##SBATCH -o test-%j.out
####################################################################

pixi run python network.py
