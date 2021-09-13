#!/bin/bash
#SBATCH --mail-type=ALL,ARRAY_TASKS 
#SBATCH --mail-user=lina.cristanchofajardo@inra.fr 

#export 
#echo $SLURM_ARRAY_TASK_ID
python3 /home/maiage/lcristancho/SIMULATIONS/Centralised_Code/Cluster_25epidemic_vacc50reps/Code_centralized.py
