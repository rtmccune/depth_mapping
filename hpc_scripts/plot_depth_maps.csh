#! /bin/bash
#BSUB -J gargamel
#BSUB -o plot_out.%J
#BSUB -e plot_err.%J
#BSUB -W 60
#BSUB -n 50
#BSUB -q ccee

source ~/.bashrc

module load PrgEnv-intel
conda activate /rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/conda/image_processing
mpirun python plot_flood_event_maps.py

conda deactivate

