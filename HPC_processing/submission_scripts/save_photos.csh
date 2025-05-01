#! /bin/bash
#BSUB -J gargamel
#BSUB -o stdout.%J
#BSUB -e stderr.%J
#BSUB -W 120
#BSUB -n 1
#BSUB -q ccee

source ~/.bashrc
conda activate /rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/conda/image_processing
python save_photos.py
conda deactivate
