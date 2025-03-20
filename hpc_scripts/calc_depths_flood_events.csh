#! /bin/bash
#BSUB -J gargamel
#BSUB -o rect_out.%J
#BSUB -e rect_err.%J
#BSUB -W 120
#BSUB -n 12
#BSUB -R span[hosts=1]
#BSUB -q gpu
#BSUB -gpu "num=1"

source ~/.bashrc

conda activate /rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/conda/image_processing
python gen_depth_maps_for_flood_events.py

conda deactivate

