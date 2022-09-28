#!/bin/bash
#SBATCH --mem=120GB
#SBATCH --ntasks=8
#SBATCH --time=UNLIMITED
##SBATCH --mail-user=USER@uni-muenster.de
##SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1

singularity exec \
  --bind /data/tod:/data/ant-ml \
  --bind /data/tod-res:/data/ant-ml-res \
  /data/sifs/torch.sif python train_recurrent.py "$@"
