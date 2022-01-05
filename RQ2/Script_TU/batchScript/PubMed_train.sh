#!/bin/bash -l
#SBATCH -J benign
#SBATCH -N 1
#SBATCH --mail-type=end,fail
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=xueqi.dang@uni.lu
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH -C skylake
conda activate graph
 
python trainer_TU.py --type GraphNN --data NCI1 --epochs 30 

python trainer_TU.py --type GraphNN --data Mutagenicity --epochs 30 

