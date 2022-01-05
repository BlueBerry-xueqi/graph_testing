#!/bin/bash -l
#SBATCH -J benign
#SBATCH -N 1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=xueqi.dang@uni.lu
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH -C skylake
conda activate graph
 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 5 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 10 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 15 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 20 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 25 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 30 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 35 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 40 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 45 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 50 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 55 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 60 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 65 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 70 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 75 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 80 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 85 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 90 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type ARMA --data PubMed --metrics entropy --select_ratio 95 --exp 3 --retrain_epochs 20 

python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 5 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 10 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 15 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 20 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 25 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 30 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 35 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 40 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 45 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 50 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 55 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 60 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 65 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 70 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 75 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 80 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 85 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 90 --exp 3 --retrain_epochs 20 
python selection_PubMed.py --type AGNN --data PubMed --metrics entropy --select_ratio 95 --exp 3 --retrain_epochs 20 

