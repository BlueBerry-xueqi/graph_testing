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
 
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 10 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 15 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 20 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 25 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 30 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 35 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 40 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 45 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 50 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 55 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 60 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 65 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 70 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 75 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 80 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 85 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 90 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 95 --mse_epochs 10
python selection_Cora.py --type AGNN --data Cora --metrics mini --select_num 100 --mse_epochs 10

python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 10 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 15 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 20 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 25 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 30 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 35 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 40 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 45 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 50 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 55 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 60 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 65 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 70 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 75 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 80 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 85 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 90 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 95 --mse_epochs 10
python selection_Cora.py --type GCN --data Cora --metrics mini --select_num 100 --mse_epochs 10

python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 10 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 15 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 20 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 25 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 30 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 35 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 40 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 45 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 50 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 55 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 60 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 65 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 70 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 75 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 80 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 85 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 90 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 95 --mse_epochs 10
python selection_Cora.py --type GAT --data Cora --metrics mini --select_num 100 --mse_epochs 10

python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 10 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 15 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 20 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 25 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 30 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 35 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 40 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 45 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 50 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 55 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 60 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 65 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 70 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 75 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 80 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 85 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 90 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 95 --mse_epochs 10
python selection_Cora.py --type ARMA --data Cora --metrics mini --select_num 100 --mse_epochs 10

