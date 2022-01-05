import os

model = ["AGNN", "GCN", "GAT", "ARMA"]
dataset = ["Cora"]
metrics = [ "kmeans", "spec", "Hierarchical", "GMM", "random", "mini", "kplus"]

ratios = []
for i in range(2, 21):
    ratios.append(i * 5)

# root
root = f"batchScript/"
if not os.path.isdir(root):
    os.makedirs(root, exist_ok=True)


def writeFile(models, datasets, metrics, select_nums):
    pre = "#!/bin/bash -l\n" \
          "#SBATCH -J benign\n" \
          "#SBATCH -N 1\n" \
          "#SBATCH --mail-type=end,fail\n" \
          "#SBATCH --mail-user=xueqi.dang@uni.lu\n" \
          "#SBATCH -n 2\n" \
          "#SBATCH -p gpu\n" \
          "#SBATCH --gres=gpu:2\n" \
          "#SBATCH --time=0-24:00:00\n" \
          "#SBATCH -C skylake\n" \
          "conda activate graph\n \n"

    # create saved directory
    savedpath = f"{root}/"
    if not os.path.isdir(savedpath):
        os.makedirs(savedpath, exist_ok=True)

    for dataset in datasets:
        for metric in metrics:
            filename = f'retrain-{dataset}-{metric}.sh'
            with open(savedpath + filename, 'w') as f:
                f.write(pre)
                for model in models:
                    for num in select_nums:
                        if metric == "random":
                            f.write("python selection_Cora.py "
                                    f"--type {model} "
                                    f"--data {dataset} "
                                    f"--metrics {metric} "
                                    f"--select_num {num} "
                                    f"--mse_epochs 100\n")
                        else:
                            f.write("python selection_Cora.py "
                                    f"--type {model} "
                                    f"--data {dataset} "
                                    f"--metrics {metric} "
                                    f"--select_num {num} "
                                    f"--mse_epochs 10\n")

                    f.write("\n")


writeFile(model, dataset, metrics, ratios)


print("end")