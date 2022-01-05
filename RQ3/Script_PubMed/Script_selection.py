import os

# model = [ "GCN", "GAT"]
model = ["ARMA", "AGNN"]
dataset = ["PubMed"]

metrics = ["random", "deepgini", "entropy", "margin", "l_con"]
metrics_new = ["MCP", "variance", "kmeans", "spec", "Hierarchical", "GMM"]

ratios = []
for i in range(1, 20):
    ratios.append(i * 5)

# root
root = f"batchScript/"
if not os.path.isdir(root):
    os.makedirs(root, exist_ok=True)


def writeFile(models, datasets, metrics, ratios):
    pre = "#!/bin/bash -l\n" \
          "#SBATCH -J benign\n" \
          "#SBATCH -N 1\n" \
          "#SBATCH --mail-type=end,fail\n" \
          "#SBATCH --mail-user=xueqi.dang@uni.lu\n" \
          "#SBATCH -n 2\n" \
          "#SBATCH -p gpu\n" \
          "#SBATCH --gres=gpu:2\n" \
          "#SBATCH --time=0-12:00:00\n" \
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
                    for ratio in ratios:
                        f.write("python selection_PubMed.py "
                                f"--type {model} "
                                f"--data {dataset} "
                                f"--metrics {metric} "
                                f"--select_ratio {ratio} "
                                "--exp 3 "
                                "--retrain_epochs 20 \n")
                    f.write("\n")


writeFile(model, dataset, metrics, ratios)
writeFile(model, dataset, metrics_new, ratios)

print("end")