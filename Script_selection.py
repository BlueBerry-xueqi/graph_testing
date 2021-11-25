import os

datasets_big = ["MCF-7", "MCF-7H", "MOLT-4", "MOLT-4H", "NCI-H23", "OVCAR-8",
                "OVCAR-8H", "P388", "PC-3", "PC-3H", "SN12CH",
                "reddit_threads"]
models = ["gin", "gat", "gmt", "gcn"]
metrics = ["DeepGini", "random", "max_probability"]
ratios = [1, 3, 5, 10, 15]


def root_path():
    root_big = f"batchScript/LargeDataset/retrain/"
    if not os.path.isdir(root_big):
        os.makedirs(root_big, exist_ok=True)
    return root_big


def writeFile(models, datasets, metrics, ratios):
    root = root_path()
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

    for metric in metrics:
        # DeepGini
        if metric == "DeepGini":
            filename = 'retrain_large_data_DeepGini.sh'
            with open(root + filename, 'w') as f:
                f.write(pre)

                for dataset in datasets:
                    for model in models:
                        for ratio in ratios:
                            f.write("python selection_graph_classification.py "
                                    f"--type {model} "
                                    f"--data {dataset} "
                                    f"--metrics {metric} "
                                    f"--select_ratio {ratio} "
                                    "--exp 3 "
                                    "--epochs 10 \n")
                        f.write("\n")
        # random
        elif metric == "random":
            filename = 'retrain_large_data_random.sh'
            with open(root + filename, 'w') as f:
                f.write(pre)
                for model in models:
                    for dataset in datasets:
                        for ratio in ratios:
                            f.write("python selection_graph_classification.py "
                                    f"--type {model} "
                                    f"--data {dataset} "
                                    f"--metrics {metric} "
                                    f"--select_ratio {ratio} "
                                    "--exp 3 "
                                    "--epochs 10 \n")
                        f.write("\n")

        elif metric == "max_probability":
            filename = 'retrain_large_data_max_probability.sh'
            with open(root + filename, 'w') as f:
                f.write(pre)
                for model in models:
                    for dataset in datasets:
                        for ratio in ratios:
                            f.write("python selection_graph_classification.py "
                                    f"--type {model} "
                                    f"--data {dataset} "
                                    f"--metrics {metric} "
                                    f"--select_ratio {ratio} "
                                    "--exp 3 "
                                    "--epochs 10 \n")
                        f.write("\n")


writeFile(models, datasets_big, metrics, ratios)
