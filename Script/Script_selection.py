import os

# dataset
datasets_sample = ["DD", "PTC_MR", "DHFR", "MCF-7H", "MOLT-4", "PTC_MM"]

datasets_large = ["MCF-7", "MOLT-4", "NCI-H23", "OVCAR-8", "P388", "PC-3", "SF-295", "SW-620H", "UACC257",
               "COIL-DEL", "COIL-RAG", "DBLP_v1", "github_stargazers", "REDDIT-MULTI-12K", "COLORS-3", "TRIANGLES"]

# datasets that have small size, from different types
datasets_small = ["BZR", "COX2", "DHFR", "MUTAG",
                  "PTC_FM","PTC_MM", "Tox21_AhR_testing", "Tox21_AR-LBD_testing", "Tox21_aromatase_testing",
                  "KKI", "OHSU", "Peking_1",
                  "Cuneiform", "MSRC_9","MSRC_21",
                  "highschool_ct1", "infectious_ct1", "tumblr_ct1",
                  "SYNTHETIC", "SYNTHETICnew", "Synthie"]

datasets_test = ["MCF-7"]

datasets_mul_classes = ["ENZYMES", "Fingerprint", "Letter-low", "MSRC_9", "MSRC_21C", "REDDIT-MULTI-12K", "REDDIT-MULTI-5K"]


# models, metrics and ratios
models = ["gin", "gat"]
metrics = ["margin", "random", "DeepGini", "max_probability", "Entropy"]
# metrics = ["random", "DeepGini"]
ratios = [0.1, 0.15, 0.2, 0.25]


# root
root = f"batchScript/selection/dataset_test/"
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
    filename = f'retrain.sh'
    with open(root + filename, 'w') as f:
        f.write(pre)
        for metric in metrics:
                for dataset in datasets:
                    for model in models:
                        for ratio in ratios:
                            f.write("python selection_graph_classification.py "
                                    f"--type {model} "
                                    f"--data {dataset} "
                                    f"--metrics {metric} "
                                    f"--select_ratio {ratio} "
                                    "--exp 3 "
                                    "--retrain_epochs 2 \n")
                        f.write("\n")


# writeFile(models, datasets_small, metrics, ratios)
# writeFile(models, datasets_large, metrics, ratios)
# writeFile(models, datasets_small, metrics, ratios)
writeFile(models, datasets_mul_classes, metrics, ratios)
print("end")
