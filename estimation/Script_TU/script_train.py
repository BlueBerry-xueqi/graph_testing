import os

model = [ "GraphNN"]
dataset = ["NCI1", "Mutagenicity"]


# root
root = f"batchScript/"
if not os.path.isdir(root):
    os.makedirs(root, exist_ok=True)


def writeFile(models, datasets, filename):
    pre = "#!/bin/bash -l\n" \
          "#SBATCH -J benign\n" \
          "#SBATCH -N 1\n" \
          "#SBATCH --mail-type=end,fail\n" \
          "#SBATCH --mail-type=end,fail\n" \
          "#SBATCH --mail-user=xueqi.dang@uni.lu\n" \
          "#SBATCH -n 2\n" \
          "#SBATCH -p gpu\n" \
          "#SBATCH --gres=gpu:2\n" \
          "#SBATCH --time=0-12:00:00\n" \
          "#SBATCH -C skylake\n" \
          "conda activate graph\n \n"
    filename = f'{filename}_train.sh'
    with open(root + filename, 'w') as f:
        f.write(pre)
        for dataset in datasets:
            for model in models:
                f.write("python trainer_TU.py "
                        f"--type {model} "
                        f"--data {dataset} "
                        f"--epochs 30 \n")
            f.write("\n")


writeFile(model, dataset, 'PubMed')


print("end")
