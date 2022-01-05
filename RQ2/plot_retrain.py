import os
import pdb

import matplotlib.pyplot as plt
import numpy as np

from parser import Parser

args = Parser().parse()
root_train = "pretrained_all/train_accuracy"
root_retrain = "MSE"

# all datasets
ratios = []
for i in range(2, 20):
    ratios.append(i * 5)
datasets = ["NCI1"]
#  CiteSeer
# Tox21_AhR_training, NCI109 Mutagenicity
models = ["GraphNN"]
# GraphNN

# metrics = ["random", "GMM", "Hierarchical", "kmeans"]
metrics = ["random", "kmeans"]


def plotGraph():
    for dataset in datasets:
        savedpath_plot = f"acc_graph/{dataset}/"
        if not os.path.isdir(savedpath_plot):
            os.makedirs(savedpath_plot, exist_ok=True)
        for model in models:
            # acc_train = np.load(f"{root_train}/{model}_{dataset}/test_accuracy.npy")

            accs_random = []
            accs_GMM = []
            accs_Hierarchical = []
            accs_kmeans = []
            accs_spec = []

            for metricNo, metric in enumerate(metrics):
                for ratioNo, ratio in enumerate(ratios):
                    retrain_file_path = f"{root_retrain}/{model}_{dataset}/{metric}/mse{ratio}.npy"
                    if not ratio == 0:

                        acc_retrain = np.load(retrain_file_path)
                        if metric == "random":
                            accs_random.append(acc_retrain)
                        if metric == "GMM":
                            accs_GMM.append(acc_retrain)
                        if metric == "Hierarchical":
                            accs_Hierarchical.append(acc_retrain)
                        if metric == "kmeans":
                            accs_kmeans.append(acc_retrain)
                        if metric == "spec":
                            accs_spec.append(acc_retrain)

            plt.plot(ratios, accs_random, marker='o')
            # plt.plot(ratios, accs_GMM, marker='o')
            # plt.plot(ratios, accs_Hierarchical, marker='o')
            plt.plot(ratios, accs_kmeans, marker='o')
            # plt.plot(ratios, accs_spec, marker='o')
            plt.legend(["random", "K-Means"])

            plt.title(f"{model}_{dataset}")
            plt.show()

        plt.savefig(f"{savedpath_plot}/{model}_{dataset}.pdf")


print("program end!")

# run the program
plotGraph()
