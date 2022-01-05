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
for i in range(1, 10):
    ratios.append(i * 5)
datasets = ["CiteSeer"]
#  CiteSeer
# Tox21_AhR_training, NCI109 Mutagenicity
models = ["AGNN"]
# GraphNN

metrics = ["random","entropy", "margin", "deepgini", "l_con", "GMM", "Hierarchical", "kmeans", "MCP"]




def plotGraph():
    for dataset in datasets:
        savedpath_plot = f"acc_graph/{dataset}/"
        if not os.path.isdir(savedpath_plot):
            os.makedirs(savedpath_plot, exist_ok=True)
        for model in models:
            # acc_train = np.load(f"{root_train}/{model}_{dataset}/test_accuracy.npy")

            accs_DeepGini = []
            accs_random = []
            accs_max = []
            accs_margin = []
            accs_entropy = []

            accs_GMM = []
            accs_Hierarchical = []
            accs_kmeans = []
            accs_MCP = []
            accs_spec = []
            accs_variance = []


            for metricNo, metric in enumerate(metrics):
                for ratioNo, ratio in enumerate(ratios):
                    retrain_file_path = f"{root_retrain}/{model}_{dataset}/{metric}/mse{ratio}.npy"
                    if not ratio == 0:

                        acc_retrain = np.load(retrain_file_path)
                        if metric == "deepgini":
                            accs_DeepGini.append(acc_retrain)
                        if metric == "random":
                            accs_random.append(acc_retrain)
                        if metric == "l_con":
                            accs_max.append(acc_retrain)
                        if metric == "margin":
                            accs_margin.append(acc_retrain)
                        if metric == "entropy":
                            accs_entropy.append(acc_retrain)

                        if metric == "GMM":
                            accs_GMM.append(acc_retrain)
                        if metric == "Hierarchical":
                            accs_Hierarchical.append(acc_retrain)
                        if metric == "kmeans":
                            accs_kmeans.append(acc_retrain)
                        if metric == "MCP":
                            accs_MCP.append(acc_retrain)
                        if metric == "spec":
                            accs_spec.append(acc_retrain)
                        if metric == "variance":
                            accs_variance.append(acc_retrain)

            plt.plot(ratios, accs_random, marker='o')
            plt.plot(ratios, accs_DeepGini, marker='o')
            plt.plot(ratios, accs_max, marker='o')
            plt.plot(ratios, accs_margin, marker='o')
            plt.plot(ratios, accs_entropy, marker='o')

            plt.plot(ratios, accs_GMM, marker='o')
            plt.plot(ratios, accs_Hierarchical, marker='o')
            plt.plot(ratios, accs_kmeans, marker='o')
            plt.plot(ratios, accs_MCP, marker='o')
            # plt.plot(ratios, accs_spec, marker='o')
            # plt.plot(ratios, accs_variance, marker='o')
            plt.legend(["random", "DeepGini", "least confidence", "margin", "entropy", "GMM", "Hierarchical", "K-Means", "MCP"])

            # # # # # # test
            # plt.plot(ratios, accs_max, marker='o')
            # plt.plot(ratios, accs_DeepGini, marker='*')
            # plt.plot(ratios, accs_entropy, marker='*')
            # plt.legend(["random", "deepgini", "entropy", "baseline"])

            plt.title(f"{model}_{dataset}")
            plt.show()

        plt.savefig(f"{savedpath_plot}/{model}_{dataset}.pdf")


print("program end!")

# run the program
plotGraph()
