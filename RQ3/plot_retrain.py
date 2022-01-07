import os
import pdb

import matplotlib.pyplot as plt
import numpy as np

from parser import Parser

args = Parser().parse()
root_train = "pretrained_all/train_accuracy"
root_retrain = "retrained_all/train_accuracy"

# all datasets
ratios = [0]
for i in range(1, 20):
    ratios.append(i * 5)
datasets = ["MNIST"]
#  CiteSeer
# Tox21_AhR_training, NCI109 Mutagenicity
models = ["NN"]
# GraphNN

metrics = ["random", "entropy", "margin", "deepgini", "l_con", "variance", "MCP", "GMM", "kmeans",
           "Hierarchical", "spec"]

# metrics = ["random", "entropy", "margin", "deepgini", "l_con", "variance", "GMM",
#            "Hierarchical"]
# metrics = ["random", "entropy", "margin", "deepgini", "l_con", "variance", "MCP", "GMM",
#            "Hierarchical"]
# metrics = ["random", "entropy", "margin", "deepgini", "l_con", "MCP", "GMM"]


def plotGraph():
    for dataset in datasets:
        savedpath_plot = f"acc_graph/{dataset}/"
        if not os.path.isdir(savedpath_plot):
            os.makedirs(savedpath_plot, exist_ok=True)
        for model in models:
            acc_train = np.load(f"{root_train}/{model}_{dataset}/test_accuracy.npy")
            print("train accuracy is: ", acc_train)

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
            accs_DeepGini.append(acc_train)
            accs_random.append(acc_train)
            accs_max.append(acc_train)
            accs_margin.append(acc_train)
            accs_entropy.append(acc_train)

            accs_GMM.append(acc_train)
            accs_Hierarchical.append(acc_train)
            accs_kmeans.append(acc_train)
            accs_MCP.append(acc_train)
            accs_spec.append(acc_train)
            accs_variance.append(acc_train)

            for metricNo, metric in enumerate(metrics):
                for ratioNo, ratio in enumerate(ratios):
                    retrain_file_path = f"{root_retrain}/{model}_{dataset}/{metric}/test_accuracy_ratio{ratio}.npy"
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

            plt.figure(dpi=180)
            plt.style.use('seaborn-whitegrid')


            # plot lines
            plt.plot(ratios, accs_GMM, marker="o", markersize=3)
            plt.plot(ratios, accs_Hierarchical, marker="o", markersize=3)
            plt.plot(ratios, accs_kmeans, marker="o", markersize=3)
            plt.plot(ratios, accs_spec, marker="o", markersize=3, color='blue')

            plt.plot(ratios, accs_DeepGini, marker="o", markersize=3)
            plt.plot(ratios, accs_max, marker="o", markersize=3)
            plt.plot(ratios, accs_margin, marker="o", markersize=3, color='black')
            plt.plot(ratios, accs_entropy, marker="o", markersize=3)
            plt.plot(ratios, accs_variance, marker="o", markersize=3)
            plt.plot(ratios, accs_MCP, marker="o", markersize=3)
            plt.plot(ratios, accs_random, marker="o", color='red', markersize=3)

            plt.legend(["GMM", "Hierarchical","Kmeans", "Spectrum", "DeepGini", "Least Confidence", "Margin", "Entropy",
                        "Variance", "MCP", "Random"], frameon=True)

            # plt.legend(["GMM", "Hierarchical", "DeepGini", "Least Confidence", "Margin", "Entropy",
            #             "Variance", "Random"], frameon=True)

            # set x and y axis name
            plt.xlabel('Percentage of retrained data selected')
            plt.ylabel('Test accuracy')
            plt.title(f"{model} - {dataset}")
            plt.show()

        plt.savefig(f"{savedpath_plot}/{model}_{dataset}.pdf")


print("program end!")

# run the program
plotGraph()
