import os

import matplotlib.pyplot as plt
import numpy as np

from parser import Parser

args = Parser().parse()
root_train = "pretrained_all/train_accuracy"
root_retrain = "retrained_all/train_accuracy"

# all datasets
ratios = []
for i in range(1, 10):
    ratios.append(i * 10)
datasets = ["MNIST"]
# Tox21_AhR_training, NCI109
models = ["gra"]

metrics = ["random", "deepgini", "entropy", "margin", "l_con"]
# metrics = ["random", "entropy", "deepgini"]


def plotGraph():
    for dataset in datasets:
        savedpath_plot = f"acc_graph/{dataset}/"
        if not os.path.isdir(savedpath_plot):
            os.makedirs(savedpath_plot, exist_ok=True)
        for model in models:

            accs_DeepGini = []
            accs_random = []
            accs_max = []
            accs_margin = []
            accs_entropy = []

            accs_BALD = []
            accs_K = []
            for metricNo, metric in enumerate(metrics):
                accs_train = []
                for ratioNo, ratio in enumerate(ratios):
                    # train process
                    acc_train = np.load(f"{root_train}/{model}_{dataset}/test_accuracy.npy")
                    print(acc_train)
                    accs_train.append(acc_train)

                    retrain_file_path = f"{root_retrain}/{model}_{dataset}/{metric}/test_accuracy_ratio{ratio}.npy"
                    # get accuracy
                    acc_retrain = np.load(retrain_file_path)
                    # print("model: "+str(model)+"  |dataset: "+str(dataset)+"  |exp: "+str(exp)+"  |metrics: "+str(
                    # metric)+"  |ratio: "+str(ratio)+"  |acc: "+str(acc_retrain)+"\n") add accuracy
                    # p3
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
                    if metric == "bald":
                        accs_BALD.append(acc_retrain)
                    if metric == "kmeans":
                        accs_K.append(acc_retrain)

            plt.plot(ratios, accs_random, marker='o')
            # plt.plot(ratios, accs_BALD, marker='*')
            # plt.plot(ratios, accs_K, marker='+')
            plt.plot(ratios, accs_DeepGini, marker='*')
            plt.plot(ratios, accs_max, marker='+')
            plt.plot(ratios, accs_margin, marker='o')
            plt.plot(ratios, accs_entropy, marker='*')
            plt.plot(ratios, accs_train, marker='^')
            plt.legend(["random",   "DeepGini","least confidence",  "margin", "entropy", "baseline"])

            # # # test
            # plt.plot(ratios, accs_random, marker='o')
            # plt.plot(ratios, accs_DeepGini, marker='*')
            # plt.plot(ratios, accs_entropy, marker='*')
            # plt.plot(ratios, accs_train, marker='^')
            # plt.legend(["random", "deepgini", "entropy", "baseline"])

            plt.title(f"{model}_{dataset}")
            plt.show()

        plt.savefig(f"{savedpath_plot}/{model}_{dataset}.pdf")


print("program end!")

# run the program
plotGraph()
