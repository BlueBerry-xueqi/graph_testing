import pdb

import matplotlib.pyplot as plt
import numpy as np

from parser import Parser

args = Parser().parse()
root_train = "pretrained_all/train_accuracy"
root_retrain = "retrained_all/train_accuracy"

# all datasets
ratios = [1, 3, 5, 10, 15]
models = ["gin", "gat", "gmt", "gcn"]
datasets = ["DD", "PTC_MR"]
metrics = ["DeepGini", "random", "max_probability"]
ratios = [1, 3, 5, 10, 15]


# # sample
# ratios = [1, 3, 5, 10, 15]
# models = ["gin"]
# datasets = ["DD"]
# metrics = ["DeepGini", "random"]


def plotGraph():
    for dataset in datasets:
        for model in models:
            accs_train = []
            accs_DeepGini = []
            accs_random = []
            accs_max = []

            for metricNo, metric in enumerate(metrics):
                for ratioNo, ratio in enumerate(ratios):
                    # train process
                    acc_train = np.load(f"{root_train}/{model}_{dataset}/test_accuracy.npy")
                    accs_train.append(acc_train)

                    # define the list to get the avg
                    list_DeepGini = []
                    list_random = []
                    list_max = []
                    for exp in range(0, 3):
                        retrain_file_path = f"{root_retrain}/{metric}/{model}_{dataset}/{exp}/test_accuracy_ratio{ratio}.npy"
                        # get accuracy
                        acc_retrain = np.load(retrain_file_path)
                        # print("model: "+str(model)+"  |dataset: "+str(dataset)+"  |exp: "+str(exp)+"  |metrics: "+str(
                        # metric)+"  |ratio: "+str(ratio)+"  |acc: "+str(acc_retrain)+"\n") add accuracy
                        if metric == "DeepGini":
                            list_DeepGini.append(acc_retrain)
                        if metric == "random":
                            list_random.append(acc_retrain)
                        if metric == "max_probability":
                            list_max.append(acc_retrain)

                    accs_DeepGini_avg = sum(list_DeepGini) / 3
                    accs_random_avg = sum(list_random) / 3
                    accs_max_avg = sum(list_max) / 3
                    if accs_DeepGini_avg != 0:
                        accs_DeepGini.append(accs_DeepGini_avg)
                    if accs_random_avg != 0:
                        accs_random.append(accs_random_avg)
                    if accs_max_avg != 0:
                        accs_max.append(accs_max_avg)

            acc_train = acc_train[0]
            accs_train = [acc_train, acc_train, acc_train, acc_train, acc_train]
            # draw graph
            plt.plot(ratios, accs_random, marker='o')
            plt.plot(ratios, accs_DeepGini, marker='*')
            plt.plot(ratios, accs_max, marker='+')
            plt.plot(ratios, accs_train, marker='^')

            plt.legend(["DeepGini", "random", "max_probability", "original"])
            plt.savefig(f"acc_graph/{model}_{dataset}.pdf")
            plt.close()
            print("program end!")


# run the program
plotGraph()
