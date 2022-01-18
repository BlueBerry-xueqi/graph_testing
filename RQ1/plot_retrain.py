import os
import pdb

import matplotlib.pyplot as plt
import numpy as np

from parser import Parser

args = Parser().parse()
root_train = "pretrained_all/train_accuracy"
root_mse = "misclassed_ratio"

# all datasets
ratios = []
for i in range(1, 21):
    ratios.append(i * 5)
datasets = ["PubMed"]
models = ["AGNN"]

metrics = ["entropy", "margin", "l_con", "deepgini", "variance", "l_p", "m_c", "random", "mcp"]


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
            accs_mcp = []
            accs_m_c = []
            accs_l_p = []
            accs_variance = []

            for metricNo, metric in enumerate(metrics):
                for ratioNo, ratio in enumerate(ratios):
                    if ratio == 100 and metric == "mcp":
                        accs_mcp.append(1)
                    else:
                        retrain_file_path = f"{root_mse}/{model}_{dataset}/{metric}/mis{ratio}.npy"
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
                        if metric == "mcp":
                            accs_mcp.append(acc_retrain)
                        if metric == "m_c":
                            accs_m_c.append(acc_retrain)
                        if metric == "l_p":
                            accs_l_p.append(acc_retrain)
                        if metric == "variance":
                            accs_variance.append(acc_retrain)
            plt.figure(dpi=180)
            plt.style.use('seaborn-whitegrid')

            plt.plot(ratios, accs_DeepGini, marker="o", markersize=3)
            plt.plot(ratios, accs_max, marker="o", markersize=3)
            plt.plot(ratios, accs_margin, marker="o", markersize=3)
            plt.plot(ratios, accs_entropy, marker="o", markersize=3)
            plt.plot(ratios, accs_l_p, marker="o", markersize=3)
            plt.plot(ratios, accs_m_c, marker="o", markersize=3)
            plt.plot(ratios, accs_mcp, marker="o", markersize=3)
            plt.plot(ratios, accs_variance, marker="o", markersize=3)
            plt.plot(ratios, accs_random, marker="o", markersize=3, color='blue')

            plt.legend(["DeepGini", "Least Confidence", "Margin", "Entropy", "LC-variant I",
                        "LC-variant II", "MCP", "Variance", "Random"], frameon=True, prop = {'size':7}, loc = 4)

            plt.xlabel('Percentage of test data executed')
            plt.ylabel('Percentage of fault detected')

            plt.xlim((0, 105))

            plt.title(f"{model} - {dataset}")
            plt.show()

        plt.savefig(f"{savedpath_plot}/{model}_{dataset}.pdf")


print("program end!")

# run the program
plotGraph()
