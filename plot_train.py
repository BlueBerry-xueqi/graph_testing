import numpy as np

# datasets big
from matplotlib import pyplot as plt

datasets = ["PROTEINS", "DD", "ENZYMES"]
models = ["TUGIN"]


root_pretrain = "pretrained_all/train_accuracy"


def get_train_accuracy(datasets, root, savedname):
    for model in models:
        accs_train = []
        for dataset in datasets:
            acc_train = np.load(f"{root}/{model}_{dataset}/test_accuracy.npy")
            accs_train.append(acc_train)
            print(f"model: {model} === dataset: {dataset} ===  acc: {acc_train}")

        # plt.bar(datasets, accs_train)
        # plt.xlabel('datasets')
        # plt.ylabel('accuracy')
        #
        # # set y-axis
        # my_y_ticks = np.arange(0, 1, 0.1)
        # plt.yticks(my_y_ticks)
        # plt.title("Accuracy of multiple classes datasets")
        #
        # plt.legend(models)
        # plt.show()
        # plt.savefig(f"acc_graph/{savedname}.pdf")


get_train_accuracy(datasets, root_pretrain, "Bioinformatics")
