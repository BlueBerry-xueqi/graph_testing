'''
sa.py is from Surprised Adequacy Paper open source.
'''

import numpy as np
import time
import os
from multiprocessing import Pool
from tqdm import tqdm



import numpy as np
def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]

        


def _get_saved_path(base_path, dataset, dtype, layer_names):
    """Determine saved path of ats and pred
    Args:
        base_path (str): Base save path.
        dataset (str): Name of dataset.
        dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
        layer_names (list): List of layer names.
    Returns:
        ats_path: File path of ats.
        pred_path: File path of pred (independent of layers)
    """

    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + dtype + "_pred" + ".npy"),
         os.path.join(base_path, dataset + "_groundtruth_" + dtype + "_pred" + ".npy")
    )


def get_ats(
        model,
        dataset_loader,
        name,
        layer_names,
        save_path=None,
        batch_size=128,
        is_classification=True,
        num_classes=10,
        num_proc=10,
        device="cpu"
):
    """Extract activation traces of dataset from model.
    Args:
        model (keras model): Subject model.
        dataset (list): Set of inputs fed into the model.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        is_classification (bool): Task type, True if classification task or False.
        num_classes (int): The number of classes (labels) in the dataset.
        num_proc (int): The number of processes for multiprocessing.
    Returns:
        ats (list): List of (layers, inputs, neuron outputs).
        pred (list): List of predicted classes.
        ground_truth (list)
    """

    pred, ground_truth, extracted_layers_outputs = model.extract_intermediate_outputs(dataset_loader)
    dataset_len = len(dataset_loader.dataset)
    prefix = info("[" + name + "] ")
    if is_classification:
        p = Pool(num_proc)
        print(prefix + "Model serving")
        print(prefix + "Processing ATs")
        ats = None
        for (layer_name, layer_output) in extracted_layers_outputs.items():
            print("Layer: " + layer_name)
            if layer_output[0].ndim == 3:
                # For convolutional layers
                layer_matrix = np.array(
                    p.map(_aggr_output, [layer_output[i] for i in range( dataset_len )])
                )
            else:
                layer_matrix = np.array(layer_output)

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None

    if save_path is not None:
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)
        np.save(save_path[2], ground_truth)

    return ats, pred, ground_truth


def find_closest_at(at, train_ats):
    """The closest distance between subject AT and training ATs.
    Args:
        at (list): List of activation traces of an input.
        train_ats (list): List of activation traces in training set (filtered)
    Returns:
        dist (int): The closest distance.
        at (list): Training activation trace that has the closest distance.
    """

    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), train_ats[np.argmin(dist)])

def _get_train_target_ats(model, traindataset_loader, testdata_loader, target_name, layer_names, **kwargs):
    """Extract ats of train and target inputs. If there are saved files, then skip it.
    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: keyboard args.
    Returns:
        train_ats (list): ats of train set.
        train_pred (list): pred of train set.
        target_ats (list): ats of target set.
        target_pred (list): pred of target set.
        ground_truth_tagrget (list): ground truth
    """

    saved_train_path = _get_saved_path(kwargs["save_path"], kwargs["d"], "train", layer_names)
    if os.path.exists(saved_train_path[0]) and os.path.exists(saved_train_path[1]) and os.path.exists(saved_train_path[2]):
       print(infog("Found saved {} ATs, skip serving".format("train")))
      #  In case train_ats is stored in a disk
       train_ats = np.load(saved_train_path[0])
       train_pred = np.load(saved_train_path[1])
    else:
        train_ats, train_pred, _  = get_ats(
            model,
            traindataset_loader,
            "train",
            layer_names,
            num_classes=kwargs['num_classes'],
            is_classification=kwargs['is_classification'],
            save_path=saved_train_path,
        )
        if saved_train_path is not None:
            print(infog("train ATs is saved at " + saved_train_path[0]))

    saved_target_path = _get_saved_path(
       kwargs["save_path"], kwargs["d"], target_name, layer_names
    )
    if os.path.exists(saved_target_path[0]) and os.path.exists(saved_target_path[1]) and os.path.exists(saved_target_path[2]) :
       print(infog("Found saved {} ATs, skip serving").format(target_name))
       # In case target_ats is stored in a disk
       target_ats = np.load(saved_target_path[0])
       target_pred = np.load(saved_target_path[1])
       ground_truth_tagrget =  np.load(saved_target_path[2])
    else:
        target_ats,target_pred, ground_truth_tagrget = get_ats(
            model,
            testdata_loader,
            target_name,
            layer_names,
            num_classes=kwargs['num_classes'],
            is_classification=kwargs['is_classification'],
            save_path=saved_target_path
        )
        print(infog(target_name + " ATs is saved at " + saved_target_path[0]))

    return train_ats, train_pred, target_ats, target_pred, ground_truth_tagrget



def fetch_sihoutete(model, x_train_loader, x_target_loader, target_name, layer_names, **kwargs):
    """Distance-based SA
    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: keyboard args.
    Returns:
        dsa (list): List of dsa for each target input.
    """

    #assert args.is_classification == True

    prefix = info("[" + target_name + "] ")
    train_ats, train_pred, target_ats, target_pred, ground_truth_tagrget = _get_train_target_ats(
        model, x_train_loader, x_target_loader, target_name, layer_names, **kwargs
    )

    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)

    si = []

    print(prefix + "Fetching sihoutete")
    for i, at in enumerate(tqdm(target_ats)):
        label = target_pred[i]
        a_dist = compute_avg(at, train_ats[class_matrix[label]])
        b_dist = find_min_at(
            at, train_ats, class_matrix,label
        )
        si.append((b_dist-a_dist) /np.max([a_dist, b_dist]) )
    del target_ats, train_ats, train_pred
    return si, target_pred,ground_truth_tagrget

def find_min_at(at, train_ats, class_matrix,label):
    vmin = None
    for key, value in class_matrix.items():
        if key == label:
            continue
        avgd =  compute_avg(at, train_ats[class_matrix[key]])
        if vmin == None:
            vmin = avgd
        if avgd < vmin:
            vmin = avgd
    return vmin



def compute_avg(at, train_ats):
    """The avg distance between subject AT and training ATs.
       Args:
           at (list): List of activation traces of an input.
           train_ats (list): List of activation traces in training set (filtered)
       Returns:
           dist (int): The closest distance.
           at (list): Training activation trace that has the closest distance.
       """

    dist = np.linalg.norm(at - train_ats, axis=1)
    return np.mean(dist)

def _get_kdes(train_ats, train_pred, class_matrix, **kwargs):
    """Kernel density estimation
    Args:
        train_ats (list): List of activation traces in training set.
        train_pred (list): List of prediction of train set.
        class_matrix (list): List of index of classes.
        args: Keyboard args.
    Returns:
        kdes (list): List of kdes per label if classification task.
        removed_cols (list): List of removed columns by variance threshold.
    """

    removed_cols = []
    if kwargs['is_classification']:
        for label in range(kwargs['num_classes']):
                col_vectors = np.transpose(train_ats[class_matrix[label]])
                for i in range(col_vectors.shape[0]):
                    if (
                            np.var(col_vectors[i]) < kwargs['var_threshold']
                            and i not in removed_cols
                    ):
                        removed_cols.append(i)

        kdes = {}
        for label in tqdm(range(kwargs['num_classes']), desc="kde"):
                refined_ats = np.transpose(train_ats[class_matrix[label]])
                refined_ats = np.delete(refined_ats, removed_cols, axis=0)
                #print("Org Shape {}, delete {}".format(s, refined_ats.shape))
                if refined_ats.shape[0] == 0:
                    print(
                        warn("ats were removed by threshold {}".format(kwargs['var_threshold']))
                    )
                    break
                #print(refined_ats.shape)
                kdes[label] = refined_gaussian_kde(refined_ats)

    else:
         col_vectors = np.transpose(train_ats)
         for i in range(col_vectors.shape[0]):
             if np.var(col_vectors[i]) < kwargs['var_threshold']:
                 removed_cols.append(i)

         refined_ats = np.transpose(train_ats)
         refined_ats = np.delete(refined_ats, removed_cols, axis=0)
         if refined_ats.shape[0] == 0:
             print(warn("ats were removed by threshold {}".format(kwargs['var_threshold'])))
         kdes = [refined_gaussian_kde(refined_ats)]

    print(infog("The number of removed columns: {}".format(len(removed_cols))))

    return kdes, removed_cols



def get_sc(lower, upper, k, sa):
    """Surprise Coverage
    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
        k (int): The number of buckets.
        sa (list): List of lsa or dsa.
    Returns:
        cov (int): Surprise coverage.
    """

    buckets = np.digitize(sa, np.linspace(lower, upper, k))
    return len(list(set(buckets))) / float(k) * 100

import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def infog(msg):
    return Colors.OKGREEN + msg + Colors.ENDC


def info(msg):
    return Colors.OKBLUE + msg + Colors.ENDC


def warn(msg):
    return Colors.WARNING + msg + Colors.ENDC


def fail(msg):
    return Colors.FAIL + msg + Colors.ENDC

