import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from sklearn.metrics import classification_report
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import normalize


def contingency_matrix(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``."""
    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size
    # Following works because coo_matrix sums duplicate entries. Is roughly
    # twice as fast as np.histogram2d.
    cmatrix = coo_matrix(
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=np.int)
    # cmatrix = cmatrix.toarray()
    return cmatrix, ref_classes, sys_classes


def bcubed(ref_labels, sys_labels, cm=None):
    ref_labels = np.array(ref_labels)
    sys_labels = np.array(sys_labels)
    if cm is None:
        cm, _, _ = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()
    dividor = cm.sum(axis=0)
    row_norm = (cm / cm.sum(axis=0))
    precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))

    dividor = cm.sum(axis=1)
    col_norm = cm / np.expand_dims(cm.sum(axis=1), 1)
    recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
    f1 = 2*(precision*recall)/(precision + recall)
    return precision, recall, f1


def bcubed_sparse(ref_labels, sys_labels, cm=None):
    ref_labels = np.array(ref_labels)
    sys_labels = np.array(sys_labels)
    if cm is None:
        cm, _, _ = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()
    cm_col_norm = normalize(cm, norm='l1', axis=0)

    cm_row_norm = normalize(cm, norm="l1", axis=1)
    recall = np.sum(cm_norm.multiply(cm_row_norm))
    precision = np.sum(cm_norm.multiply(cm_col_norm))

    f1 = 2*(precision*recall)/(precision + recall)
    return precision, recall, f1


if __name__ == "__main__":
    import torch
    torch.manual_seed(1)
    gold = torch.randint(10, (920256,))
    pred = torch.randint(10, (920256,))
    # b3 = bcubed(gold, pred)
    b3 = bcubed_sparse(gold, pred)
    print(b3)
