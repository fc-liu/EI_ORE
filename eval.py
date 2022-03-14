"""
This code is adjusted from https://github.com/ttthy/ure/blob/master/ure/scorer.py
"""

from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score
import torch
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize


def bcubed_correctness(gold, pred, na_id=-1):
    # remove NA
    gp = [(x, y) for x, y in zip(gold, pred) if x != na_id]
    gold = [x for x, _ in gp]
    pred = [y for _, y in gp]

    # compute 'correctness'
    l = len(pred)
    assert(len(gold) == l)
    gold = torch.IntTensor(gold)
    pred = torch.IntTensor(pred)
    gc = ((gold.unsqueeze(0) - gold.unsqueeze(1)) == 0).int()
    pc = ((pred.unsqueeze(0) - pred.unsqueeze(1)) == 0).int()
    c = gc * pc
    return c, gc, pc


def bcubed_precision(c, gc, pc):
    pcsum = pc.sum(1)
    total = torch.where(pcsum > 0, pcsum.float(), torch.ones(pcsum.shape))
    return ((c.sum(1).float() / total).sum() / gc.shape[0]).item()


def bcubed_recall(c, gc, pc):
    gcsum = gc.sum(1)
    total = torch.where(gcsum > 0, gcsum.float(), torch.ones(gcsum.shape))
    return ((c.sum(1).float() / total).sum() / pc.shape[0]).item()


def bcubed_score(gold, pred, na_id=-1):
    c, gc, pc = bcubed_correctness(gold, pred, na_id)
    prec = bcubed_precision(c, gc, pc)
    rec = bcubed_recall(c, gc, pc)
    return prec, rec, 2 * (prec * rec) / (prec + rec)


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


def v_measure(gold, pred):
    homo = homogeneity_score(gold, pred)
    comp = completeness_score(gold, pred)
    v_m = v_measure_score(gold, pred)
    return homo, comp, v_m


def check_with_bcubed_lib(gold, pred):
    import bcubed
    ldict = dict([('item{}'.format(i), set([k])) for i, k in enumerate(gold)])
    cdict = dict([('item{}'.format(i), set([k])) for i, k in enumerate(pred)])

    precision = bcubed.precision(cdict, ldict)
    recall = bcubed.recall(cdict, ldict)
    fscore = bcubed.fscore(precision, recall)

    print('P={} R={} F1={}'.format(precision, recall, fscore))


if __name__ == '__main__':
    gold = [0, 0, 0, 0, 0, 1, 1, 2, 1, 3, 4, 1, 1, 1]
    pred = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]

    print(bcubed_sparse(gold, pred), 'should be 0.69')
    print(bcubed_score(gold, pred, na_id=1000), 'should be 0.69')
    # check_with_bcubed_lib(gold, pred)
    homo = homogeneity_score(gold, pred)
    v_m = v_measure_score(gold, pred)
    ari = adjusted_rand_score(gold, pred)
    print(homo, v_m, ari)
