import numpy as np
import sklearn.metrics
import torch

def f1_score(actual, predicted, average='macro'):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return sklearn.metrics.f1_score(actual, predicted, average=average)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    ret = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret
