from copy import deepcopy
import numpy as np
from sklearn import metrics
from utils import MovingAverage, to_numpy
from scipy.special import softmax
import torch

def stable_log1pex(x):
    return -np.minimum(x, 0) + np.log(1+np.exp(-np.abs(x)))

def cal_llloss_with_logits(label, logits):
    ll = -np.mean(label*(-stable_log1pex(logits)) + (1-label)*(-logits - stable_log1pex(logits)))
    return ll

def prob_clip(x):
    return np.clip(x, a_min=1e-5, a_max=1-1e-5)

def cal_llloss_with_prob(label, prob):
    ll = -np.mean(label*np.log(prob_clip(prob)) + (1-label)*(np.log(prob_clip(1-prob))))
    return ll

def cal_auc(label, pos_prob):
    fpr, tpr, thresholds = metrics.roc_curve(label, pos_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def cal_prauc(label, pos_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(label, pos_prob)
    area = metrics.auc(recall, precision)
    return area

def cal_acc(label, prob):
    label = np.reshape(label, (-1,))
    prob = np.reshape(label, (-1,))
    prob_acc = np.mean(label*prob)
    return prob_acc

def stable_softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)

def cal_metric(pred, target, pos_label=1):
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    target = target.reshape((-1))
    _pred = {}
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            _v = v.detach().cpu().numpy()
        else:
            _v = v
        _pred[k] = _v
    pred = _pred
    res = {
        "acc": None,
        "auc": None,
        "prauc": None,
        "ll": None
    }
    if len(target) == 0:
        return res
    if "prob" not in pred:
        assert "logits" in pred
        pred["prob"] = softmax(pred["logits"], axis=1)
        
    prob = to_numpy(pred["prob"])
    _y = np.argmax(prob, axis=1)
    target = to_numpy(target)
    acc = np.mean((_y == target).astype(np.float32))
    auc = cal_auc(target, prob[:,pos_label])
    prauc = cal_prauc(target, prob[:,pos_label])
    ll = cal_llloss_with_prob(target, prob[:,pos_label])
    res["acc"] = acc
    res["auc"] = auc
    res["prauc"] = prauc
    res["ll"] = ll
    res["pos_target"] = (target == pos_label).astype(np.int64)
    res["pos_prob"] = prob[:,pos_label]
    return res

def cal_calibration(pos_target, pos_prob, bin_num=10):
    bins = [(i+1)/bin_num for i in range(bin_num)]
    bins[-1] += 1e-5
    bins.insert(0, -1e-5)
    errs = []
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i+1]
        mid = (lower + upper) / 2
        bin_mask = np.logical_and(pos_prob >= lower, pos_prob < upper)
        if np.sum(bin_mask) == 0:
            continue
        acc = np.mean(pos_target[bin_mask] == 1)
        err = np.abs(acc - mid)
        errs.append(err)
    ece = np.mean(errs)
    mce = np.max(errs)
    return {
        "mce": mce,
        "ece": ece
    }

class MetricAccumulator:
    
    def __init__(self, calibration_bin_num=10):
        self.value = None
        self.array = None
        self.calibration_bin_num = calibration_bin_num
    
    def add(self, batch_metric, batch_size=1):
        if batch_metric is None:
            return
        if self.value is None:
            self.value = {}
            self.array = {}
            for k, v in batch_metric.items():
                if np.isscalar(v):
                    self.value[k] = MovingAverage()
                    self.value[k].add(v, batch_size)
                else:
                    self.array[k] = []
                    self.array[k].append(v)
        else:
            for k, v in batch_metric.items():
                if np.isscalar(v):
                    self.value[k].add(v, batch_size)
                else:
                    self.array[k].append(v)
    
    def reset(self):
        if self.value is None:
            return
        else:
            self.value = None
            self.array = None
            
    def cal_calibration(self):
        if "pos_target" not in self.array:
            return {"ece": None, "mce": None}
        pos_target = np.concatenate(self.array["pos_target"], axis=0)
        pos_prob = np.concatenate(self.array["pos_prob"], axis=0)
        res = cal_calibration(pos_target, pos_prob, self.calibration_bin_num)
        mce, ece = res["mce"], res["ece"]
        self.value["mce"] = MovingAverage(mce)
        self.value["ece"] = MovingAverage(ece)
        return res
                
    def __str__(self):
        res = "MetricAccumulator:\n"
        if self.value is None:
            return "Metric: No value"
        self.cal_calibration()
        for k, v in self.value.items():
            res += "{}: {}\n".format(k, v)
        res = res.strip()
        return res
    
def metric_mean_and_std(metric_list):
    print("calculating metric mean and std")
    if len(metric_list) == 0:
        return None
    elif len(metric_list) == 1:
        return metric_list[0]
    else:
        value_dict = {}
        for metric in metric_list:
            for k, v in metric.value.items():
                if k not in value_dict:
                    value_dict[k] = []
                value_dict[k].append(v.avg)
        for k, v in value_dict.items():
            mean = np.mean(v)
            std = np.std(v)
            print("{}: {} +- {}".format(k, mean, std))