from fastai.learner import Metric
from fastai.torch_core import to_detach
import torch
import torch.nn.functional as F
import numpy as np


class MSEMetric(Metric):
    def __init__(self): self.preds = []

    def accumulate(self, learn):
        cats, conts, mu, logvar = learn.pred
        cat_targs, cont_targs = learn.y
        norm_conts = conts.new([conts.size(1)])
        self.preds.append(to_detach(F.mse_loss(conts, cont_targs, reduction='sum') / norm_conts))

    @property
    def value(self):
        return torch.Tensor(self.preds).mean()


class CEMetric(Metric):
    def __init__(self, total_cats):
        self.preds = []
        self.total_cats = total_cats

    def accumulate(self, learn):
        cats, conts, mu, logvar = learn.pred
        cat_targs, cont_targs = learn.y
        CE = cats.new([0])
        pos = 0
        for i, (k, v) in enumerate(self.total_cats.items()):
            CE += F.cross_entropy(cats[:, pos:pos + v], cat_targs[:, i], reduction='sum')
            pos += v

        norm = cats.new([len(self.total_cats.keys())])
        self.preds.append(to_detach(CE / norm))

    @property
    def value(self):
        return torch.Tensor(self.preds).mean()


class KLDMetric(Metric):
    def __init__(self): self.preds = []

    def accumulate(self, learn):
        cats, conts, mu, logvar = learn.pred
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.preds.append(to_detach(KLD))

    @property
    def value(self):
        return torch.Tensor(self.preds).mean()


class MUMetric(Metric):
    def __init__(self): self.preds = []

    def accumulate(self, learn):
        cats, conts, mu, logvar = learn.pred
        self.preds.append(to_detach(mu.mean()))

    @property
    def value(self):
        return torch.Tensor(self.preds).mean()


class StdMetric(Metric):
    def __init__(self): self.preds = []

    def accumulate(self, learn):
        cats, conts, mu, logvar = learn.pred
        self.preds.append(to_detach((logvar.exp_() ** .5).mean()))

    @property
    def value(self):
        return torch.Tensor(self.preds).mean()


def mean_absolute_relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
