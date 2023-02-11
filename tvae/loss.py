from fastai.callback.core import Callback
from fastai.losses import CrossEntropyLossFlat, MSELossFlat
from fastai.torch_core import Module
from fastcore.basics import store_attr
import torch


class VAERecreatedLoss(Module):
    "Measures how well we have created the original tabular inputs, plus the KL Divergence with the unit normal distribution"

    def __init__(self, cat_dict, dataset_size, bs, total_cats):
        super().__init__()
        ce = CrossEntropyLossFlat(reduction='sum')
        mse = MSELossFlat(reduction='sum')
        store_attr('cat_dict,ce,mse,dataset_size,bs,total_cats')

    def forward(self, preds, cat_targs, cont_targs):
        if (len(preds) == 5):
            cats, conts, mu, logvar, kl_weight = preds
        else:
            cats, conts, mu, logvar = preds
            kl_weight = 1

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        CE = cats.new([0])
        pos = 0
        for i, (k, v) in enumerate(self.total_cats.items()):
            CE += self.ce(cats[:, pos:pos + v], cat_targs[:, i])
            pos += v

        norm_cats = cats.new([len(self.total_cats.keys())])
        norm_conts = conts.new([conts.size(1)])
        total = (self.mse(conts, cont_targs) / norm_conts) + (CE / norm_cats)

        # This factor depends on your batch size and the size of the dataset.  A good rule of thumb is df.shape[0] / batch_size
        # if we don't have this, the KLD loss might become much larger than the reconstruction loss
        total *= self.dataset_size / self.bs

        return (total + (kl_weight * KLD)) / cats.size(0)


class AnnealedLossCallback(Callback):
    def after_pred(self):
        kl = self.learn.pred[0].new(1)
        opt = self.opt.hypers[0]
        if 'kl_weight' in opt:
            kl[0] = self.opt.hypers[0]['kl_weight']
            self.learn.pred = self.learn.pred + (kl,)

    def after_batch(self):
        if (len(self.learn.pred) > 4):
            cats, conts, mu, logvar, _ = self.learn.pred
        else:
            cats, conts, mu, logvar = self.learn.pred

        self.learn.pred = (cats, conts, mu, logvar)
