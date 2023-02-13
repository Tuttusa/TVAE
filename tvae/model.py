import json
from dataclasses import dataclass, field, asdict
from typing import List
import uuid
import pacmap
import pathlib
import numpy as np
import hashlib

from optuna import Study
from scipy.stats import entropy
from sklearn.metrics import r2_score
from fastai.callback.schedule import combine_scheds, SchedCos, SchedNo, ParamScheduler
from fastai.data.block import TransformBlock
from fastai.data.transforms import Normalize, RandomSplitter
from fastai.imports import noops
from fastai.layers import Swish, LinBnDrop, SigmoidRange
from fastai.optimizer import ranger
from fastai.tabular.core import TabularPandas, TabDataLoader, Categorify, FillMissing, TabularProc
from fastai.tabular.learner import TabularLearner
from fastai.tabular.model import get_emb_sz, TabularModel
from fastai.torch_core import tensor, to_device, Module
from fastcore.basics import store_attr
from fastcore.foundation import L
from fastcore.meta import delegates
from fastcore.transform import ItemTransform
import torch
import pandas as pd
from torch import nn, HalfTensor
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.distributions import Normal
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from tvae.loss import VAERecreatedLoss, AnnealedLossCallback
from tvae.metrics import CEMetric, KLDMetric, MUMetric, StdMetric, MSEMetric, mean_absolute_relative_error
from tvae.paths import models_path
from tvae.utils import VAEConfig, DataConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pd.set_option('display.float_format', lambda x: '%.3f' % x)


class ReadTabBatchIdentity(ItemTransform):
    "Read a batch of data and return the inputs as both `x` and `y`"

    def __init__(self, to):
        super().__init__()
        store_attr()

    def encodes(self, to):
        if not to.with_cont:
            res = (tensor(to.cats).long(),) + (tensor(to.cats).long(),)
        else:
            res = (tensor(to.cats).long(), tensor(to.conts).float()) + (
                tensor(to.cats).long(), tensor(to.conts).float())
        if to.device is not None:
            res = to_device(res, to.device)
        return res


class TabularPandasIdentity(TabularPandas):

    @property
    def total_cats(self):
        return {k: len(v) for k, v in self.classes.items()}

    @property
    def low(self):
        return (self.cont_min - np.array(list(self.means.values()))) / np.array(list(self.stds.values()))

    @property
    def high(self):
        return (self.cont_max - np.array(list(self.means.values()))) / np.array(list(self.stds.values()))


@delegates()
class TabDataLoaderIdentity(TabDataLoader):
    "A transformed `DataLoader` for AutoEncoder problems with Tabular data"
    do_item = noops

    def __init__(self, dataset, bs=16, shuffle=False, after_batch=None, num_workers=0, **kwargs):
        if after_batch is None:
            after_batch = L(TransformBlock().batch_tfms) + \
                ReadTabBatchIdentity(dataset)
        super().__init__(dataset, bs=bs, shuffle=shuffle,
                         after_batch=after_batch, num_workers=num_workers, **kwargs)

    def create_batch(self, b): return self.dataset.iloc[b]

    def do_item(self, s): return 0 if s is None else s


TabularPandasIdentity._dl_type = TabDataLoaderIdentity


class SetMinMax(TabularProc):

    def encodes(self, to):
        to.cont_min = to[to.cont_names].min().values
        to.cont_max = to[to.cont_names].max().values


class BatchSwapNoise(Module):
    "Swap Noise Module"

    def __init__(self, p):
        super().__init__()
        store_attr()

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.size()) > (1 - self.p)
            l1 = torch.floor(torch.rand(x.size()) * x.size(0)
                             ).type(torch.LongTensor)
            l2 = (mask.type(torch.LongTensor) * x.size(1))
            res = (l1 * l2).view(-1)
            idx = torch.arange(x.nelement()) + res
            idx[idx >= x.nelement()] = idx[idx >= x.nelement()] - x.nelement()
            return x.flatten()[idx].view(x.size())
        else:
            return x


class TabularVAE(TabularModel):
    def __init__(self, emb_szs, n_cont, hidden_size, cats, low, high, layers=[1024, 512, 256], ps=0.2,
                 embed_p=0.01, bswap=None, act_cls=Swish()):
        super().__init__(emb_szs, n_cont, layers=layers,
                         out_sz=hidden_size, embed_p=embed_p, act_cls=act_cls)

        self.bswap = bswap
        self.cats = cats
        self.activation_cats = sum([v for k, v in cats.items()])

        self.layers = nn.Sequential(
            *L(self.layers.children())[:-1] + nn.Sequential(LinBnDrop(256, hidden_size, p=ps, act=act_cls)))

        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_std = nn.Linear(hidden_size, hidden_size)

        if self.bswap != None:
            self.noise = BatchSwapNoise(self.bswap)
        self.decoder = nn.Sequential(
            LinBnDrop(hidden_size, 256, p=ps, act=act_cls),
            LinBnDrop(256, 512, p=ps, act=act_cls),
            LinBnDrop(512, 1024, p=ps, act=act_cls)
        )

        self.decoder_cont = nn.Sequential(
            LinBnDrop(1024, n_cont, p=ps, bn=False, act=None),
            SigmoidRange(low=low, high=high)
        )

        self.decoder_cat = LinBnDrop(
            1024, self.activation_cats, p=ps, bn=False, act=None)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = HalfTensor(*mu.size()).normal_().to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc_mu(h), F.softplus(self.fc_std(h))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x_cat, x_cont=None):
        encoded = super().forward(x_cat, x_cont)
        z, mu, logvar = self.bottleneck(encoded)

        return z, mu, logvar

    def decode(self, z):
        decoded_trunk = self.decoder(z)

        decoded_cats = self.decoder_cat(decoded_trunk)

        decoded_conts = self.decoder_cont(decoded_trunk)

        return decoded_cats, decoded_conts

    def forward(self, x_cat, x_cont=None, encode=False):
        if (self.bswap != None):
            x_cat = self.noise(x_cat)
            x_cont = self.noise(x_cont)

        z, mu, logvar = self.encode(x_cat, x_cont)
        if (encode):
            return z

        decoded_cats, decoded_conts = self.decode(z)

        return decoded_cats, decoded_conts, mu, logvar


class Reducer:
    def __init__(self):
        self.reducer = None

    def trained(self):
        return (self.reducer is not None) and (self.reducer.tree is not None)

    def train(self, xenc: np.array, reducer_args: dict):
        default_reducer_args = {'n_components': 2, 'n_neighbors': None, 'MN_ratio': 0.5, 'FP_ratio': 2.0,
                                'save_tree': True, 'num_iters': 1, 'verbose': True}
        default_reducer_args.update(reducer_args)
        
        self.reducer = pacmap.PaCMAP(**default_reducer_args)
        self.reducer.fit(xenc)

    def process(self, xenc, num_iter=10):
        emb = self.reducer.transform(xenc)
        return emb

    def save(self, path):
        if self.reducer is not None:
            pacmap.save(self.reducer, path)

    def load(self, path: pathlib.Path):
        if path.exists():
            self.reducer = pacmap.load(path)
        return self


class TVAE:
    def __init__(self, config: VAEConfig, data_config: DataConfig, path=None, name=None):

        self.name = name

        self.config = config
        self.data_config = data_config

        if name is None:
            self.name = hashlib.sha256(json.dumps(
                self.config.to_dict()).encode('utf-8')).hexdigest()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        f = combine_scheds([.1, .3, .6], [SchedCos(
            0, 0), SchedCos(0, 1), SchedNo(1, 1)])

        cbs = [ParamScheduler({'kl_weight': f}), AnnealedLossCallback()]

        to = TabularPandasIdentity(self.data_config.df, [Categorify, FillMissing, Normalize, SetMinMax],
                                   self.data_config.cat_names, self.data_config.cont_names,
                                   splits=RandomSplitter(seed=32)(self.data_config.df))

        metrics = [MSEMetric(), CEMetric(to.total_cats),
                   KLDMetric(), MUMetric(), StdMetric()]

        dls = to.dataloaders(bs=self.config.batch_size)
        dls.n_inp = 2

        model = TabularVAE(get_emb_sz(to.train), len(self.data_config.cont_names), self.config.hidden_size,
                           ps=self.config.dropout,
                           cats=to.total_cats,
                           embed_p=self.config.embed_p, bswap=self.config.bswap,
                           low=tensor(to.low, device=device),
                           high=tensor(to.high, device=device))

        model.to(device)
        loss_func = VAERecreatedLoss(
            to.total_cats, self.data_config.df.shape[0], self.config.batch_size, to.total_cats)
        learn = TabularLearner(dls, model, lr=self.config.lr, loss_func=loss_func, wd=self.config.wd,
                               opt_func=ranger,
                               cbs=cbs,
                               metrics=metrics).to_fp16()

        self.learn = learn
        self.to = to
        self.reducer = Reducer()

        self.model_path = path
        if path is None:
            self.model_path = models_path.joinpath(self.name)

    def train(self):
        self.learn.fit_flat_cos(self.config.epochs, lr=self.config.lr)

    def _to_continuous_dataframe(self, cont_preds):
        if isinstance(cont_preds, torch.Tensor):
            cont_preds = cont_preds.cpu().numpy()
        return pd.DataFrame(
            (cont_preds * np.array(list(self.to.stds.values()))) +
            np.array(list(self.to.means.values())),
            columns=self.to.cont_names)

    def _to_cat_dataframe(self, cat_preds):
        cat_reduced = np.zeros((cat_preds.shape[0], len(self.to.total_cats)))
        pos = 0
        for i, (k, v) in enumerate(self.to.total_cats.items()):
            cat_reduced[:, i] = cat_preds[:, pos:pos + v].argmax(axis=1)
            pos += v
        cat_reduced = pd.DataFrame(cat_reduced, columns=self.to.cat_names)
        return cat_reduced

    def _new_unique_rows_generated(self, synth_df, real_df):
        uniq_synth = synth_df[self.data_config.cat_names].drop_duplicates()
        uniq_real = real_df[self.data_config.cat_names].drop_duplicates()
        new_uniq = pd.concat([uniq_real, uniq_synth], axis=0).drop_duplicates()

        new_vals_uniq = (new_uniq.shape[0] -
                         uniq_real.shape[0]) / uniq_real.shape[0]

        return new_vals_uniq

    def _continuous_performance(self, cont_preds, cont_targs):
        """
        evaluate the performance on continuous columns
        """
        cont_preds = pd.DataFrame(cont_preds, columns=self.to.cont_names)
        cont_targs = pd.DataFrame(cont_targs, columns=self.to.cont_names)

        preds = self._to_continuous_dataframe(cont_preds)
        targets = self._to_continuous_dataframe(cont_targs)

        mi = (np.abs(targets - preds)).min().to_frame().T
        ma = (np.abs(targets - preds)).max().to_frame().T
        mean = (np.abs(targets - preds)).mean().to_frame().T
        median = (np.abs(targets - preds)).median().to_frame().T
        r2 = pd.DataFrame.from_dict(
            {c: [r2_score(targets[c], preds[c])] for c in preds.columns})
        mape = pd.DataFrame.from_dict(
            {c: [mean_absolute_relative_error(targets[c], preds[c])] for c in preds.columns})

        r2.mean(axis=1)

        for d, name in zip([mi, ma, mean, median, r2, mape], ['Min', 'Max', 'Mean', 'Median', 'R2', 'MARE']):
            d = d.insert(0, 'GroupBy', name)

        data = pd.concat([mi, ma, mean, median, r2, mape])

        return data

    def _categorical_performance(self, cat_targs, cat_preds, vae_uncert=None):
        """
        evaluate the performance on categorical columns
        """
        cat_preds = self._to_cat_dataframe(cat_preds)

        cat_targs = pd.DataFrame(cat_targs, columns=self.to.cat_names)

        accuracy = pd.DataFrame.from_dict(
            {c: [accuracy_score(cat_targs[c], cat_preds[c])] for c in cat_preds.columns})
        recall = pd.DataFrame.from_dict({c: [recall_score(cat_targs[c], cat_preds[c], average='weighted')]
                                         for c in cat_preds.columns})
        precision = pd.DataFrame.from_dict(
            {c: [precision_score(cat_targs[c], cat_preds[c], average='weighted')] for c in cat_preds.columns})

        f1 = pd.DataFrame.from_dict(
            {c: [f1_score(cat_targs[c], cat_preds[c], average='weighted')] for c in cat_preds.columns})

        for d, name in zip([accuracy, recall, precision, f1], ['Accuracy', 'Recall', 'Precision', 'F1']):
            d = d.insert(0, 'MetricName', name)

        gg = pd.concat([accuracy, recall, precision, f1])

        return gg

    def _evaluate_recon_pref(self):
        df_dec, cats, conts, dl, outs_enc = self.reconstruct(
            self.data_config.df)

        conts = self._to_continuous_dataframe(conts)
        df_d = pd.concat(
            [pd.DataFrame(cats, columns=self.to.cat_names), conts], axis=1)
        df_dec.columns = list(map(lambda x: f"{x}_rec", df_dec.columns))

        comp_df = pd.concat([df_d, df_dec], axis=1)
        comp_df_l = sum(
            list(map(list, zip(df_d.columns.tolist(), df_dec.columns.tolist()))), [])
        comp_df = comp_df[comp_df_l]

        (cat_preds, cont_preds, mu, logvar), (cat_targs,
                                              cont_targs) = self.learn.get_preds(dl=dl)

        cont_perf_data = self._continuous_performance(cont_preds, cont_targs)
        cat_perf_data = self._categorical_performance(cat_targs, cat_preds)

        return comp_df, cont_perf_data, cat_perf_data
        # df_dec

    def _analyse_entropy(self, synth_df, real_df):
        """Checks the entropy of each value of each column"""

        def cols_entropy(df):
            res_entr = {}
            res_val_count = {}
            for col in df.columns:
                res_val_count[col] = df[col].value_counts() / df[col].shape[0]
                res_entr[col] = entropy(res_val_count[col])
            res_entr = pd.DataFrame(res_entr, index=[0])
            return res_entr, res_val_count

        real_df_entr, real_df_val_c = cols_entropy(
            real_df[self.data_config.cat_names])
        synth_df_entr, synth_df_val_c = cols_entropy(
            synth_df[self.data_config.cat_names])

        comp_val_c = {}
        for c, v in real_df_val_c.items():
            comp_val_c[c] = pd.DataFrame(
                {f"{c}_real": real_df_val_c[c], f"{c}_synth": synth_df_val_c[c]}).fillna(0.0)

        comp_entr = (synth_df_entr - real_df_entr).mean()

        return comp_entr, real_df_entr, synth_df_entr, comp_val_c

    def _get_pvalue_uncertainty(self, org_enc, new_enc, use_encoder=False):
        if use_encoder:
            org_enc = self.encode(org_enc)[0]
            new_enc = self.encode(new_enc)[0]

        org_gn = torch.exp(Normal(torch.from_numpy(org_enc.mean(axis=0)), torch.from_numpy(org_enc.std(axis=0)))
                           .log_prob(torch.from_numpy(new_enc))).mean(axis=1)

        return org_gn

    def _evaluate_ood_perf(self, N=None):
        N = self.data_config.df.shape[0] if N is None else N
        real_df = self._transform(self.data_config.df)[0]
        synth_df = self.generate(N, 0.0, 2.0)[0]
        new_uniq = self._new_unique_rows_generated(synth_df, real_df)
        comp_entr, real_df_entr, synth_df_entr, value_count_comp = self._analyse_entropy(
            synth_df, real_df)

        t_alea_unc = self._get_pvalue_uncertainty(
            real_df, synth_df, use_encoder=True)
        x_alea_unc = self._get_pvalue_uncertainty(
            real_df, synth_df, use_encoder=True)

        total_alea_unc = (x_alea_unc.numpy(), t_alea_unc.numpy())

        return new_uniq, comp_entr, total_alea_unc, value_count_comp, real_df_entr, synth_df_entr

    def evaluate(self, N):
        recon_perf = self._evaluate_recon_pref()
        ood_perf = self._evaluate_ood_perf(N)
        return recon_perf, ood_perf

    def train_and_evaluate(self, N=10000):
        self.train()
        return self.evaluate(N)

    def train_dimension_reduction(self, reducer_args={}):

        xenc = self.encode(self.data_config.df)[0]
        self.reducer.train(xenc, reducer_args)

    def reduce_embed_dims(self, xenc, encode=False, num_iters=10):
        if not self.reducer.trained():
            self.train_dimension_reduction(
                reducer_args={'num_iters': num_iters})

        if encode:
            xenc = self.encode(xenc)[0]

        emb = self.reducer.process(xenc)
        return emb

    def _recon_df(self, cont_preds, cat_preds):
        cont_df = self._to_continuous_dataframe(cont_preds)
        cat_df = self._to_cat_dataframe(cat_preds)
        res = pd.concat([cat_df, cont_df], axis=1)
        return res

    def _transform(self, df):
        "reconstruct a dataframe comming from the decoder to a real dataframe"
        dl = self.learn.dls.test_dl(df)
        x_cats = torch.cat(list(map(lambda x: x[0], dl)))
        x_conts = torch.cat(list(map(lambda x: x[1], dl)))

        x_conts = self._to_continuous_dataframe(x_conts)
        df_d = pd.concat(
            [pd.DataFrame(x_cats, columns=self.to.cat_names), x_conts], axis=1)

        return df_d, x_cats, x_conts

    def encode(self, df):
        with torch.no_grad():
            self.learn.model.eval()
            dl = self.learn.dls.test_dl(df)
            cats = torch.cat(list(map(lambda x: x[0], dl))).to(self.device)
            conts = torch.cat(list(map(lambda x: x[1], dl))).to(self.device)
            outs_enc = self.learn.model.encode(cats, conts)[0].cpu().numpy()
        return outs_enc, dl, cats, conts

    def decode(self, outs_enc):
        with torch.no_grad():
            self.learn.model.eval()
            if isinstance(outs_enc, np.ndarray):
                outs_enc = torch.from_numpy(outs_enc).to(self.device)
            outs_dec_cats, outs_dec_conts = self.learn.model.decode(
                outs_enc.float())
            df_dec = self._recon_df(
                outs_dec_conts.cpu().numpy(), outs_dec_cats.cpu().numpy())
        return df_dec

    def reconstruct(self, df, transform=True):
        with torch.no_grad():
            outs_enc, dl, cats, conts = self.encode(df)
            df_dec = self.decode(outs_enc)
            if transform:
                df_dec, cats, conts = self._transform(df_dec)

        return df_dec, cats, conts, dl, outs_enc

    def generate(self, N, mean, std):
        with torch.no_grad():
            self.learn.model.eval()
            outs_enc = torch.normal(mean=mean, std=std, size=(
                N, self.config.hidden_size)).to(self.device)
            outs_dec_cats, outs_dec_conts = self.learn.model.decode(outs_enc)
            df_dec = self._recon_df(
                outs_dec_conts.cpu().numpy(), outs_dec_cats.cpu().numpy())

        return df_dec, outs_enc

    def vae_uncert(self, outs_enc):
        org_gn = torch.exp(Normal(torch.from_numpy(outs_enc.mean(axis=0)),
                                  torch.from_numpy(outs_enc.std(axis=0))).
                           log_prob(torch.from_numpy(outs_enc))).sum(axis=1)

        return org_gn

    def save(self):
        # save model
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.config.save(self.model_path.joinpath('config.pkl').as_posix())
        self.data_config.save(self.model_path.joinpath(
            'data_config.pkl').as_posix())
        self.learn.save(file=self.model_path.joinpath('model').as_posix())
        self.reducer.save(self.model_path.joinpath('reducer'))

    def load(self):
        # load model
        self.learn = self.learn.load(
            self.model_path.joinpath('model').as_posix())
        self.config = VAEConfig.load(
            self.model_path.joinpath('config.pkl').as_posix())
        self.data_config = DataConfig.load(
            self.model_path.joinpath('data_config.pkl').as_posix())
        self.reducer = self.reducer.load(self.model_path.joinpath('reducer'))

        return self

    def make_encoding_distribution_plots(self, N=1000, show=False, legend=False):
        out_enc = self.encode(self.data_config.df.sample(N))[0]
        outs_enc_df = pd.DataFrame(
            out_enc, columns=list(range(out_enc.shape[1])))

        g = sns.kdeplot(data=outs_enc_df, legend=legend)

        plt.title(f"Latent dimensions distribution")

        if show:
            plt.show()

    def compare_synthetic_data_distributions(self, N=10000):
        all_cols = self.data_config.cat_names + self.data_config.cont_names
        real_df = self._transform(self.data_config.df)[0]
        synth_df = self.generate(N, 0.0, 2.0)[0]
        for scol in all_cols:
            print(scol)
            sns.kdeplot(data=real_df[[scol]], palette="crest")
            sns.kdeplot(data=synth_df[[scol]], palette='rocket')
            plt.title(f" {scol} distribution")
            plt.show()
