from fastai.data.external import untar_data, URLs
import pandas as pd

from tvae.optim import Optim


def adult_dataset():
    path = untar_data(URLs.ADULT_SAMPLE)
    df = pd.read_csv(path / 'adult.csv')

    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'native-country']
    cont_names = ['age', 'education-num', 'capital-loss', 'capital-gain', 'hours-per-week']

    all_cols = cat_names + cont_names

    t = ['age', 'race']

    if 'fnlwgt' in df.columns:
        df = df.drop(columns='fnlwgt')

    t_df = df[[c for c in all_cols if c in t]]
    t_cat_cols = [c for c in cat_names if c in t]
    t_cont_cols = [c for c in cont_names if c in t]

    x_df = df[[c for c in all_cols if c not in t]]
    x_cat_cols = [c for c in cat_names if c not in t]
    x_cont_cols = [c for c in cont_names if c not in t]

    return t_df, t_cat_cols, t_cont_cols, x_df, x_cat_cols, x_cont_cols, df, cat_names, cont_names, all_cols


t_df, t_cat_cols, t_cont_cols, x_df, x_cat_cols, x_cont_cols, df, cat_names, cont_names, all_cols = adult_dataset()
# %%
from tvae.model import TVAE, VAEConfig

config = VAEConfig()

# %%
tvae = TVAE(config=config, df=x_df, cat_names=x_cat_cols, cont_names=x_cont_cols)
#
# %%
# recon_perf, ood_perf = tvae.train_and_evaluate(N=10000)
#
# #%%
# tvae.make_encoding_distribution_plots(show=True)
#
# #%%
# tvae.compare_synthetic_data_distributions()

# %%
optim = Optim()
optim.find_best_config(x_df, x_cat_cols, x_cont_cols, nb_trials=1)

# tvawe = TVAE(config=config.from_study(optim.study), df=x_df, cat_names=x_cat_cols, cont_names=x_cont_cols)
# tvawe.train_and_evaluate()

# %%
# optim.sweep_plot()

# %%
optim.pareto_plot()

print("hell")
