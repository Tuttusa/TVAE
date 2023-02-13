import uuid
from typing import List

import optuna
import pandas as pd
import matplotlib.pyplot as plt

from tvae.model import TVAE
from tvae.paths import tuning_path
from tvae.utils import VAEConfig, DataConfig

class Optim:
    def __init__(self, save_path=None, name=str(uuid.uuid4())):

        if save_path is None:
            tuning_path.mkdir(parents=True, exist_ok=True)
            save_path = tuning_path.joinpath(
                name+'_vae_compression.db').as_posix()

        self.study = optuna.create_study(storage=f"sqlite:///{save_path}",
                                         study_name="vae_compression",
                                         load_if_exists=True, directions=["maximize", "minimize", "maximize"])

    def find_best_config(self, df: pd.DataFrame, cat_cols: List[str], cont_cols: List[str], nb_trials=100):

        def objective(trial):
            """
            optuna optimization to minimize the reconstruction error of the vae and the size of the vae
            """
            nb_layers = trial.suggest_int("nb_layers", 2, 6)
            strt_layer_size = trial.suggest_int("layers", 24, 1024)
            layers = [int(strt_layer_size / (e + 1)) for e in range(nb_layers)]
            trial.set_user_attr("layers", layers)

            config = VAEConfig(
                hidden_size=trial.suggest_int("hidden_size", 1, 128),
                dropout=trial.suggest_float("dropout", 0.0, 0.5),
                embed_p=trial.suggest_float("embed_p", 0.0, 0.5),
                wd=trial.suggest_float("wd", 0.01, 0.1),
                bswap=trial.suggest_float("bswap", 0.0, 0.5),
                lr=trial.suggest_float("lr", 1e-3, 4e-3),
                epochs=trial.suggest_int("epochs", 9, 30),
                batch_size=trial.suggest_int("batch_size", 128, 1024),
                layers=layers
            )

            tvae = TVAE(config=config, data_config=DataConfig(df=df, cat_names=cat_cols, cont_names=cont_cols))

            recon_perf, ood_perf = tvae.train_and_evaluate(N=10000)

            f1_s = recon_perf[2][recon_perf[2]['MetricName']
                                 == 'F1'].iloc[:, 1:].to_numpy().mean()
            mape = recon_perf[1][recon_perf[1]['GroupBy']
                                 == 'MARE'].iloc[:, 1:].to_numpy().mean()

            recon_error = f1_s - mape

            new_ood = ood_perf[0]

            trial.set_user_attr("f1", f1_s)
            trial.set_user_attr("mape", mape)
            trial.set_user_attr("new_ood", new_ood)

            return recon_error, config.hidden_size, new_ood

        self.study.optimize(objective, n_trials=nb_trials)

    def pareto_plot(self, save=True):
        study_plot = optuna.visualization.plot_pareto_front(self.study, target_names=["reconstruction", "hidden_size",
                                                                                      "ood_generation"],
                                                            targets=lambda x: x.values[:3])
        if save:
            study_plot.write_html(tuning_path.joinpath(
                "t_vae_study_plot.html").as_posix())

        return study_plot

    def sweep_plot(self, save=True):
        study_sweep_plot = optuna.visualization.plot_parallel_coordinate(self.study,
                                                                         target=lambda x: x.values[0])
        if save:
            study_sweep_plot.write_html(tuning_path.joinpath(
                "t_vae_study_plot_sweep.html").as_posix())

        return study_sweep_plot
