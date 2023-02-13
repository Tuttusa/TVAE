from dataclasses import dataclass, field, asdict
from typing import List
from optuna import Study
import pandas as pd


class SaveLoadMixin:

    @classmethod
    def load(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            return obj

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


@dataclass
class VAEConfig(SaveLoadMixin):
    hidden_size: int = 64
    dropout: float = 0.0
    embed_p: float = 0.0
    wd: float = 0.01
    bswap: float = 0.1
    lr: float = 4e-3
    epochs: int = 2
    batch_size: int = 1024
    layers: List[int] = field(default_factory=lambda: [1024, 512, 256])

    def from_study(self, study: Study, trial_num=0):
        for k, v in study.best_trials[trial_num].params.items():
            setattr(self, k, v)
        return self

    def to_dict(self):
        return asdict(self)


@dataclass
class DataConfig(SaveLoadMixin):
    df: pd.DataFrame = None
    cat_names: List[str] = None
    cont_names: List[str] = None
