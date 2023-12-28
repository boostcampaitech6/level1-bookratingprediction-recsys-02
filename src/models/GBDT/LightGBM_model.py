from lightgbm import LGBMModel
from src.ml_config.LightGBM import LightGBMConfig

class LightGBMModel(LGBMModel):

    def __init__(self):
        super().__init__(**LightGBMConfig)

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)