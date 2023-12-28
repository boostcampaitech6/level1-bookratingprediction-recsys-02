from lightgbm import LGBMRegressor
from src.ml_config.LightGBM import LightGBMConfig

class LightGBMModel(LGBMRegressor):

    def __init__(self):
        super().__init__(**LightGBMConfig)

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)