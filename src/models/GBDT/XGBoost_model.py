from xgboost import XGBRegressor
from src.ml_config.XGBoost import XGBoostConfig

class XGBoostModel(XGBRegressor):

    def __init__(self):
        super().__init__(**XGBoostConfig)


    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
