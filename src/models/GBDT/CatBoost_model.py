from catboost import CatBoostRegressor
from src.ml_config.CatBoost import CatBoostConfig

class CatBoostModel(CatBoostRegressor):

    def __init__(self, args):
        
        super().__init__(**CatBoostConfig)
#            learning_rate=args.lr, depth=args.depth, 
#            iterations=args.iterations,
#            min_child_samples=40,
#            grow_policy='Depthwise',
#            #early_stopping_rounds=100,
#        )

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
