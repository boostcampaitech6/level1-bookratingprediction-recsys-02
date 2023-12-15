from catboost import CatBoostRegressor

class CatBoostModel(CatBoostRegressor):

    def __init__(self, args):
        super().__init__(
            learning_rate=args.lr, depth=args.depth, 
            iterations=args.iterations,
            early_stopping_rounds=20,
        )

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
