XGBoostConfig = {
    'learning_rate': 0.01,
    'early_stopping_rounds': 25,
    'tree_method':'hist',
    'device':'cuda',
    'booster':'gbtree',
    'n_estimators': 100000,
    'max_depth': 14,
    'min_child_weight': 1,
    'gamma': 1,
    'colsample_bytree': 0.5,
    'subsample': 0.8,
    'random_state': 42,
}