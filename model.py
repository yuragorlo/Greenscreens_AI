import lightgbm as lgb


class Model:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'n_estimators': 500,
                'learning_rate': 0.1,
                'max_depth': -1,
                'num_leaves': 63,
                'subsample': 1.0,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'min_split_gain': 0.15,
                'min_child_samples': 15,
                'min_gain_to_split': 0.0,
                'boosting_type': 'gbdt',
                'data_sample_strategy': 'goss',
                'force_row_wise': True,
                'objective': 'regression',
                'verbosity': -1
            }
        else:
            self.params = params
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

