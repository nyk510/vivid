from vivid.estimators.linear import TunedRidgeBlock, TunedLogisticBlock


def test_ridge(regression_data):
    df, y = regression_data
    oof = TunedRidgeBlock(name='test_ridge', n_trials=10)
    oof.fit(df, y)


def test_logistic(binary_data):
    df, y = binary_data
    oof = TunedLogisticBlock(name='test_logistic', n_trials=10)
    oof.fit(df, y)
