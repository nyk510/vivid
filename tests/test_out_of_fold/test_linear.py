from vivid.out_of_fold.linear import RidgeOutOfFold, LogisticOutOfFold


def test_ridge(regression_data):
    df, y = regression_data
    oof = RidgeOutOfFold(name='test_ridge', n_trials=10)
    oof.fit(df, y)


def test_logistic(binary_data):
    df, y = binary_data
    oof = LogisticOutOfFold(name='test_logistic', n_trials=10)
    oof.fit(df, y)
