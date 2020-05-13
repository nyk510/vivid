import os

from vivid.out_of_fold.linear import RidgeOutOfFold, LogisticOutOfFold


def test_ridge(regression_data, output_dir):
    df, y = regression_data
    oof = RidgeOutOfFold(name='test_ridge', n_trials=10, root_dir=output_dir)
    oof.fit(df, y)


def test_logistic(binary_data, output_dir):
    df, y = binary_data
    oof = LogisticOutOfFold(name='test_logistic', n_trials=10, root_dir=output_dir)
    oof.fit(df, y)

    assert os.path.exists(os.path.join(oof.output_dir, 'feature_importance.csv')), os.listdir(oof.output_dir)
