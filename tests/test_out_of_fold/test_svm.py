from vivid.out_of_fold.svm import SVCOptunaOutOfFold, SVROptunaOutOfFold, SVROutOfFold


def test_optuna_svm(binary_data):
    oof = SVCOptunaOutOfFold(n_trials=1, name='svm_01')
    train_df, y = binary_data
    oof.fit(train_df, y)


def test_normals(regression_data):
    train_df, y = regression_data
    for cls in [SVROutOfFold, SVROutOfFold]:
        oof_feat = cls(name='sample')
        oof_feat.fit(train_df, y)


def test_optuna(regression_data):
    oof_feat = SVROptunaOutOfFold(n_trials=1, name='optuna_svr')
    train_df, y = regression_data
    oof_feat.fit(train_df, y)
