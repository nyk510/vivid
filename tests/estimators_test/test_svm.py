from vivid.estimators.svm import TunedSVCBlock, TunedSVRVBlock, SVRBlock


def test_optuna_svm(binary_data):
    oof = TunedSVCBlock(n_trials=1, name='svm_01')
    train_df, y = binary_data
    oof.fit(train_df, y)


def test_normals(regression_data):
    train_df, y = regression_data
    for cls in [SVRBlock, SVRBlock]:
        oof_feat = cls(name='sample')
        oof_feat.fit(train_df, y)


def test_optuna(regression_data):
    oof_feat = TunedSVRVBlock(n_trials=1, name='optuna_svr')
    train_df, y = regression_data
    oof_feat.fit(train_df, y)
