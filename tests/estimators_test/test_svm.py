from vivid.estimators.svm import TunedSVCBlock, TunedSVRVBlock, SVRBlock


def test_optuna_svm(binary_data, experiment):
    oof = TunedSVCBlock(n_trials=1, name='svm_01')
    train_df, y = binary_data
    oof.fit(train_df, y, experiment)


def test_normals(regression_data, experiment):
    train_df, y = regression_data
    for cls in [SVRBlock, SVRBlock]:
        oof_feat = cls(name='sample')
        oof_feat.fit(train_df, y, experiment)


def test_optuna(regression_data, experiment):
    oof_feat = TunedSVRVBlock(n_trials=1, name='optuna_svr')
    train_df, y = regression_data
    oof_feat.fit(train_df, y, experiment)
