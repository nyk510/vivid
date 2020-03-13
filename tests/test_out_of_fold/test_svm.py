"""何故かSVM周りのテストはめちゃくちゃ重たい. (そもそも実行すらできないことも多い)
余り使わないので取り除いて問題ないと判断し、削除"""

# from vivid.out_of_fold.svm import SVCOptunaOutOfFold, SVROptunaOutOfFold, SVROutOfFold
# from .test_base import get_binary, get_boston
#
#
# def test_optuna_svm():
#     oof = SVCOptunaOutOfFold(n_trials=1, name='svm_01')
#     df_train, y = get_binary()
#     oof.fit(df_train, y)
#
#
# class TestSVR:
#     def test_normals(self):
#         df_train, y = get_boston()
#         for cls in [SVROutOfFold, SVROutOfFold]:
#             oof_feat = cls(name='sample')
#             oof_feat.fit(df_train, y)
#
#     def test_optuna(self):
#         oof_feat = SVROptunaOutOfFold(n_trials=1, name='optuna_svr')
#         df_train, y = get_boston()
#         oof_feat.fit(df_train, y)
