"""test about netural network out-of-fold feature"""

from vivid.estimators.neural_network import KerasClassifierBlock, KerasRegressorBlock


# binary のみの対応なので multiclass regression model に対応できていない. 今後修正する
# def test_keras_classifier(binary_data, experiment):
#     df, y = binary_data
#     model = KerasClassifierBlock(name='skeras',
#                                  add_init_param={'epochs': 1})
#     model.fit(df, y, experiment)
#     model.transform(df)


def test_keras_regressor(regression_data, experiment):
    df, y = regression_data
    model = KerasRegressorBlock(name='skeras',
                                add_init_param={'epochs': 1})
    model.fit(df, y, experiment)
    model.transform(df)
