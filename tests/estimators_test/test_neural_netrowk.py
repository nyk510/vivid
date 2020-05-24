"""test about netural network out-of-fold feature"""

from vivid.estimators.neural_network import KerasClassifierBlock, KerasRegressorBlock


def test_keras_classifier(binary_data):
    df, y = binary_data
    model = KerasClassifierBlock(name='skeras', add_init_param={'epochs': 1})
    model.fit(df, y)
    model.predict(df)


def test_keras_regressor(regression_data):
    df, y = regression_data
    model = KerasRegressorBlock(name='skeras', add_init_param={'epochs': 1})
    model.fit(df, y)
    model.predict(df)
