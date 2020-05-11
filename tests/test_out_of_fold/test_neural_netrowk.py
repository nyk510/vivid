"""test about netural network out-of-fold feature"""

from vivid.out_of_fold.neural_network import SkerasClassifierOutOfFoldFeature, SkerasRegressorOutOfFoldFeature


def test_keras_classifier(binary_data):
    df, y = binary_data
    model = SkerasClassifierOutOfFoldFeature(name='skeras', add_init_param={'epochs': 1})
    model.fit(df, y)
    model.predict(df)


def test_keras_regressor(regression_data):
    df, y = regression_data
    model = SkerasRegressorOutOfFoldFeature(name='skeras', add_init_param={'epochs': 1})
    model.fit(df, y)
    model.predict(df)
