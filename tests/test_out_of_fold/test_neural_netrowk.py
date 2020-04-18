"""test about netural network out-of-fold feature"""

from vivid.out_of_fold.neural_network import SkerasClassifierOutOfFoldFeature, SkerasRegressorOutOfFoldFeature
from .test_base import get_binary


class TestLinear:
    def setup_method(self):
        df, y = get_binary()
        self.df = df
        self.y = y

    def test_keras_classifier(self):
        model = SkerasClassifierOutOfFoldFeature(name='skeras', add_init_param={'epochs': 1})
        model.fit(self.df, self.y)
        model.predict(self.df)

    def test_keras_regressor(self):
        model = SkerasRegressorOutOfFoldFeature(name='skeras', add_init_param={'epochs': 1})
        model.fit(self.df, self.y)
        model.predict(self.df)
