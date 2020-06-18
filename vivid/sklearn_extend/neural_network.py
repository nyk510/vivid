from typing import Union

from keras.callbacks import Callback
from keras.layers import Dropout, Dense, Input, BatchNormalization, ReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import roc_auc_score


class ROCAucCallback(Callback):
    def __init__(self, training_data, validation_data):
        super(ROCAucCallback, self).__init__()
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x, verbose=0)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc'] = roc_auc_score(self.y, y_pred)
        logs['norm_gini'] = (roc_auc_score(self.y, y_pred) * 2) - 1

        y_pred_val = self.model.predict(self.x_val, verbose=0)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)
        logs['norm_gini_val'] = (roc_auc_score(self.y_val, y_pred_val) * 2) - 1

        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (
            str(round(roc, 5)), str(round(roc_val, 5)), str(round((roc * 2 - 1), 5)), str(round((roc_val * 2 - 1), 5))),
              end=10 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return


def dense_bnn_block(input_tensor, hidden_dim, activation=ReLU, dropout_ratio=.5):
    x = Dense(hidden_dim)(input_tensor)
    x = BatchNormalization()(x)
    x = activation()(x)
    if dropout_ratio > 0:
        x = Dropout(rate=dropout_ratio)(x)
    return x


class ScikitKerasMixin:
    """
    Mixin Class adapt Keras model to scikit-learn api
    """

    def fit(self: Union['ScikitKerasMixin', KerasClassifier], x, y=None, sample_weight=None, **kwargs):
        self.sk_params['n_input'] = x.shape[1]
        history = super(ScikitKerasMixin, self).fit(x, y=y, sample_weight=sample_weight, **kwargs)
        self.history_ = history
        return self

    def bottleneck(self, input_tensor):
        hidden_dims = [
            1024, 512, 256, 128
        ]
        x = input_tensor
        for hidden in hidden_dims:
            x = dense_bnn_block(x, hidden_dim=hidden)

        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        feature = Dropout(rate=.2)(x)
        return feature

    def __call__(self, n_input: int) -> Model:
        """
        define keras model class and compile
        Args:
            n_input:
        Returns:
        """

        raise NotImplementedError()


class ScikitKerasClassifier(ClassifierMixin, ScikitKerasMixin, KerasClassifier):
    def __call__(self, n_input) -> Model:
        input = Input(shape=(n_input,))
        feature = self.bottleneck(input)
        output = Dense(1, activation='sigmoid')(feature)
        model = Model(input, outputs=output)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=1e-3),
                      metrics=['accuracy'])
        model.summary()
        self.model = model
        return model


class SKerasRegressor(RegressorMixin, ScikitKerasMixin, KerasRegressor):
    def __call__(self, n_input, *args, **kwargs):
        input = Input(shape=(n_input,))
        feature = self.bottleneck(input)
        output = Dense(1)(feature)
        model = Model(input, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=1e-3))
        model.summary()
        self.model = model
        return model
