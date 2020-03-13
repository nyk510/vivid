import copy
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna import Study
from optuna.trial import Trial
from sklearn.base import is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

from vivid.core import AbstractFeature
from vivid.env import Settings
from vivid.metrics import binary_metrics, upper_accuracy, regression_metrics, root_mean_squared_error
from vivid.sklearn_extend import PrePostProcessModel
from vivid.utils import timer

FOLD_CLASSES = {
    'kfold': KFold,
    'stratified': StratifiedKFold,
    'group': GroupKFold
}


class BaseOutOfFoldFeature(AbstractFeature):
    """
    入力データに対して Out of Fold 特徴量を作成する基底クラス
    train 時に K-Fold CV を行って K 数だけモデルを作成し
    テスト時にはそれらの予測値の平均値を返します

    学習に用いるパラメータはクラス変数 `initial_params` で決まります.
    インスタンスごとに別の変数を使いたい場合, `add_init_param` に追加したいパラメータを渡してください.
    """

    num_cv = Settings.N_FOLDS

    fold_seed = Settings.RANDOM_SEED
    fold_group_key = 'id'
    fold_type = 'default'

    initial_params = {}
    model_class = None
    _serialize_filaname = 'fitted_models.joblib'

    def __init__(self, name, parent=None,
                 add_init_param=None, root_dir=None, verbose=1):
        """

        Args:
            parent:
            name:
            add_param_dist(None | dict):
                None 以外の dict が与えられると class 変数として定義されている
                探索空間 `param_dist` をこの値で更新する (dict.update)
            num_cv_in_search(int):
                各パラメータ探索での CV の数
            num_search(int):
                各パラメータ探索を最大で何回まで行うか
            verbose:
        """
        self.verbose = verbose

        self.is_regression_model = is_regressor(self.model_class())

        self._initial_params = copy.deepcopy(self.initial_params)

        if add_init_param:
            self._initial_params.update(add_init_param)

        super(BaseOutOfFoldFeature, self).__init__(name, parent, root_dir=root_dir)
        self.fitted_models = []
        self.logger.info(self.name)
        self.finish_fit = False

    @property
    def serializer_path(self):
        if self.is_recording:
            return os.path.join(self.output_dir, self._serialize_filaname)
        return None

    def load_best_models(self):
        if self.output_dir is None:
            raise NotFittedError('Feature run without recording. Must Set Output Dir. ')

        if os.path.exists(self.serializer_path):
            return joblib.load(self.serializer_path)

        raise NotFittedError('Model Serialized file {} not found.'.format(self.serializer_path) +
                             'Run fit before load model.')

    def save_best_models(self, best_models):
        joblib.dump(best_models, self.serializer_path)

    def get_fold_splitting(self, X, y, groups=None):
        """
        学習用データと target の配列から各CVにたいする index 集合を生成する method

        Args:
            X: 学習用データの numpy array
            y: target numpy array
            groups: GroupKFold の場合に使用する group

        Returns:
            List([np.ndarray, np.ndarray])
        """
        if len(X) != len(y):
            raise ValueError(f'X and y must has same samples. actually: {len(X)} {len(y)}')

        fold_name = self.fold_type
        if fold_name == 'default':
            if self.is_regression_model:
                fold_name = 'kfold'
            else:
                fold_name = 'stratified'
        fold_class = FOLD_CLASSES.get(fold_name, None)
        if fold_class is None:
            raise ValueError(f'Invalid Fold Name: {self.fold_type}')

        if fold_class is GroupKFold:
            return fold_class(n_splits=self.num_cv).split(X, y, groups)

        return fold_class(n_splits=self.num_cv, random_state=self.fold_seed, shuffle=True).split(X, y)

    def get_folds(self, X, y, groups):
        splits = self.get_fold_splitting(X, y, groups)

        for idx_train, idx_valid in splits:
            x_i, y_i = X[idx_train], y[idx_train]
            x_valid, y_valid = X[idx_valid], y[idx_valid]
            yield (x_i, y_i), (x_valid, y_valid), (idx_train, idx_valid)

    def _predict_trained_models(self, df_test):
        if not self.finish_fit:
            models = self.load_best_models()
        else:
            models = self.fitted_models

        if self.is_regression_model:
            kfold_predicts = [model.predict(df_test.values) for model in models]
        else:
            kfold_predicts = [model.predict(df_test.values, prob=True)[:, 1] for model in models]
        preds = np.asarray(kfold_predicts).mean(axis=0)
        df = pd.DataFrame(preds.T, columns=[str(self)])
        return df

    def get_best_model_parameters(self, X, y):
        """
        モデルの学習で使う parameter を取得します.
        特定の基準にしたがってパラメータを選定したい場合, 例えばグリッドサーチを使ってパラメータを選ぶなどの
        場合にはこのメソッドをオーバーライドしてください

        Args:
            X:
            y:

        Returns:

        """
        return self._initial_params

    def call(self, df_source, y=None, test=False):
        if test:
            return self._predict_trained_models(df_source)

        x_train, y_train = df_source.values, y

        # Note: float32 にしないと dtype が int になり, 予測確率を代入しても 0 のままになるので注意
        pred_train = np.zeros_like(y_train, dtype=np.float32)
        params = self.get_best_model_parameters(x_train, y_train)

        for i, ((x_i, y_i), (x_valid, y_valid), (_, idx_valid)) in enumerate(
            self.get_folds(x_train, y_train, groups=None)):  # [NOTE] Group KFold is not Supported yet
            self.logger.info('start k-fold: {}/{}'.format(i + 1, self.num_cv))

            with timer(self.logger, format_str='Fold: {}/{}'.format(i + 1, self.num_cv) + ' {:.1f}[s]'):
                clf = self.fit_model(x_i, y_i, params, x_valid=x_valid, y_valid=y_valid, cv=i)

                if self.is_regression_model:
                    pred_i = clf.predict(x_valid).reshape(-1)
                else:
                    pred_i = clf.predict(x_valid, prob=True)[:, 1]

                pred_train[idx_valid] = pred_i
                self.fitted_models.append(clf)

        self.finish_fit = True
        self.after_kfold_fitting(df_source, y, pred_train)
        df_train = pd.DataFrame(pred_train, columns=[str(self)])
        return df_train

    def create_model(self, model_params, prepend_name, recording=False) -> PrePostProcessModel:
        target_logscale = model_params.pop('target_logscale', False)
        target_scaling = model_params.pop('target_scaling', None)
        input_logscale = model_params.pop('input_logscale', False)
        input_scaling = model_params.pop('input_scaling', None)

        model = PrePostProcessModel(model_class=self.model_class,
                                    model_params=model_params,
                                    target_logscale=target_logscale,
                                    target_scaling=target_scaling,
                                    input_logscale=input_logscale,
                                    input_scaling=input_scaling,
                                    output_dir=self.output_dir if recording else None,
                                    prepend_name=prepend_name,
                                    verbose=self.verbose,
                                    logger=self.logger)
        return model

    def fit_model(self, X: np.ndarray, y: np.ndarray,
                  model_params: dict, x_valid, y_valid, cv=None) -> PrePostProcessModel:
        """
        `PrePostProcessModel` を学習させます.
        recordable_model_params には target/input に関するスケーリングを含めたパラメータ情報を与えてください

        Args:
            X: 特徴量
            y: ターゲット変数
            model_params(dict): モデルに渡すパラメータの dict.
            x_valid:
            y_valid:
            cv:
                None 以外が与えられた時
                    CVのIndexと解釈し
                    文字列評価の結果を prefix としてモデルを学習する
                None のとき
                    output dir が存在していてもモデルを保存せずに学習します

        Returns:
            trained prepost process model
        """
        recording = cv is not None
        model = self.create_model(model_params, prepend_name=str(cv), recording=recording)
        model.fit(X, y)
        return model

    def after_kfold_fitting(self, df_source, y, predict):
        try:
            self.show_metrics(y, predict)
        except Exception as e:
            self.logger.warn(e)

        if self.is_recording:
            self.save_best_models(self.fitted_models)

    def show_metrics(self, y, prob_predict):
        if self.is_regression_model:
            metric_df = regression_metrics(y, prob_predict)
        else:
            metric_df = binary_metrics(y, prob_predict)
        self.logger.info(metric_df)

        if self.is_recording:
            metric_df.to_csv(os.path.join(self.output_dir, 'metrics.csv'))

        if not self.is_regression_model:
            self._generate_binary_result_graph(y, prob_predict)

    def _generate_binary_result_graph(self, y, prob_predict):
        df_upper_acc = upper_accuracy(y, prob_predict)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        df_upper_acc.plot(x='ratio', y='accuracy', ax=ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(min(df_upper_acc.accuracy) - .05, 1)
        ax.set_title('Upper Accuracy')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'upper_accuray.png'), dpi=150)
        df_upper_acc.to_csv(os.path.join(self.output_dir, 'upper_accuracy.csv'), index=False)
        plt.close(fig)


class BaseOptunaOutOfFoldFeature(BaseOutOfFoldFeature):
    """
    Model Based CV Feature with optuna tuning
    """
    optuna_jobs = -1

    def __init__(self, n_trials=200, **kwargs):
        super(BaseOptunaOutOfFoldFeature, self).__init__(**kwargs)
        self.study = None  # type: Study
        self.n_trails = n_trials

    def generate_model_class_try_params(self, trial: Trial) -> dict:
        """
        model の init で渡すパラメータの探索範囲を取得する method

        NOTE:
            この method で変えられるのはあくまで init に渡す引数だけです.
            より複雑な条件を変更する際には `get_object` を override することを検討して下さい.

        Args:
            trial(Trial):

        Returns(dict):

        """
        return {}

    def get_score_method(self):
        if self.is_regression_model:
            return root_mean_squared_error
        return roc_auc_score

    def evaluate_predict(self, y_true, model, x_valid) -> float:
        """
        calculate model quality in each fold set.

        デフォルト設定の時 regression model では rmse, classification のとき auc を score に用います

        Args:
            y_true(np.array): ground truth value.
                shape = (n_classes,) or (n_samples, n_classes,)
            model(PrePostProcessModel): trained model instance
            x_valid(np.array): validation input data. shape = (n_samples, n_features,)

        Returns(float):
            score value.
            [Note] Since optuna cant deal with maximum objective, the more minimum score is better.
            for example, auc score is better as the score bigger, so must return minus auc.
        """
        score_method = self.get_score_method()
        if self.is_regression_model:
            y_pred = model.predict(x_valid)
            return score_method(y_true, y_pred)

        y_pred = model.predict(x_valid, prob=True)
        # shape = (n_samples, n_classes) なので第一次元だけつかう
        score = score_method(y_true, y_pred[:, 1])
        return -score

    def generate_try_parameter(self, trial: Trial) -> dict:
        """

        Args:
            trial(Trial):

        Returns:

        """
        model_params = copy.deepcopy(self._initial_params)
        add_model_params = self.generate_model_class_try_params(trial)
        model_params.update(add_model_params)
        return model_params

    def get_objective(self, trial, X, y) -> float:
        """
        trial ごとの objective の値を返す関数

        Args:
            trial:
            X:
            y:

        Returns:
        """
        score = .0
        params = self.generate_try_parameter(trial)

        for (x_train, y_train), (x_valid, y_valid), _ in self.get_folds(X, y, groups=None):
            clf = self.fit_model(x_train, y_train,
                                 model_params=params, x_valid=x_valid, y_valid=y_valid, cv=None)
            score += self.evaluate_predict(y_valid, clf, x_valid)
        return score / self.num_cv

    def get_best_model_parameters(self, X, y) -> dict:
        """
        PrePostProcessModel に渡す最適なパラメータを Optuna Study を用いて取得する
        実験 Study の実体の実装は `get_objective` に記述してください

        Args:
            X:
            y:

        Returns(dict):
            最適なパラメータ
            `target_logscale` など変換に関するものもこの dict の中に含めてください
        """
        self.logger.info('start optimize by optuna')

        self.study = optuna.study.create_study()
        objective = lambda trial: self.get_objective(trial, X, y)
        self.study.optimize(objective, n_trials=self.n_trails, n_jobs=self.optuna_jobs)
        self.logger.info('best trial params: {}'.format(self.study.best_params))
        self.logger.info('best value: {}'.format(self.study.best_value))

        best_params = copy.deepcopy(self._initial_params)
        best_params.update(self.study.best_params)
        self.logger.info('best model paras: {}'.format(best_params))

        if self.is_recording:
            self.study.trials_dataframe().to_csv(os.path.join(self.output_dir, 'study_log.csv'))

            with open(os.path.join(self.output_dir, 'best_params.json'), 'w') as f:
                json.dump(best_params, f, indent=4)
            with open(os.path.join(self.output_dir, 'best_trial_params.json'), 'w') as f:
                json.dump(self.study.best_params, f, indent=4)

        return best_params
