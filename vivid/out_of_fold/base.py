import copy
import json
import os
from typing import List, Union, Callable, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna import Study
from optuna.trial import Trial
from sklearn.base import is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _BaseScorer
from sklearn.model_selection import KFold
from sklearn.model_selection import check_cv

from vivid.core import AbstractFeature
from vivid.env import Settings
from vivid.metrics import binary_metrics, upper_accuracy, regression_metrics
from vivid.sklearn_extend import PrePostProcessModel
from vivid.utils import timer


def create_default_cv():
    return KFold(n_splits=Settings.N_FOLDS, shuffle=True, random_state=Settings.RANDOM_SEED)


class BaseOutOfFoldFeature(AbstractFeature):
    """Base class that creates Out of Fold features for input data
    K-Fold CV is performed at the time of train to create K number of models.
    Returns the average of those predicted values ​​when testing

    The parameters used for learning are determined by the class variable `initial_params`.
    If you want to use a different variable for each instance, pass the parameter you want to add to `add_init_param`.
    """
    initial_params = {}
    model_class = None
    _serialize_filaname = 'fitted_models.joblib'

    def __init__(self, name, parent=None, cv=None, groups=None, sample_weight=None,
                 add_init_param=None, root_dir=None, verbose=1):
        """

        Args:
            name: Model name. Recommended to use a unique string throughout the same project.
            parent: parent feature instance.
            cv: Kfold instance or Number or None.
                If Set None, use default cv strategy.
            groups:
                Groups which use in group-k-fold
                only valid if you set Group K-Fold on cv.
            sample_weight:
                sample weight. shape = (n_train,)
            add_init_param: additional init params. class attribute `init_params` are updated by it.
            root_dir:
            verbose:
        """
        self.verbose = verbose

        if cv is None:
            cv = create_default_cv()
        self.cv = cv
        self._checked_cv = None
        self.groups = groups
        self.sample_weight = sample_weight

        self.is_regression_model = is_regressor(self.model_class())

        self._initial_params = copy.deepcopy(self.initial_params)

        if add_init_param:
            self._initial_params.update(add_init_param)

        super(BaseOutOfFoldFeature, self).__init__(name, parent, root_dir=root_dir)
        self.logger.info(self.name)
        self.finish_fit = False

    @property
    def serializer_path(self):
        if self.is_recording:
            return os.path.join(self.output_dir, self._serialize_filaname)
        return None

    @property
    def num_cv(self):
        if self._checked_cv:
            return self._checked_cv.n_splits
        return None

    def load_best_models(self):
        if self.output_dir is None:
            raise NotFittedError('Feature run without recording. Must Set Output Dir. ')

        if not os.path.exists(self.serializer_path):
            raise NotFittedError('Model Serialized file {} not found.'.format(self.serializer_path) +
                                 'Run fit before load model.')
        param_list = joblib.load(self.serializer_path)
        models = []
        for params in param_list:
            model = self.create_model({}, prepend_name=params['prepend_name'], recording=True)
            model.load_trained_model()
            models.append(model)
        return models

    def save_best_models(self, best_models):
        joblib.dump(best_models, self.serializer_path)

    def get_fold_splitting(self, X, y):
        if self._checked_cv is None:
            self._checked_cv = check_cv(self.cv, y, classifier=self.model_class)
        return self._checked_cv.split(X, y, self.groups)

    def _predict_trained_models(self, test_df: pd.DataFrame):
        if not self.finish_fit:
            models = self.load_best_models()
        else:
            models = self.fitted_models

        if self.is_regression_model:
            kfold_predicts = [model.predict(test_df.values) for model in models]
        else:
            kfold_predicts = [model.predict(test_df.values, prob=True)[:, 1] for model in models]
        preds = np.asarray(kfold_predicts).mean(axis=0)
        df = pd.DataFrame(preds.T, columns=[str(self)])
        return df

    def generate_default_model_parameter(self, X, y) -> dict:
        """
        generate model init parameter. It be shared with all Fold.
        If you like to change parameter or explore more best parameter, override this method.

        Args:
            X: input feature. shape = (n_train, n_features).
            y: target. shape = (n_train,)

        Returns:
            model parameter dict
        """
        return self._initial_params

    def get_params_on_each_fold(self, default_params: dict, indexes_set: Tuple[np.ndarray, np.ndarray]) -> dict:
        """
        generate model init parameter (pass to sklearn classifier/regressor constructor) on each fold

        Args:
            default_params: default model parameter.
            indexes_set: index set (idx_train, idx_valid).

        Returns:
            init parameter
        """
        params = copy.deepcopy(default_params)
        return params

    def get_fit_params_on_each_fold(self, model_params: dict,
                                    training_set: Tuple[np.ndarray, np.ndarray],
                                    validation_set: Tuple[np.ndarray, np.ndarray],
                                    indexes_set: Tuple[np.ndarray, np.ndarray]) -> dict:
        """
        generate fit params (i.e. kwrgs in `clf.fit(X, y, **kwrgs)`) on each fold.
        By default, set sample_weight when sample_weight is passed to the constructor.

        Args:
            model_params: model parameters. It pass to model constructor.
            validation_set: validation (X_valid, y_valid) dataset. tuple of numpy array.
            indexes_set: index set (idx_train, idx_valid). Use when set sample_weight on each fold.

        Returns:
            fit parameter dict
        """
        params = {}
        if self.sample_weight is not None:
            params['sample_weight'] = self.sample_weight[indexes_set[0]]
        return params

    def call(self, df_source: pd.DataFrame, y=None, test=False) -> pd.DataFrame:
        if test:
            return self._predict_trained_models(df_source)

        X, y = df_source.values, y
        default_params = self.generate_default_model_parameter(X, y)

        models, oof = self.run_oof_train(X, y, default_params)
        self.fitted_models = models

        self.finish_fit = True
        self.after_kfold_fitting(df_source, y, oof)
        oof_df = pd.DataFrame(oof, columns=[str(self)])
        return oof_df

    def run_oof_train(self, X, y, default_params) -> ([List[PrePostProcessModel], np.ndarray]):
        """
        main training loop.

        Args:
            X: input feature. shape = (n_samples, n_features)
            y: target. shape = (n_samples, n_classes)
            default_params: default model parameter. pass to model constructor (not fit)
                If you change fit parameter like `eval_metric`, override get_fit_params_on_each_fold.

        Returns:
            list of fitted models and out-of-fold numpy array.
        """
        oof = np.zeros_like(y, dtype=np.float32)
        splits = self.get_fold_splitting(X, y)
        models = []

        self.logger.info('CV: {}'.format(str(self._checked_cv)))

        for i, (idx_train, idx_valid) in enumerate(splits):
            self.logger.info('start k-fold: {}/{}'.format(i + 1, self.num_cv))

            X_i, y_i = X[idx_train], y[idx_train]
            X_valid, y_valid = X[idx_valid], y[idx_valid]

            with timer(self.logger, format_str='Fold: {}/{}'.format(i + 1, self.num_cv) + ' {:.1f}[s]'):
                clf = self._fit_model(X_i, y_i,
                                      default_params=default_params,
                                      validation_set=(X_valid, y_valid),
                                      indexes_set=(idx_train, idx_valid),
                                      prepend_name=i,
                                      recording=True)

            if self.is_regression_model:
                pred_i = clf.predict(X_valid).reshape(-1)
            else:
                pred_i = clf.predict(X_valid, prob=True)[:, 1]

            oof[idx_valid] = pred_i
            models.append(clf)
        return models, oof

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

    def _fit_model(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   default_params: dict,
                   validation_set: tuple,
                   indexes_set: tuple,
                   prepend_name=None,
                   recording=False) -> PrePostProcessModel:
        """
        in model_params, add scaling parameters for target / input (ex. target_scaling = False)

        Args:
            X: training feature. numpy array. shape = (n_train, n_features)
            y: target. shape = (n_train, n_classes)
            default_params(dict): parameters pass into model constructor
            validation_set:
            indexes_set:
            prepend_name: prepend name (use on save model)
            recording: If True, save trained model to local storage.

        Returns:
            trained model
        """
        model_params = self.get_params_on_each_fold(default_params, indexes_set)
        model = self.create_model(model_params, prepend_name=str(prepend_name), recording=recording)

        # MEMO: validation data are not transform so validation score is invalid (in boosting model, eval_set)
        model._before_fit(X, y)
        x_valid, y_valid = validation_set
        x_valid = model.input_transformer.transform(x_valid)
        y_valid = model.target_transformer.transform(y_valid)

        fit_params = self.get_fit_params_on_each_fold(model_params,
                                                      training_set=(X, y),
                                                      validation_set=(x_valid, y_valid),
                                                      indexes_set=indexes_set)
        if fit_params is None:
            fit_params = {}
        model.fit(X, y, **fit_params)
        return model

    def after_kfold_fitting(self, df_source, y, predict):
        try:
            self.show_metrics(y, predict)
        except Exception as e:
            self.logger.warn(e)

        if not self.is_recording:
            return

        model_outputs = []
        for m in self.fitted_models:
            model_outputs.append(m.get_params(deep=False))
        joblib.dump(model_outputs, self.serializer_path)

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
    optuna_jobs = -1  # optuna parallels
    SCORING_STRATEGY_CHOICES = ['fold', 'whole']  # choice of scoring strategy

    def __init__(self, n_trials=200, scoring_strategy='fold', scoring: Union[str, Callable] = 'default',
                 **kwargs):
        """
        Optuna Optimization Model Feature

        Args:
            n_trials: total number of trials.
            scoring_strategy:
                out-of-fold scoring strategy.
                If set as `"fold"`, the score are calculated each by fold and use mean of them for optimization.
                If set as `"whole"`, the score is calculated whole data.
            scoring: scoring method. String or Scoring Object (scoring object must be satisfied check_scoring validation)
            **kwargs: pass to superclass
        """
        super(BaseOptunaOutOfFoldFeature, self).__init__(**kwargs)
        self.study = None  # type: Union[Study, None]
        self.n_trails = n_trials

        if scoring_strategy not in self.SCORING_STRATEGY_CHOICES:
            raise ValueError('`scoring_strategy` must be in {}'.format(','.join(self.SCORING_STRATEGY_CHOICES)))
        self.scoring_strategy = scoring_strategy
        if scoring == 'default':
            if self.is_regression_model:
                scoring = 'neg_root_mean_squared_error'
            else:
                scoring = 'roc_auc'
        scoring = check_scoring(self.model_class, scoring=scoring, allow_none=False)
        self.scoring_method = scoring  # type: _BaseScorer

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

    def calculate_score(self, y_true, y_pred, sample_weight) -> float:
        if sample_weight is not None:
            return self.scoring_method._sign * self.scoring_method._score_func(y_true,
                                                                               y_pred,
                                                                               sample_weight=sample_weight,
                                                                               **self.scoring_method._kwargs)
        else:
            return self.scoring_method._sign * self.scoring_method._score_func(y_true,
                                                                               y_pred,
                                                                               **self.scoring_method._kwargs)

    def get_objective(self, trial: Trial, X, y) -> float:
        """
        calculate objective value for each trial

        Args:
            trial:
            X:
            y:

        Returns:
            score of this trial
        """
        params = self.generate_try_parameter(trial)
        models, oof = self.run_oof_train(X, y, default_params=params)

        scores = []
        sample_weight = self.sample_weight
        if self.scoring_strategy == 'whole':
            score = self.calculate_score(y, oof, sample_weight)
        elif self.scoring_strategy == 'fold':
            for idx_train, idx_valid in self.get_fold_splitting(X, y):
                sample_weight_i = sample_weight[idx_valid] if sample_weight is not None else None
                score_i = self.calculate_score(y[idx_valid], oof[idx_valid], sample_weight=sample_weight_i)
                scores.append(score_i)
            score = np.mean(scores)
        else:
            raise ValueError()
        return score

    def generate_default_model_parameter(self, X, y) -> dict:
        """
        The main roop which explore model parameter by optuna.

        Args:
            X:
            y:

        Returns:
            best model parameter
        """
        self.logger.info('start optimize by optuna')

        self.study = optuna.study.create_study()
        objective = lambda trial: self.get_objective(trial, X, y)

        # Stop model logging while optuna optimization
        self.logger.disabled = True
        self.study.optimize(objective, n_trials=self.n_trails, n_jobs=self.optuna_jobs)
        self.logger.disabled = False

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
