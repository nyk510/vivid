import copy
import os
from collections.abc import Iterable
from typing import List, Union, Callable, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna import Study
from optuna.trial import Trial
from sklearn.base import is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _BaseScorer, SCORERS
from sklearn.model_selection import KFold, check_cv
from tabulate import tabulate

from vivid.core import AbstractFeature
from vivid.env import Settings
from vivid.metrics import binary_metrics, regression_metrics
from vivid.sklearn_extend import PrePostProcessModel
from vivid.utils import timer
from vivid.visualize import visualize_feature_importance, visualize_roc_auc_curve, visualize_pr_curve, \
    visualize_distributions, NotSupportedError


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
    _parameter_path = 'model_parameters.joblib'

    def __init__(self, name, parent=None, cv=None, groups=None, sample_weight=None,
                 add_init_param=None, root_dir=None):
        """

        Args:
            name: Model name. Recommended to use a unique string throughout the same project.
            parent: parent feature instance.
            cv: Kfold instance or Number or Iterable or None.
                If Set None, use default_loader cv strategy.
            groups:
                Groups which use in group-k-fold
                only valid if you set Group K-Fold on cv.
            sample_weight:
                sample weight. shape = (n_train,)
            add_init_param:
                additional init params. class attribute `init_params` are updated by it.
            root_dir:
                root_dir, pass to `AbstractFeature`.
        """

        if cv is None:
            cv = create_default_cv()
        self.cv = cv
        self._checked_cv = None
        self.groups = groups
        self.sample_weight = sample_weight
        self._initial_params = copy.deepcopy(self.initial_params)

        if add_init_param:
            self._initial_params.update(add_init_param)

        super(BaseOutOfFoldFeature, self).__init__(name, parent, root_dir=root_dir)
        self.logger.info(self.name)
        self.is_train_finished = False

    @property
    def is_regression_model(self):
        """whether the `model_class` instance is regression model or Not"""
        return is_regressor(self.model_class())

    @property
    def model_param_path(self):
        """If it is recording context, return the path to the array of k-fold model parameters"""
        if self.is_recording:
            return os.path.join(self.output_dir, self._parameter_path)
        return None

    @property
    def num_cv(self):
        if self._checked_cv:
            return self._checked_cv.n_splits
        return None

    def load_best_models(self) -> List[PrePostProcessModel]:
        """load fitted models from local model parameters."""
        if self.output_dir is None:
            raise NotFittedError('Feature run without recording. Must Set Output Dir. ')

        if not os.path.exists(self.model_param_path):
            raise NotFittedError('Model Serialized file {} not found.'.format(self.model_param_path) +
                                 'Run fit before load model.')
        param_list = joblib.load(self.model_param_path)
        models = []
        for params in param_list:
            model = self.create_model({}, output_dir=params.get('output_dir'))
            model.load_trained_model()
            models.append(model)
        return models

    def get_fold_splitting(self, X, y) -> Iterable:
        # If cv is iterable obj, convert to list and return
        if isinstance(self.cv, Iterable):
            return list(self.cv)
        if self._checked_cv is None:
            self._checked_cv = check_cv(self.cv, y, classifier=self.model_class)
        return list(self._checked_cv.split(X, y, self.groups))

    def _predict_trained_models(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_train_finished:
            models = self.load_best_models()
        else:
            models = self._fitted_models

        if self.is_regression_model:
            fold_predicts = [model.predict(test_df.values) for model in models]
        else:
            fold_predicts = [model.predict(test_df.values, prob=True)[:, 1] for model in models]
        preds = np.asarray(fold_predicts).mean(axis=0)
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

    def get_model_params_on_each_fold(self, default_params: dict, indexes_set: Tuple[np.ndarray, np.ndarray]) -> dict:
        """
        Generate model init parameter (pass to sklearn classifier/regressor constructor) on each fold
        it is pass to `model_class`` constructor (i.e. __init__(**params))

        Args:
            default_params: default parameter.
            indexes_set: train and validation index list for each fold.

        Returns:
            parameter pass to model class
        """
        params = copy.deepcopy(default_params)
        return params

    def get_fit_params_on_each_fold(self, model_params: dict,
                                    training_set: Tuple[np.ndarray, np.ndarray],
                                    validation_set: Tuple[np.ndarray, np.ndarray],
                                    indexes_set: Tuple[np.ndarray, np.ndarray]) -> dict:
        """
        generate fit params (i.e. kwrgs in `clf.fit(X, y_true, **kwrgs)`) on each fold.
        By default_loader, set sample_weight when sample_weight is passed to the constructor.

        Args:
            model_params: model parameters. It have been pass to model constructor.
            training_set: training (X_train, y_train) dataset. tuple of numpy array.
            validation_set: validation (X_valid, y_valid) dataset. tuple of numpy array.
            indexes_set: train and validation index list for each fold.

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

        with self.exp_backend.mark_time(prefix='train_'):
            models, oof = self.run_oof_train(X, y, default_params)

        self._fitted_models = models

        self.is_train_finished = True
        oof_df = pd.DataFrame(oof, columns=[str(self)])
        return oof_df

    def run_oof_train(self, X, y, default_params,
                      n_fold: Union[int, None] = None,
                      silent=False) -> ([List[PrePostProcessModel], np.ndarray]):
        """
        main training loop.

        Args:
            X:
                training array.
            y:
                target array
            default_params:
                model parameter using by default. pass to model constructor (not fit)
                If you change fit parameter like `eval_metric`, override get_fit_params_on_each_fold.
            n_fold:
                Number of fold to fit. If set None, learn for all folds.
                If set number, stop fit model reach to the value.
                    * if n_fold = None, run all folds
                    * if n_fold = 1, stop one fold.
                    * if n_fold > num_cv, run all folds
                    * if n_fold <= 0, no fold run, return empty list and zero vector out-of-fold
        Returns:
            list of fitted models and out-of-fold numpy array.
        """
        oof = np.zeros_like(y, dtype=np.float32)
        splits = self.get_fold_splitting(X, y)
        models = []

        for i, (idx_train, idx_valid) in enumerate(splits):
            if n_fold is not None and i >= max(0, n_fold):
                self.logger.info(f'Stop K-Fold at {i}')
                break

            self.logger.info('start k-fold: {}/{}'.format(i + 1, self.num_cv))

            X_i, y_i = X[idx_train], y[idx_train]
            X_valid, y_valid = X[idx_valid], y[idx_valid]

            with timer(self.logger, format_str='Fold: {}/{}'.format(i + 1, self.num_cv) + ' {:.1f}[s]'):
                output_i = None if not self.is_recording or silent else os.path.join(self.output_dir, f'fold_{i:02d}')
                clf = self._fit_model(X_i, y_i,
                                      default_params=default_params,
                                      validation_set=(X_valid, y_valid),
                                      indexes_set=(idx_train, idx_valid),
                                      output_dir=output_i)

            if self.is_regression_model:
                pred_i = clf.predict(X_valid).reshape(-1)
            else:
                pred_i = clf.predict(X_valid, prob=True)[:, 1]

            oof[idx_valid] = pred_i
            models.append(clf)

        return models, oof

    def create_model(self, model_params, output_dir=None) -> PrePostProcessModel:
        """
        create new `PrePostProcessModel` instance that has our `model_class`.

        Args:
            model_params:
                parameters pass to `PrePostProcessModel` constructor (i.e. __init__ method)
            output_dir:
                created pre-post process model output.

        Returns:
            new PrePoseProcessModel
        """
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
                                    output_dir=output_dir,
                                    logger=self.logger)
        return model

    def _fit_model(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   default_params: dict,
                   validation_set: tuple,
                   indexes_set: tuple,
                   output_dir=None) -> PrePostProcessModel:
        """
        train a new model class.

        Notes:
            in model_params, add scaling parameters for target / input (ex. target_scaling = False)

        Args:
            X: training feature. numpy array. shape = (n_train, n_features)
            y: target. shape = (n_train, n_classes)
            default_params: parameters pass into model constructor
            validation_set:
            indexes_set:
            output_dir: prepend name (use on save model)

        Returns:
            trained model
        """
        model_params = self.get_model_params_on_each_fold(default_params, indexes_set)
        model = self.create_model(model_params, output_dir=output_dir)

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

    def post_fit(self,
                 input_df: pd.DataFrame, parent_output_df: pd.DataFrame,
                 out_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        if self.is_recording:
            self.save_model_parameters(self._fitted_models)
        return super(BaseOutOfFoldFeature, self).post_fit(input_df, parent_output_df, out_df, y)

    def save_model_parameters(self, best_models: List[PrePostProcessModel]) -> List[dict]:
        model_parameters = []
        use_keys = ['model_params', 'output_dir']
        for m in best_models:
            param_i = m.get_params(deep=False)
            add_param_i = {}
            for k in use_keys:
                add_param_i[k] = param_i[k]

            model_parameters.append(add_param_i)

        joblib.dump(model_parameters, self.model_param_path)
        return model_parameters


class BaseOptunaOutOfFoldFeature(BaseOutOfFoldFeature):
    """
    Model Based CV Feature with optuna tuning
    """
    optuna_jobs = -1  # optuna parallels
    SCORING_STRATEGY_CHOICES = ['fold', 'whole']  # choice of scoring strategy

    def __init__(self,
                 n_trials=200,
                 scoring_strategy='fold',
                 scoring: Union[str, Callable, None] = None,
                 **kwargs):
        """
        Optuna Optimization Model Feature

        Args:
            n_trials:
                total number of trials.
            scoring_strategy:
                out-of-fold scoring strategy.
                If set as `"fold"`, the score are calculated each by fold and use mean of them for optimization.
                If set as `"whole"`, the score is calculated whole data.
            scoring:
                scoring method. String or Scoring Object
                 (scoring obj must be satisfied check_scoring validation)
            **kwargs:
                pass to superclass
        """
        super(BaseOptunaOutOfFoldFeature, self).__init__(**kwargs)
        self.study = None  # type: Union[Study, None]
        self.n_trails = n_trials

        if scoring_strategy not in self.SCORING_STRATEGY_CHOICES:
            raise ValueError('`scoring_strategy` must be in {}'.format(','.join(self.SCORING_STRATEGY_CHOICES)))
        self.scoring_strategy = scoring_strategy
        if scoring is None:
            if self.is_regression_model:
                scoring = 'neg_root_mean_squared_error'
            else:
                scoring = 'roc_auc'

        try:
            scoring = check_scoring(self.model_class, scoring=scoring, allow_none=False)
        except ValueError as e:
            s = f'Invalid scoring argument: {scoring}. You can select scoring method from pre-defineds as follow\n'
            s += ', '.join(SCORERS.keys())
            raise ValueError(s)
        self.scoring_method = scoring  # type: _BaseScorer

        self.exp_backend.mark('n_trials', self.n_trails)
        self.exp_backend.mark('scoring_strategy', self.scoring_strategy)
        self.exp_backend.mark('scoring', str(self.scoring_method))

    def generate_model_class_try_params(self, trial: Trial) -> dict:
        """method to get the range of parameters to look for in the model's init
        The created value overrides the class variable and becomes the initial value of the model.
        More details, see `generate_try_parameter` as below.

        Notes:
            This method only changes the arguments you pass to init:
            Consider overriding `get_object` if you want to change more complex conditions.

        Args:
            trial(Trial):
                current trial object. create parameter using it.

        Returns:
            generated model parameter
        """
        raise NotImplementedError(
            'Must implement `generate_model_class_try_params`.'
            'This method is the most core method of this class, which creates '
            'the parameters to use in each optimization attempt.')

    def generate_try_parameter(self, trial: Trial) -> dict:
        """
        generate parameters used by out of fold training.

        Args:
            trial(Trial):
                current trial object. create parameter using it.

        Returns:
            parameters used by train out-of-fold in optimizing
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
                optuna Trial Object
            X:
                training array.
            y:
                target array

        Returns:
            score of this trial
        """
        params = self.generate_try_parameter(trial)
        models, oof = self.run_oof_train(X, y, default_params=params, silent=True)

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
        The main loop which explore model parameter by optuna.

        Args:
            X:
            y:

        Returns:
            tuned parameter
        """
        self.logger.info('start optimize by optuna')

        self.study = optuna.study.create_study(direction='maximize')
        objective = lambda trial: self.get_objective(trial, X, y)

        # Stop model logging while optuna optimization
        with self.exp_backend.mark_time('optuna_') as exp:
            with self.set_silent():
                self.study.optimize(objective, n_trials=self.n_trails, n_jobs=self.optuna_jobs)

        self.logger.info('best trial params: {}'.format(self.study.best_params))
        self.logger.info('best value: {}'.format(self.study.best_value))

        best_params = copy.deepcopy(self._initial_params)
        best_params.update(self.study.best_params)
        self.logger.info('best model paras: {}'.format(best_params))

        self.exp_backend.mark('optuna_best_value', self.study.best_value)
        self.exp_backend.save_object('study_log', self.study.trials_dataframe())
        self.exp_backend.save_object('best_params', best_params)
        self.exp_backend.save_object('best_trial_params', self.study.best_params)

        return best_params


class AbstractReport:
    def call(self, feature_instance: BaseOutOfFoldFeature, source_df: pd.DataFrame, y: np.ndarray,
             oof: np.ndarray) -> List:
        pass


class MetricReport(AbstractReport):
    """calculate predict score and logging"""

    def call(self, feature_instance: BaseOutOfFoldFeature, source_df: pd.DataFrame, y: np.ndarray, oof: np.ndarray):
        if feature_instance.is_regression_model:
            metric_df = regression_metrics(y, oof)
        else:
            metric_df = binary_metrics(y, oof)

        feature_instance.exp_backend.mark('train_metrics', metric_df['score'])

        s_metric = tabulate(metric_df.T, headers='keys')
        for s in s_metric.split('\n'):
            feature_instance.logger.info(s)

        return [
            ('metrics.csv', metric_df)
        ]


class FeatureImportanceReport(AbstractReport):
    """plot feature importance report"""

    def __init__(self, n_importance_plot=50):
        super(FeatureImportanceReport, self).__init__()
        self.n_importance_plot = n_importance_plot

    def call(self, feature_instance: BaseOutOfFoldFeature, source_df: pd.DataFrame, y: np.ndarray,
             oof: np.ndarray) -> List:

        if not hasattr(feature_instance, '_fitted_models'): return []
        try:
            fig, ax, importance_df = visualize_feature_importance(feature_instance._fitted_models,
                                                                  columns=source_df.columns,
                                                                  top_n=self.n_importance_plot,
                                                                  plot_type='boxen')
            return [
                ('feature_importance.csv', importance_df),
                ('importance.png', fig)
            ]
        except NotSupportedError:
            return []


class CurveFigureReport(AbstractReport):
    def __init__(self, visualizers=None):
        super(CurveFigureReport, self).__init__()
        if visualizers is None:
            visualizers = [
                visualize_pr_curve,
                visualize_roc_auc_curve,
                visualize_distributions
            ]
        self.visualizers = visualizers

    def call(self, feature_instance: BaseOutOfFoldFeature, source_df: pd.DataFrame, y: np.ndarray,
             oof: np.ndarray) -> List:
        if feature_instance.is_regression_model:
            return []

        outputs = []

        for vis in self.visualizers:
            try:
                fig, ax = vis(y, oof)
                name = vis.__name__.replace('visualize_', '')
                outputs.append(
                    (f'{name}.png', fig)
                )
            except Exception as e:
                feature_instance.logger.warning(f'Error has occurred: {vis}')
                feature_instance.logger.warning(e)
        return outputs


class ShowMetricMixin:
    """show model metrics using out-of-fold prediction"""
    n_importance_plot = 50

    def get_show_metric_generators(self) -> List[AbstractReport]:
        return [
            MetricReport(),
            FeatureImportanceReport(n_importance_plot=self.n_importance_plot),
            CurveFigureReport()
        ]

    def post_fit(self: Union['ShowMetricMixin', BaseOutOfFoldFeature],
                 input_df: pd.DataFrame, parent_output_df: pd.DataFrame, out_df, y):
        for func in self.get_show_metric_generators():
            try:
                outputs = func.call(feature_instance=self,
                                    source_df=parent_output_df,
                                    y=y,
                                    oof=out_df.values[:, 0])

                if not self.is_recording:
                    continue

                for filename, obj in outputs:
                    self.exp_backend.save_object(filename, obj)
            except Exception as e:
                self.logger.warning(f'Error has occurred: {func}')
                self.logger.warning(e)

        return super(ShowMetricMixin, self).post_fit(input_df, parent_output_df, out_df, y)


class GenericOutOfFoldFeature(ShowMetricMixin, BaseOutOfFoldFeature):
    pass


class GenericOutOfFoldOptunaFeature(ShowMetricMixin, BaseOptunaOutOfFoldFeature):
    pass


class EnsembleFeature(ShowMetricMixin, AbstractFeature):
    def __init__(self, agg='mean', **kwargs):
        super(EnsembleFeature, self).__init__(**kwargs)
        self.agg = agg

    @property
    def is_regression_model(self):
        """proxy to parent"""
        if isinstance(self._primary_parent, BaseOutOfFoldFeature):
            return self._primary_parent.is_regression_model
        return None

    def call(self, df_source: pd.DataFrame, y=None, test=False) -> pd.DataFrame:
        x = df_source.agg(self.agg, axis=1).values
        return pd.DataFrame(x, columns=[f'{self.name}_{self.agg}'])
