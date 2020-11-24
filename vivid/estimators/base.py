import copy
from typing import List, Tuple, Optional, Union, Callable

import numpy as np
import optuna
import pandas as pd
from optuna import Study
from optuna.trial import Trial
from sklearn.base import is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _BaseScorer, SCORERS
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder

from vivid.backends.experiments import ExperimentBackend
from vivid.core import AbstractEvaluation, BaseBlock, SimpleEvaluation
from vivid.sklearn_extend import PrePostProcessModel
from vivid.utils import get_logger
from .evaluations import FeatureImportanceReport, MetricReport, curve_figure_reports
from .evaluations import ConfusionMatrixReport

from vivid.metrics import binary_metrics, multiclass_metrics, regression_metrics
from .utils import to_pretty_lines

logger = get_logger(__name__)


class EstimatorMixin:
    is_estimator = True


def _get_default_model_evaluations(evaluations: Union[None, List[AbstractEvaluation]]):
    if isinstance(evaluations, list):
        return evaluations
    return [
        FeatureImportanceReport(),
        MetricReport(),
        *curve_figure_reports(),
        SimpleEvaluation(),
        ConfusionMatrixReport()
    ]


def _run_predict(model: PrePostProcessModel,
                 X,
                 is_regression: bool) -> np.ndarray:
    if is_regression:
        return model.predict(X).reshape(-1, 1)

    pred = model.predict(X, prob=True)
    n_classes = pred.shape[1]
    if n_classes > 2:
        return pred
    return pred[:, 1].reshape(-1, 1)


class MetaBlock(EstimatorMixin, BaseBlock):
    """Base class that creates Out of Fold features for input data
    K-Fold CV is performed at the time of fit to create K number of models.
    Returns the average of those predicted values ​​when testing

    The parameters used for learning are determined by the class variable `initial_params`.
    If you want to use a different variable for each instance, pass the parameter you want to add to `add_init_param`.
    """
    is_estimator = True
    initial_params = {}
    model_class = None

    def __init__(self,
                 name,
                 parent=None,
                 cv=None,
                 groups=None,
                 add_init_param: Union[None, dict] = None,
                 sample_weight=None,
                 evaluations: Union[None, List[AbstractEvaluation]] = None):
        """
        Args:
            name:
                Model name. Recommended to use a unique string throughout the same project.
            parent:
                parent_blocks feature instance.
            cv:
                Kfold instance or Number or Iterable or None.
                If Set None, use default_loader cv strategy.
            groups:
                Groups which use in group-k-fold
                only valid if you set Group K-Fold on cv.
            sample_weight:
                sample weight. shape = (n_train,)
            add_init_param:
                additional init params (None or dict).
                class attribute `init_params` are updated by it and pass to `model_class` constructor.
            evaluations:
                List of AbstractEvaluation subclass or None.
                学習 (fit) が終了したあと, このメソッドが EvaluationEnv を引数として呼び出されます.
        """

        self.cv = cv
        self.groups = groups
        self.sample_weight = sample_weight
        self._initial_params = copy.deepcopy(self.initial_params)
        if add_init_param:
            self._initial_params.update(add_init_param)
        self._output_dim = None

        super(MetaBlock, self).__init__(name, parent, evaluations=_get_default_model_evaluations(evaluations))

    def _to_hash(self) -> str:
        s = super(MetaBlock, self)._to_hash()
        parameters = {}
        parameters.update(self._initial_params)
        parameters.update({
            'cv': str(self.cv),
            'group': self.groups.shape if self.groups else None,
            'sample_weight': self.sample_weight.shape if self.sample_weight else None
        })
        for k, v in parameters.items():
            s += '{}\t{}'.format(k, v)
        return s

    @property
    def is_regression_model(self) -> bool:
        """whether the `model_class` instance is regression model or Not"""
        return is_regressor(self.model_class())

    def _check_has_fitted_models(self) -> bool:
        return hasattr(self, '_fitted_models')

    def _check_has_models_in_exp(self, experiment: ExperimentBackend) -> bool:
        try:
            output_dirs = experiment.get_marked().get('cv_dirs', None)
        except (FileNotFoundError, AttributeError):
            return False
        return output_dirs is not None

    def check_is_fitted(self, experiment: ExperimentBackend) -> bool:
        """学習済み (i.e. fit が呼び出されたかどうか) を判定します.
        experiment に学習済みモデルが保存されている場合には True を返します.
        """
        if self._check_has_fitted_models():
            return True

        return self._check_has_models_in_exp(experiment)

    def clear_fit_cache(self):
        del self._fitted_models
        import gc
        logger.info('[{}] clear cache models {}'.format(self.name, gc.collect()))
        super(MetaBlock, self).clear_fit_cache()

    def _get_fold_dir(self, current_cv: int):
        return f'cv={current_cv:02d}'

    def frozen(self, experiment: ExperimentBackend) -> 'MetaBlock':
        """
        save fitted models to the experiment

        Args:
            experiment:
                保存する対象となる environment
        Returns:
            myself
        """
        if not self._check_has_fitted_models():
            raise NotFittedError()
        dir_names = [self._get_fold_dir(i) for i in range(len(self._fitted_models))]
        for name, model in zip(dir_names, self._fitted_models):
            with experiment.as_environment(name, style='nested') as fold_env:
                fold_env.save_as_python_object('model', model)
        experiment.mark('cv_dirs', dir_names)
        return self

    def unzip(self, experiment: ExperimentBackend) -> 'MetaBlock':
        """load fitting models from experiment.

        Raises:
            NotFittedError
                there is no `cv_dirs` in marked object.
                fit が呼ばれた段階で, MetaBlock class は cv_dirs に cv ごとのディレクトリを保存しています.
                これが参照できない場合 fit が呼ばれていないと判断して `NotFittedError` を送出します

        Args:
            experiment:
                読み込み対象の experiment

        Returns:
            myself
        """
        if not self._check_has_models_in_exp(experiment):
            raise NotFittedError('`cv_dirs` is not found in marked object. Must be call fit before `unzip`.')
        mark = experiment.get_marked()
        output_dirs = mark.get('cv_dirs', None)  # type: List[str]
        if output_dirs is None:
            raise NotFittedError('`cv_dirs` is not found in marked object. Must be call fit before `unzip`.')

        models = []
        for out_dir in output_dirs:
            with experiment.as_environment(out_dir, style='nested') as fold_env:
                model = fold_env.load_object('model')
            models.append(model)

        self._fitted_models = models
        return self

    def get_fold_splitting(self, X, y) -> List:
        # If cv is iterable obj, convert to list and return
        cv = check_cv(self.cv, y=y,
                      classifier=not self.is_regression_model)
        splits = cv.split(X, y, groups=self.groups)
        return list(splits)

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        models = self._fitted_models
        fold_predicts = [_run_predict(model, source_df.values, is_regression=self.is_regression_model)
                         for model in models]
        preds = np.asarray(fold_predicts).mean(axis=0)
        df = pd.DataFrame(preds)
        return df

    def generate_default_model_parameter(self, X, y, experiment: Optional[ExperimentBackend] = None) -> dict:
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
            indexes_set: fit and validation index list for each fold.

        Returns:
            parameter pass to model class
        """
        params = copy.deepcopy(default_params)
        return params

    def get_fit_params_on_each_fold(self, model_params: dict,
                                    training_set: Tuple[np.ndarray, np.ndarray],
                                    validation_set: Tuple[np.ndarray, np.ndarray],
                                    indexes_set: Tuple[np.ndarray, np.ndarray],
                                    experiment: ExperimentBackend) -> dict:
        """
        generate fit params (i.e. kwrgs in `classifier.fit(X, y_true, **kwrgs)`) on each fold.
        By default, set sample_weight when sample_weight is passed to the constructor.

        Args:
            model_params: model parameters. It have been pass to model constructor.
            training_set: training (X_train, y_train) dataset. tuple of numpy array.
            validation_set: validation (X_valid, y_valid) dataset. tuple of numpy array.
            indexes_set: fit and validation index list for each fold.

        Returns:
            fit parameter dict
        """
        params = {}
        if self.sample_weight is not None:
            params['sample_weight'] = self.sample_weight[indexes_set[0]]
        return params

    def fit(self, source_df, y, experiment: ExperimentBackend) -> pd.DataFrame:
        X, y = source_df.values, y
        default_params = self.generate_default_model_parameter(X, y, experiment)
        with experiment.mark_time(prefix='fit'):
            models, oof = self.run_oof_train(X, y, default_params, experiment=experiment)

        self._fitted_models = models
        experiment.mark('n_cv', len(models))
        return pd.DataFrame(oof)

    def get_calculate_metrics(self):
        if self.is_regression_model:
            return regression_metrics

        if self._output_dim == 1:
            return binary_metrics
        return multiclass_metrics

    def run_oof_train(self, X, y, default_params, n_max: Union[int, None] = None,
                      experiment: Optional[ExperimentBackend] = None) -> ([List[PrePostProcessModel], np.ndarray]):
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
            n_max:
                Number of fold to fit. If set None, learn for all folds.
                If set number, stop fit model reach to the value.
                    * if n_fold = None, run all folds
                    * if n_fold = 1, stop one fold.
                    * if n_fold > num_cv, run all folds
                    * if n_fold <= 0, no fold run, return empty list and zero vector out-of-fold

        Returns:
            list of fitted models and out-of-fold numpy array.

            out-of-fold: shape = (n_train, output_dim)

        """
        if self.is_regression_model:
            self._output_dim = 1
        else:
            le = LabelEncoder()
            le.fit(y)
            n_classes = len(le.classes_)
            self._output_dim = 1 if n_classes == 2 else n_classes

        oof = np.zeros(shape=(len(y), self._output_dim), dtype=np.float32)

        splits = self.get_fold_splitting(X, y)
        models = []
        if experiment is None:
            experiment = ExperimentBackend()
        self.n_splits_ = len(splits)

        for i, (idx_train, idx_valid) in enumerate(splits):

            with experiment.as_environment(self._get_fold_dir(i), style='nested') as exp_i:
                if n_max is not None and i >= max(0, n_max):
                    exp_i.logger.info(f'Stop K-Fold at {i}')
                    break

                exp_i.logger.info('start k-fold: {}/{}'.format(i, self.n_splits_))

                X_i, y_i = X[idx_train], y[idx_train]
                X_valid, y_valid = X[idx_valid], y[idx_valid]

                clf = self._fit_model(X_i, y_i,
                                      default_params=default_params,
                                      validation_set=(X_valid, y_valid),
                                      indexes_set=(idx_train, idx_valid),
                                      experiment=exp_i)

                pred_i = _run_predict(clf, X_valid, is_regression=self.is_regression_model)
                oof[idx_valid] = pred_i
                models.append(clf)

                calculator = self.get_calculate_metrics()
                metric = calculator(y_valid, pred_i)
                exp_i.mark('metrics', metric)
                for l in to_pretty_lines(metric):
                    exp_i.logger.info(l)

                exp_i.mark('model_params', clf.get_params(deep=True))
                exp_i.mark('n_fold', i)
                exp_i.mark('split_info', {
                    'train_shape': idx_train.sum(),
                    'valid_shape': idx_valid.sum()
                })
        return models, oof

    def create_model(self, model_params) -> PrePostProcessModel:
        """
        create new `PrePostProcessModel` instance that has our `model_class`.

        Args:
            model_params:
                parameters pass to `PrePostProcessModel` constructor (i.e. __init__ method)

        Returns:
            new PrePoseProcessModel
        """
        target_logscale = model_params.pop('target_logscale', False)
        target_scaling = model_params.pop('target_scaling', None)
        input_logscale = model_params.pop('input_logscale', False)
        input_scaling = model_params.pop('input_scaling', None)

        model = PrePostProcessModel(instance=self.model_class(**model_params),
                                    target_logscale=target_logscale,
                                    target_scaling=target_scaling,
                                    input_logscale=input_logscale,
                                    input_scaling=input_scaling)
        return model

    def _fit_model(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   default_params: dict,
                   validation_set: tuple,
                   indexes_set: tuple,
                   experiment: ExperimentBackend) -> PrePostProcessModel:
        """
        fit a new model class.

        Notes:
            in model_params, add scaling parameters for target / input (ex. target_scaling = False)

        Args:
            X: training feature. numpy array. shape = (n_train, n_features)
            y: target. shape = (n_train, n_classes)
            default_params: parameters pass into model constructor
            validation_set:
            indexes_set:
            experiment:

        Returns:
            trained model
        """
        model_params = self.get_model_params_on_each_fold(default_params, indexes_set)
        model = self.create_model(model_params)

        # MEMO: validation data are not transform so validation score is invalid (in boosting model, eval_set)
        model._before_fit(X, y)
        x_valid, y_valid = validation_set
        x_valid = model.input_transformer.transform(x_valid)
        y_valid = model.target_transformer.transform(y_valid)

        fit_params = self.get_fit_params_on_each_fold(model_params,
                                                      training_set=(X, y),
                                                      validation_set=(x_valid, y_valid),
                                                      indexes_set=indexes_set,
                                                      experiment=experiment)
        if fit_params is None:
            fit_params = {}

        with experiment.mark_time('fit'):
            model.fit(X, y, **fit_params)
        return model


class TunerBlock(MetaBlock):
    """
    Model Based CV Feature with optuna tuning
    """
    SCORING_STRATEGY_CHOICES = ['fold', 'whole']  # choice of scoring strategy

    def __init__(self,
                 name,
                 parent=None,
                 evaluations=None,
                 n_trials=200,
                 scoring_strategy: str = 'fold',
                 scoring: Union[str, Callable, None] = None,
                 optuna_jobs: int = 1,
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
                scoring method. String or Scoring Object.
                When optimizing parameters, the best parameters in the sense of this score are chosen.
                By default (pass None)
                - for regression model, use `RMSE` metric and
                - for classifier model, use `negative log likelihood`.
                 (scoring obj must be satisfied check_scoring validation)
            optuna_jobs:
                optuna parallel jobs.
                [NOTE]
                    when set > 1, pre-post process model fail to jit input / target scaling class.
                    so it is recommend to set = 1.
        """
        super(TunerBlock, self).__init__(name=name, parent=parent, evaluations=evaluations)

        self.study = None  # type: Union[Study, None]
        self.n_trails = n_trials
        self.optuna_jobs = optuna_jobs

        if scoring_strategy not in self.SCORING_STRATEGY_CHOICES:
            raise ValueError('`scoring_strategy` must be in {}'.format(','.join(self.SCORING_STRATEGY_CHOICES)))
        self.scoring_strategy = scoring_strategy
        if scoring is None:
            if self.is_regression_model:
                scoring = 'neg_root_mean_squared_error'
            else:
                scoring = 'neg_log_loss'

        try:
            scoring = check_scoring(self.model_class, scoring=scoring, allow_none=False)
        except ValueError as e:
            s = f'Invalid scoring argument: {scoring}. You can select scoring method from pre-defined as follow\n'
            s += ', '.join(SCORERS.keys())
            raise ValueError(s)
        self.scoring_method = scoring  # type: _BaseScorer

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
            parameters used by fit out-of-fold in optimizing
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

    def get_objective(self, trial: Trial, X, y, experiment) -> float:
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
        models, oof = self.run_oof_train(X, y, default_params=params, experiment=experiment)

        if not self.is_regression_model:
            # log計算の時の overflow を防ぐ処理
            oof = np.clip(oof, 1e-7, 1 - 1e-7)

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
            import warnings
            warnings.warn(
                'Invalid storing strategy pass: {}. Use the `whole` strategy instead'.format(self.scoring_strategy))
            score = self.calculate_score(y, oof, sample_weight)
        return score

    def generate_default_model_parameter(self, X, y,
                                         experiment: Optional[ExperimentBackend] = None) -> dict:
        """
        The main loop which explore model parameter by optuna.

        Args:
            X:
            y:

        Returns:
            tuned parameter
        """
        experiment.logger.info('start optimize by optuna')
        experiment.mark('n_trials', self.n_trails)
        experiment.mark('scoring_strategy', self.scoring_strategy)
        experiment.mark('scoring', str(self.scoring_method))

        self.study = optuna.study.create_study(direction='maximize')

        # note: logging each trial is too noisy, so run all trial objective with silent context.
        with experiment.silent():
            objective = lambda trial: self.get_objective(trial, X, y, experiment)
            # Stop model logging while optuna optimization
            with experiment.mark_time('optuna_tuning'):
                self.study.optimize(objective, n_trials=self.n_trails, n_jobs=self.optuna_jobs)

        experiment.logger.info('best trial params: {}'.format(self.study.best_params))
        experiment.logger.info('best value: {}'.format(self.study.best_value))

        best_params = copy.deepcopy(self._initial_params)
        best_params.update(self.study.best_params)
        experiment.logger.info('best model paras: {}'.format(best_params))
        experiment.mark('optuna_best_value', self.study.best_value)
        experiment.save_object('study_log', self.study.trials_dataframe())
        experiment.save_object('best_params', best_params)
        experiment.save_object('best_trial_params', self.study.best_params)

        return best_params


AGG_FUNC = {
    'mean': np.mean,
    'max': np.max,
    'min': np.min
}


class EnsembleBlock(EstimatorMixin, BaseBlock):

    def __init__(self,
                 prefix,
                 agg='mean',
                 evaluations=None,
                 **kwargs):
        """
        Args:
            prefix:
                this blocks prefix.
                The final name will be `<prefix>_<agg>`.
            agg:
                aggregation method name.
                it must be able to pass as a first argument to pandas.DataFrame.agg function.
            evaluations:
                Abstract Evaluation Functions (or None).
                If set None, use default evaluations (see `_get_default_model_evaluations` definition)
            **kwargs:
                pass to BaseBlock
        """
        super(EnsembleBlock, self).__init__(name='{}_{}'.format(prefix, agg),
                                            evaluations=_get_default_model_evaluations(evaluations),
                                            **kwargs)
        self.agg = agg

        if not isinstance(agg, Callable):
            np_func = AGG_FUNC.get(self.agg, None)
            self.agg_func = lambda x: np_func(x, axis=1)
        else:
            self.agg_func = agg
        if self.agg_func is None:
            raise ValueError('agg must be in the following keys. {}'.format(','.join(AGG_FUNC.keys())))

        if len(self.parent_blocks) == 0:
            raise ValueError('Ensemble model must have parent blocks. ')

    def is_regression_model(self):
        return self.primary_block.is_regression_model

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        n_samples = len(source_df)
        n_models = len(self.parent_blocks)
        x = source_df.values.reshape(n_samples, n_models, -1)
        x = self.agg_func(x)
        return pd.DataFrame(x)
