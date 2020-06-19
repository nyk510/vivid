# Models

## Basic Usage

The main model classes is defined under `vivid.estimators` module. All classes are subclass of `BaseOutOfFoldFeaure` and `BaseBlock`.

## Predefined Out of Fold Feature Models

* Linear Models: `vivid.estimators.linear`
  * Logistic Regression
  * Ridge Regression
* K Nearest Neighborhood: `vivid.estimators.neighbor`
* RandomForest: `vivid.estimators.ensumble`
* Support Vecotr Machine: `vivid.estimators.svm`
* Gradient Boosted Decision Trees: `vivid.estimators.boosting`
  * LightGBM
  * XGBoost

Main constructor arguments is as follow.

* `name` (required): model name. Recommended to unique in one project.
* `root_dir`: default = `None`. If set, the model save train and test csv and training logs (like feature importance, out-of-fold metrics, training log text and so on).
* `parent`: parent feature instance. If set, in `fit` method, the model pass the input dataframe and target to the parent, use the receive dataframe from parent on the train and test phase (instead of original input_df)

### Training

`BaseOutOfFoldFeaure` use K-Fold cross validation on train model and return out-of-fold feature (sometimes calls it meta feature).

For example, to use XGBoost classifier model feature, import `XGBClassifierBlock` from `vivid.estimators.boosting`.

```python
from vivid.estimators.boosting import XGBClassifierBlock
from sklearn.dataset import load_boston
import pandas as pd

X, y = load_boston(return_X_y=True)
train_df = pd.DataFrame(X)

model = XGBClassifierBlock(name='xgb')
oof_df = model.fit(train_df, y)
```

### Save Training Logs

If you set `root_dir`, traning logs and out-of-fold metrics save on the local storage.

```python
# only change `root_dir`
model = XGBClassifierBlock(name='xgb', root_dir='/path/to/dir')
model.fit(train_df, y)
```

The output is as follow

```
/path/to/dir/xgb
├── 0_best_fitted.joblib
├── 0_input.joblib
├── 0_target.joblib
├── 1_best_fitted.joblib
├── 1_input.joblib
├── 1_target.joblib
├── 2_best_fitted.joblib
├── 2_input.joblib
├── 2_target.joblib
├── 3_best_fitted.joblib
├── 3_input.joblib
├── 3_target.joblib
├── 4_best_fitted.joblib
├── 4_input.joblib
├── 4_target.joblib
├── boxen_feature_importance.png
├── feature_importance.csv
├── fitted_models.joblib
├── log.txt
├── metrics.csv
├── test.csv  # only generate when you call predict method.
├── train.csv
├── upper_accuracy.csv
└── upper_accuray.png
```

### Predict

Call `predict` method, return predict pandas dataframe.

```python
pred_df = model.predict(test_df)
len(pred_df) == len(test_df)  # True
```

:::warning
By default, BaseOutOfFold class cache the predict values. So If you input other test data, return same predict dataframe.

```python
pred2_df = model.predict(other_test_df)
pred_df.equals(pred2_df) # True
```

When predict other test dataset, indicate `recreate=True` explicity.

```python
pred2_df = model.predict(other_test_df, ignore_storage=True)
pred_df.equals(pred2_df) # False
```
:::

## Parameter Tuning

vivid supports optimizing model parameters by [optuna](https://optuna.org/). For models that support optimization, simply run fit to search optuna for a predetermined search range, find the best parameters and training folds using it.
For example, optimize the logistic regression model parameters, import `TunedLogisticBlock` class and run fit.

```python
model = LogisticOutOfFoldFearure(n_trials=100) # can change try times
model.fit(train_df, y)
```

you can change scoring metric on optuna study. (the defefault score is `roc_auc` on binary classification and `neg_root_mean_squared_error` on regression model.) 

```python
model = LogisticOutOfFoldFearure(n_trials=100, scoring='neg_log_loss')
```

scoring must be scorer instance or keys of `SCORES` which is defined on `_scorer.py` [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_scorer.py#L688](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_scorer.py#L688)

## Stacking

If you make stacking model, set the `MergeFeature`, which has input models as first arguments `input_features`, to the parents.

```python
from vivid.core import MergeFeature
from vivid.estimators.boosting import XGBClassifierBlock
from vivid.estimators.linear import TunedLogisticBlock

single_models = [
  XGBClassifierBlock(name='xgb'),
  TunedLogisticBlock(name='logistic')
]

# create merge all single model output feature
merged = MergeFeature(single_models[:], name='merged')

stacking_models = [
  TunedLogisticBlock(name='stackd_logistic', parent=merged)
]
```

## Seed Averaging

For Gradient Boosted Decision Tree, it is well known that ensembled model change only seed make highter performance.
vivid can make seed averaging model by using `create_boosting_seed_blocks`.

```python
from vivid.estimators.boosting.block import create_boosting_seed_blocks

models = create_boosting_seed_blocks()
```
