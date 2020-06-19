# Customize

## Change Parameter Search Range

When you change parameter search range, override `generate_model_class_try_params`. 

```python
from vivid.estimators.boosting import TunedXGBRegressorBlock

class CustomXGBoostRegressor(TunedXGBRegressorBlock):
  def generate_model_class_try_params(self, trial):
    params = {
      'reg_lambda': trial.suggest_uniform(0, 1, 'reg_lambda')
    }
    return params
```

## Create Own Out Of Feature Model

When create your own out of feature model, extend `MetaBlock` class and set `model_class` field as sklearn-like model class (i.e. subclass of RegressorMixin / ClassifierMixin and BaseEstimator). The inital parameter pass to the model constructor can change the `init_params` field. i.e.

```python
from sklearn.tree import DecisionTreeRegressor

class TreeRegressorOutOfFold(MetaBlock):
  model_class = DecisionTreeRegressor
  init_params = {
    'min_leaf_samples': 4,
    'max_depth': 5
  }
```

If you create optuna tuning model, extend `BaseOptunaOutOfFoldFeature` class and override `generate_model_class_try_params` method. The example of the k-neighborhood implemented in this module is as follows.

```python
class OptunaKNeighborRegressorOutOfFold(BaseOptunaOutOfFoldFeature):
    model_class = KNeighborsRegressor
    init_params = {
      'target_scaling': True
    }

    def generate_model_class_try_params(self, trial: Trial):
        params = {
            'weights': trial.suggest_categorical('weights', ['distance', 'uniform']),
            'p': trial.suggest_uniform('p', 1, 4),
            'n_neighbors': int(trial.suggest_int('n_neighbors', 5, 30)),
            'algorithm': trial.suggest_categorical('algorithm', ['ball_tree', 'kd_tree'])
        }

        if 'tree' in params.get('algorithm', None):
            params['leaf_size'] = int(trial.suggest_int('leaf_size', 10, 200))

        return params
```

## Change CV Strategy

You want to change cv strategy, add cv kwrgs to the model constructor.

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_split=4)
model = LogisticOutOfFold(name='logistic_stratified', cv=cv)
```

::: tip
When use `GroupKFold`, add `groups` to the model constructor too.
:::

cv argument also supports more flexible format, list of indexes (train indexes and validation indexes on each fold).

```python
cv = [
  [[1, 2, 4], [3, 5]],
  [[2, 3, 5], [1, 4]]
]
model = LogisticOutOfFold(name='handmade_cv', cv=cv)
```
