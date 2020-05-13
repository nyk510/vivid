# Vivid

Support Tools for Machine Learning Vividly 🚀

## Usage

```python
from sklearn.datasets import load_boston
import pandas as pd
from vivid.out_of_fold.boosting import LGBMRegressorOutOfFold
X, y = load_boston(return_X_y=True)
df = pd.DataFrame(X)
model = LGBMRegressorOutOfFold(name='lgbm', cv=6, root_dir='./')
```

easy to make more and more complex stacking model

```python
copy_feat = CopyFeature(name='copy', root_dir='./boston_stacking')
process_feat = BostonProcessFeature(name='boston_base', root_dir='./boston_stacking')
concat_faet = [copy_feat, process_feat]

singles = [
    XGBoostRegressorOutOfFold(name='xgb_simple', parent=concat_faet),
    RFRegressorFeatureOutOfFold(name='rf', parent=concat_faet),
    KNeighborRegressorOutOfFold(name='kneighbor', parent=concat_faet),
    OptunaXGBRegressionOutOfFold(name='xgb_optuna', n_trials=20, parent=concat_faet),
    # seed averaging block
    create_boosting_seed_blocks(feature_class=XGBoostRegressorOutOfFold, prefix='xgb_', parent=concat_faet),
    create_boosting_seed_blocks(feature_class=LGBMRegressorOutOfFold, prefix='lgbm_', parent=concat_faet),

    # only processed feature
    create_boosting_seed_blocks(feature_class=LGBMRegressorOutOfFold, prefix='only_process_lgbm_',
                                parent=process_feat)
]
ens = EnsembleFeature(name='ensumble', parent=singles)  # ensemble of stackings

# create stacking models
stackings = [
    # ridge model has single models as input
    RidgeOutOfFold(name='stacking_ridge', parent=singles, n_trials=10),
    # xgboost parameter tuned by optuna
    OptunaXGBRegressionOutOfFold(name='stacking_xgb', parent=singles, n_trials=100),
]
stacking_stacking_knn \
    = KNeighborRegressorOutOfFold(name='stacking_stacking_knn', parent=stackings)
naive_xgb = XGBoostRegressorOutOfFold(name='naive_xgb', parent=copy_feat)

ens_all = RidgeOutOfFold(name='all_ridge', parent=[*singles, *stackings, ens, stacking_stacking_knn, naive_xgb])

ens_all.fit(train_df, y)
```

## Install

```bash
pip install git+https://gitlab.com/nyker510/vivid
```

## Sample Code

In `/vivid/smaples`, Some sample script codes exist.

## Developer

### Requirements

* docker
* docker-compose

create docker-image from docker-compose file

```bash
docker-compose build
docker-compose up -d
docker exec -it vivid-test bash
```

### Test

use `pytest` for test tool (see [gitlab-ci.yml](./gitlab-ci.yml)).

```bash
pytest tests
```
