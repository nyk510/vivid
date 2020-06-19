# Vivid

Support Tools for Machine Learning Vividly 🚀

[Documents](https://lucid-lumiere-9b9b91.netlify.app/)

## Usage

The concept of vivid is **easy to use**. Only make instance and run fit, vivid save model metrics and weights (like feature_imporance, pr/auc curve, training time, ...) .

```python
import pandas as pd
from sklearn.datasets import load_boston

from vivid.backends.experiments import LocalExperimentBackend
from vivid.estimators.boosting import XGBRegressorBlock

X, y = load_boston(return_X_y=True)
train_df = pd.DataFrame(X)

# create model and experiment
xgb = XGBRegressorBlock('xgb')
experiment = LocalExperimentBackend('./outputs/simple')

# run models
from vivid.runner import create_runner

runner = create_runner(blocks=xgb, experiment=experiment)
runner.fit(train_df, y)
runner.predict(train_df)
```

VIVID makes it easy to describe model/feature relationships. For example, you can easily describe stacking, which can be quite complicated if you create it normally.


## Install

```bash
pip install python-vivid
```

## Sample Code

In `/vivid/samples`, Some sample script codes exist.

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
