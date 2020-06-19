# Introduction

## Install

```bash
pip install python-vivid
```

## シンプルな使い方

もっとも簡単につかうには、モデル学習用のクラスを import して学習モデルを作り `fit` を呼び出します。
モデルは cross-validation によって学習され、学習データと同じ大きさの out-of-fold 予測値を得ます。

```python
from sklearn.datasets import load_boston
import pandas as pd

X, y = load_boston(return_X_y=True)
train_df = pd.DataFrame(X)

from vivid.estimators.boosting import XGBRegressorBlock

model = XGBRegressorBlock('xgb')
oof_df = model.fit(train_df, y)
```
