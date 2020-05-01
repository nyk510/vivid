---
home: true
heroImage: logo_transparent.png
actionText: Get Start
actionLink: /usage/
features:
  - title: 手軽に複雑なモデルが作成できます
    details: 基本的なモデルはもちろん Optuna によるパラメータチューニング・Seed Averagin・Stackingなどの複雑な処理を簡単に記述することができます。実装者はモデルの構造のみに集中してモデリングすることができます。
  - title: 自動ログ出力
    details: ディレクトリを指定するだけで自動的に学習ログを出力します。 feature importance や Accuracy などの各種メトリックを自分で記述する必要はありません。
  - title: 特徴作成を統一した記法で記述できます
    details: 学習・推論での記述方法が統一されているため、テストデータに対する特徴量がうまく作成できないといったエラーを防ぐことができます。
footer: © 2020 vivid.
---

## Install

```bash
pip install git+https://gitlab.com/nyker510/vivid
```

## Simple Usage

```python
from sklearn.datasets import load_boston
import pandas as pd
from vivid.out_of_fold.boosting import LGBMRegressorOutOfFold

X, y = load_boston(return_X_y=True)
df = pd.DataFrame(X)

model = LGBMRegressorOutOfFold(name='lgbm', cv=6, root_dir='./')
model.fit(df, y)

[2020-05-01 08:32:47,586 vivid.feature.lgbm] lgbm
[2020-05-01 08:32:47,593 vivid.feature.lgbm] CV: KFold(n_splits=6, random_state=None, shuffle=False)
[2020-05-01 08:32:47,594 vivid.feature.lgbm] start k-fold: 1/6
[2020-05-01 08:32:47,695 vivid.feature.lgbm] [100]	valid_0's rmse: 2.60877
[2020-05-01 08:32:47,763 vivid.feature.lgbm] save to: ./lgbm/0_best_fitted.joblib
[2020-05-01 08:32:47,778 vivid.feature.lgbm] Fold: 1/6 0.2[s]
[2020-05-01 08:32:47,805 vivid.feature.lgbm] start k-fold: 2/6
[2020-05-01 08:32:47,969 vivid.feature.lgbm] [100]	valid_0's rmse: 2.85395
[2020-05-01 08:32:48,042 vivid.feature.lgbm] save to: ./lgbm/1_best_fitted.joblib
[2020-05-01 08:32:48,051 vivid.feature.lgbm] Fold: 2/6 0.2[s]
[2020-05-01 08:32:48,080 vivid.feature.lgbm] start k-fold: 3/6
[2020-05-01 08:32:48,257 vivid.feature.lgbm] [100]	valid_0's rmse: 4.08426
[2020-05-01 08:32:48,330 vivid.feature.lgbm] save to: ./lgbm/2_best_fitted.joblib
[2020-05-01 08:32:48,341 vivid.feature.lgbm] Fold: 3/6 0.3[s]
[2020-05-01 08:32:48,353 vivid.feature.lgbm] start k-fold: 4/6
[2020-05-01 08:32:48,442 vivid.feature.lgbm] [100]	valid_0's rmse: 3.63623
[2020-05-01 08:32:48,516 vivid.feature.lgbm] [200]	valid_0's rmse: 3.62013
[2020-05-01 08:32:48,618 vivid.feature.lgbm] [300]	valid_0's rmse: 3.61346
[2020-05-01 08:32:48,640 vivid.feature.lgbm] save to: ./lgbm/3_best_fitted.joblib
[2020-05-01 08:32:48,664 vivid.feature.lgbm] Fold: 4/6 0.3[s]
[2020-05-01 08:32:48,677 vivid.feature.lgbm] start k-fold: 5/6
[2020-05-01 08:32:48,826 vivid.feature.lgbm] [100]	valid_0's rmse: 8.26104
[2020-05-01 08:32:48,855 vivid.feature.lgbm] save to: ./lgbm/4_best_fitted.joblib
[2020-05-01 08:32:48,870 vivid.feature.lgbm] Fold: 5/6 0.2[s]
[2020-05-01 08:32:48,881 vivid.feature.lgbm] start k-fold: 6/6
[2020-05-01 08:32:49,002 vivid.feature.lgbm] [100]	valid_0's rmse: 3.91378
[2020-05-01 08:32:49,062 vivid.feature.lgbm] save to: ./lgbm/5_best_fitted.joblib
[2020-05-01 08:32:49,071 vivid.feature.lgbm] Fold: 6/6 0.2[s]
[2020-05-01 08:32:49,074 vivid.feature.lgbm] save to ./lgbm
[2020-05-01 08:32:49,397 vivid.feature.lgbm]                               score
rmse                       4.587035
mean_squared_log_error     0.041389
median_absolute_error      2.060207
mean_squared_error        21.040892
r2_score                   0.750758
explained_variance_score   0.750852
mean_absolute_error        2.983041
/opt/conda/lib/python3.7/site-packages/sklearn/base.py:197: FutureWarning: From version 0.24, get_params will raise an AttributeError if a parameter cannot be retrieved as an instance attribute. Previously it would return None.
  FutureWarning)
[2020-05-01 08:32:49,403 vivid.feature.lgbm] for create feature: 1.810[s]
[2020-05-01 08:32:49,403 vivid.feature.lgbm] training data save to: ./lgbm/train.csv
[2020-05-01 08:32:49,406 vivid.feature.lgbm] Shape: (506, 1)
```
