# Estimators

## 概要

vivid では `vivid.estimators` 配下のモジュールで、モデルによる予測を行なう構造 Estimators を提供します。

予測モデルはすべて `vivid.estimators.base.MetaBlock` を継承しています。 `MetaBlock` 自体は vivid で扱う学習・予測を管理するクラス `BaseBlock` を継承しているため、一貫した構造を保っています。

基本的に「学習に必要なパラメータはできるだけクラス変数として記述する」方針をとっています。
モデルにバリエーションを持たせたい場合にパラメータを一部変更したり、出力先ディレクトリを変更する部分は constructor で変更しそれ以外はクラス変数として定義します。
したがってすでにあるパラメータを変更したい場合には基本的に `MetaBlock` を継承したクラスを作成する、あるいは既成のモデルのクラスを継承して変えたい部分だけ override します。

学習はすべて交差検証 (Cross Validation) の枠組みで行われ、学習時に使わなかった予測値の組 Out-Of-Fold を作成します。これによって学習データだけでモデルの性能を評価することを可能にし、かつ Stacking など予測値自体を入力とするようなモデルを自然に作成できるようになっています。

また `MetaBlock` の拡張として Optuna によるパラメータ最適化 を実行するクラス `TunedBlock` が定義されています。
パラメータの評価にも学習時と全く同じコードが使われ、交差検証によって評価されます。これによりパラメータ最適化のためにユーザーが準備することはほとんどありません。

## 定義済みのモデルの利用

以下のクラスは vivid からデフォルトで用意されていますので、すぐに利用することが可能です。

* Linear Models: `vivid.estimators.linear`
  * Logistic Regression
  * Ridge Regression
* K Nearest Neighborhood: `vivid.estimators.neighbor`
* RandomForest: `vivid.estimators.ensumble`
* Support Vecotr Machine: `vivid.estimators.svm`
* Gradient Boosted Decision Trees: `vivid.estimators.boosting`
  * LightGBM
  * XGBoost
* NeuralNetwork: `vivid.estimators.neural_network`: Keras を用いた Multilayer Neutral Network

### 定義済みモデルの構成

処理はほとんど `MetaBlock` に記述されているため、定義済みモデルの実装自体は至ってシンプルです。例えば random forest の二値分類 `RFClassifierBlock` の定義は以下のようになっています。

```python
class RFClassifierBlock(MetaBlock):
    model_class = RandomForestClassifier
    initial_params = {
        'criterion': 'gini',
        'class_weight': 'balanced'
    }
```

モデル学習のクラスを作成するのに最低限必要な要素は以下の２つです。

* `model_class`: scikit-learn estimator クラスを指定します。scikit-learn のモデルに準拠した実装であればどんなクラスでも使用することができます。
* `initail_params`: `model_class` からインスタンスを作成する際に constructor に渡されるパラメータです。

## Runner を使ったモデルの学習

scikit-learn とことなり、vivid ではモデルの結果を保存することを念頭に設計されている為、学習時には学習用データとラベルに加えて、どの場所で実行するかなどの実験条件 `experiment` も同時に渡す必要があります。また parent が設定されているスタッキングモデルの場合、モデル自体には親の特徴量を作成する機能がないため、自分で作成する必要があります。

この設定は若干煩雑ですが `vivid.runner.Runner` を使うことで簡単に実行できます。必要な情報は

* どのブロックを学習させたいか (block)
* 実行環境: どの環境で学習させるか (`experiment` / optional)

の2つです。実行環境に関しては省略することもできます。省略された場合キャッシュディレクトリに一時的なディレクトリを作成し、そこに結果が保存されます。

::: details More Info
Runner は内部で親子間の依存関係を取得して、無駄がない順番で学習を実行する機能を備えています(ナイーブに再帰的に実行すると重複するモデルが出現した時無駄になる)ので、複雑なモデル構造の場合特に Runner を使うことをおすすめします。
:::

例えば XGBoost classifier model で学習をしたい時には `XGBClassifierBlock` を `vivid.estimators.boosting` からインポートして実行します。

```python
from vivid.estimators.boosting import XGBClassifierBlock
from vivid.runner import Runner

from sklearn.dataset import load_boston
import pandas as pd

X, y = load_boston(return_X_y=True)
train_df = pd.DataFrame(X)

model = XGBClassifierBlock(name='xgb')

# runner を使って学習を実行
runner = Runner(model)
results = runner.fit(train_df, y)
```

ディレクトリを指定する場合には明示的に `experiment` を渡します。

```python
from vivid.backends.experiment import LocalExperimentBackend

exp = LocalExperimentBackend(to='/path/to/save/dir')
runner = Runner(model, experiment)
results = runner.fit(train_df, y)
```

`fit` から返されるのは `vivid.runner.EstimatorResult` の配列です。この結果はモデルの予測値が入っています。今回の場合はブロックは `model` ひとつしかありませんので長さが1の配列です。複数の block を指定した場合や、`model` が親を持っていて学習に複数の block が関与する場合等にはそれらがすべて格納されます。

::: tip
特徴量に関しては EstimatorResult に含まれないので注意してください
:::

学習時に用いられるパラメータはクラス変数 `init_params` として定義されています。以下は XGBoost の例です。

```python
class XGBClassifierBlock(BaseBoostingBlock):
    default_eval_metric = 'logloss'
    model_class = xgb.XGBClassifier
    # initial_params: ここで設定された値が model_class を作成する際に渡される。
    initial_params = {
        'learning_rate': .1,
        'reg_lambda': 1e-2,
        'n_estimators': 1000,
        'colsample_bytree': .8,
        'subsample': .7,
    }
```

クラスをオーバーライドするのは大げさで、その場で少しだけ値を変更したい場合があるでしょう。その場合にはインスタンス作成時に変更したいパラメータを `add_init_params` として渡してください。例えば XGBoost で `reg_lambda` だけ変えたい場合には以下のようにします。

```python
model = XGBClassifierBlock(name='xgb', add_init_params={ 'reg_lambda': 10 })
```

これと同じように KFold の種類や `sample_weight` なども変更することができます。詳しくは `vivid.estimators.base.MetaBlock` を参照してください。

## Runner の出力

Runner は experiment で指定されたディレクトリ配下に、モデルごとに学習済みの重みや入力の情報・out-of-fold の評価値等を保存します。例えばサンプルコード `examples/0001__simple.py` を実行した結果のディレクトリの一つ `xgb_classifier__1c720921` を見ると、以下のような情報が保存されます。

```bash
$ tree xgb_classifier__1c720921

xgb_classifier__1c720921
├── cv=00  # CV ごとの結果がディレクトリごとに出力されます.
│   ├── log.txt
│   ├── metrics.json
│   └── model.joblib
├── cv=01
│   ├── log.txt
│   ├── metrics.json
│   └── model.joblib
├── cv=02
│   ├── log.txt
│   ├── metrics.json
│   └── model.joblib
├── cv=03
│   ├── log.txt
│   ├── metrics.json
│   └── model.joblib
├── cv=04
│   ├── log.txt
│   ├── metrics.json
│   └── model.joblib
├── distribution.png # 予測分布の可視化結果
├── importance.csv # 特徴重要度の CV ごとの値
├── importance.png # 特徴重要度を可視化したもの
├── log.txt # 学習全体のログ
├── metrics.json # 学習中に記録されたメトリック. 例えば `fit` 実行全体でかかった時間など.
├── oof_output_sample.csv # out-of-fold の値のサンプル.(全体は `train_output.joblib` に保存されています)
├── parent_output_sample.csv # このモデルへの入力特徴量のサンプル
├── pr_auc.png # PR-AUC Curve が描かれたグラフ
├── roc_auc.png # ROC-AUC Curve が描かれたグラフ
├── test_output.joblib
└── train_output.joblib
```

### CVごとの記録

fold ごとのモデルの重みと学習のログです。`metric.json` には学習時の条件や学習 `fit` にかかった時間、分割の情報などが記載されています。以下はそのサンプルです。

::: details output metric.json

```json
{
    "fit": {
        "start_at": "2020-06-18 23:24:36.638234",
        "end_at": "2020-06-18 23:24:36.671734",
        "duration_minutes": "0.00"
    },
    "model_params": {
        "input_logscale": false,
        "input_scaling": false,
        "instance": {
            "class": "<class 'xgboost.sklearn.XGBClassifier'>",
            "params": {
                "objective": "binary:logistic",
                "base_score": null,
                "booster": null,
                "colsample_bylevel": null,
                "colsample_bynode": null,
                "colsample_bytree": 0.8,
                "gamma": null,
                "gpu_id": null,
                "importance_type": "gain",
                "interaction_constraints": null,
                "learning_rate": 0.1,
                "max_delta_step": null,
                "max_depth": null,
                "min_child_weight": null,
                "missing": NaN,
                "monotone_constraints": null,
                "n_estimators": 1000,
                "n_jobs": null,
                "num_parallel_tree": null,
                "random_state": null,
                "reg_alpha": null,
                "reg_lambda": 0.01,
                "scale_pos_weight": null,
                "subsample": 0.7,
                "tree_method": null,
                "validate_parameters": false,
                "verbosity": null
            }
        },
        "target_logscale": false,
        "target_scaling": false,
        "input_transformer": {
            "log": false,
            "scaling": null
        },
        "target_transformer": {
            "log": false,
            "scaling": null
        }
    },
    "n_fold": 0,
    "split_info": {
        "train_shape": 121164,
        "valid_shape": 6601
    }
}
```

:::

### 学習全体での記録

学習にかかった時間や使った特徴量のカラム名・out-of-fold の評価結果は `metric.json` に保存されます。以下はそのサンプルです。

::: details output metric.json

```json
{
    // 学習全体でのメタデータ. 開始・終了時刻と時間
    "fit": {
        "start_at": "2020-06-18 23:24:36.637258",
        "end_at": "2020-06-18 23:24:37.620713",
        "duration_minutes": "0.02"
    },
    "n_cv": 5,
    // out-of-fold で計測したモデルの予測性能. 
    // classification / regression ごとに決められたデフォルトの指標が計算されます.
    "train_metrics": {
        "accuracy_score": 0.924901185770751,
        "f1_score": 0.9602510460251046,
        "precision_score": 0.9683544303797469,
        "recall_score": 0.9522821576763485,
        "roc_auc_score": 0.9253976486860304,
        "log_loss": 0.15691278449745374,
        "average_precision_score": 0.9961752729223832
    },
    "train_meta": {
        // 出力結果の shape
        "shape": [
            506,
            1
        ],
        // このモデルへ入力された特徴量カラム
        "parent_columns": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12
        ],
        // 出力結果のカラム名
        "output_columns": [
            "predict"
        ],
        // 出力のうちの null の数 (pandas.isnull().sum())
        "output_null": {
            "predict": 0
        },
        // 特徴量が使用していたメモリ
        "memory_usage": 2152
    },
    "cv_dirs": [
        "cv=00",
        "cv=01",
        "cv=02",
        "cv=03",
        "cv=04"
    ]
}
```

:::

いくつかの学習結果は画像として保存されます。例えば二値分類問題だとデフォルトでは AUC / PR-AUC のグラフと予測値の POS / NEG での分布が保存されます。
また特徴重要度が計算できるモデルの場合、重要度が `importance.png` として保存されます。

::: tip
結果の出力はモデルインスタンスの引数 `evaluations` に渡される関数配列によって決定します。可視化結果を自分で作成したい場合には `vivid.core.AbstractEvaluation` を継承したクラスを作成して `evaluations` に指定してください。
:::

### Predict

`runner.predict` method を呼び出すことで、学習済みのモデルを使って新しいデータに対する予測を行えます。この時の返り値も `fit` と同じく `EstimatorResult` の配列です。

```python
predict_results = runner.predict(test_df)
```

## ハイパーパラメータの最適化

`optuna` によるパラメータチューニングをサポートしています。パラメータチューニングを行えるクラスも基本的なモデル特徴量クラスと同様に `fit` を呼び出すだけでパラメータ最適化を実行できます。
例えば線形モデル logistic regression に対するクラスは `vivid.features.linear.TunedLogisticBlock` です。

```python
from vivid.features.linear import TunedLogisticBlock

tuned_model = TunedLogisticBlock(n_trials=100) # can change try times
```

### 目的関数の変更

Optuna による最適化の目的関数は、デフォルトでは回帰問題のとき負の二乗和誤差平均 `neg_root_mean_squared_error` ・分類問題の時 AUC `roc_auc_score` が用いられます。別の基準を用いてパラメータを選択したい場合には引数 `scoring` を変更することで設定を変えられます。

```python
tuned_model = TunedLogisticBlock(n_trials=100, scoring='neg_log_loss')
```

`scoring` は scikit-learn の Scoreer Instance もしくは `_score.py` に定義されている `SCORES` の `key` 文字列のいずれかである必要があります。

> [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_scorer.py#L688](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_scorer.py#L688)

### 探索空間の変更

ハイパパラメータ探索を行なう空間を決めたい場合 `generate_model_class_try_params` をオーバーライドしてください。これは `optuna.Trial` を引数にとって、この試行で利用するパラメータを返すような関数です。
先ほどの例で取り上げた Logistic モデルのデフォルトの定義は以下のようになっています。

```python
# @vivid.features.linear.py
class TunedLogisticBlock(TunerBlock):
    model_class = LogisticRegression
    initial_params = {
        'solver': 'liblinear',
        'penalty': 'l2',
        'input_scaling': 'standard',
    }

    def generate_model_class_try_params(self, trial):
        return {
            'C': trial.suggest_loguniform('C', 1e-3, 1e2),
        }
```

例えば `C` 以外の `input_scaling` についても探索したい場合には以下のようになります。

```python
class CustomTunedLogisticBlock(TunedLogisticBlock):
    def generate_model_class_try_params(self, trial):
        return {
            'C': trial.suggest_loguniform('C', 1e-3, 1e2),
            'input_scaling': trial.suggest_categorical('input_scaling',
                                                      ['standard', None]),
        }
```

## Stacking モデル

Stacking (モデルの入力に別のモデルの予測値が入るような構造を持ったアーキテクチャのこと) は、予測性能を向上させるために用いられる機械学習の技術の一つです。Stacking モデルを作成したい場合には `parent` に入力としたいモデルインスタンス・あるいはその配列で指定してください。

```python
from vivid.estimators.boosting import XGBClassifierBlock
from vivid.estimators.linear import TunedLogisticBlock

single_models = [
  XGBClassifierBlock(name='xgb'),
  TunedLogisticBlock(name='logistic')
]

stacking_models = [
  TunedLogisticBlock(name='stacked_logistic', parent=single_models)
]

runner = Runner(stacking_models)
results = runner.fit(train_df, y)
```

より複雑な構造を作ることもできます。以下はその一例です。

```python
from vivid.estimators.base import EnsembleBlock
from vivid.estimators.boosting import XGBRegressorBlock
from vivid.estimators.ensumble import RFRegressorBlock
from vivid.estimators.linear import TunedRidgeBlock
from vivid.estimators.neural_network import KerasRegressorBlock
from vivid.estimators.svm import SVRBlock

# 0. シンプルなモデル
models = [
    XGBRegressorBlock('xgb'),
    TunedRidgeBlock('ridge'),
    RFRegressorBlock('rf'),
    KerasRegressorBlock('keras'),
    SVRBlock('svr')
]

# 1. シンプルなモデルをアンサンブル
ensemble = EnsembleBlock('ens', parent=models)

# 2. シンプルなモデルのスタッキング
stacking = [
    SVRBlock('stacked_svr', parent=models),
    TunedRidgeBlock('stacked_ridge', parent=models)
]

# 3. すべてを stacking した ridge モデル
stacked_stack = TunedRidgeBlock('ridge', parent=[*stacking, *models, ensemble])

# 4. 更にその結果も含めてアンサンブル
stacked_ens = EnsembleBlock('stacked_ens', parent=[*stacking, *models, ensemble, stacked_stack])
```

## Seed Averaging

For Gradient Boosted Decision Tree, it is well known that ensembled model change only seed make highter performance.
vivid can make seed averaging model by using `create_boosting_seed_blocks`.

```python
from vivid.estimators.boosting.block import create_boosting_seed_blocks

models = create_boosting_seed_blocks()
```
