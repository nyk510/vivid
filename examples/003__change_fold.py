import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from vivid import create_runner
from vivid.estimators.boosting import XGBRegressorBlock
from vivid.metrics import regression_metrics
from vivid.model_selection import StratifiedGroupKFold

if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7)
    train_df = pd.DataFrame(X_train)
    cv = StratifiedGroupKFold(n_splits=5, q=20, shuffle=True, random_state=71)
    xgb = XGBRegressorBlock('xgb_stratified', cv=cv)
    xgb_simple = XGBRegressorBlock('xgb_simple', cv=5)

    runner = create_runner([xgb, xgb_simple])
    runner.fit(train_df, y_train)

    test_df = pd.DataFrame(X_test)
    results = runner.predict(test_df)

    eval_scores = []
    for result in results:
        score = regression_metrics(y_test, result.oof_df.values[:, 0])
        eval_scores.append(pd.Series(score, name=result.block.name))

    eval_df = pd.DataFrame(eval_scores)

    from tabulate import tabulate

    print(tabulate(eval_df, headers='keys'))
