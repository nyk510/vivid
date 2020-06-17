import pytest
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, Lasso
from xgboost.sklearn import XGBClassifier

from vivid.sklearn_extend.wrapper import PrePostProcessModel
from vivid.visualize import visualize_feature_importance, NotSupportedError
from vivid.visualize import visualize_pr_curve, visualize_roc_auc_curve, visualize_distributions


@pytest.mark.parametrize('func', [
    visualize_roc_auc_curve,
    visualize_pr_curve,
    visualize_distributions
])
def test_raise_error_continuous(func):
    y_true = [1, 1, 1, 0, 0, .6]  # contains continuous value `.6`
    y_pred = [1, 2, 1, .5, .3, 9]

    with pytest.raises(ValueError):
        func(y_true, y_pred)


@pytest.mark.parametrize('function', [
    visualize_roc_auc_curve,
    visualize_pr_curve,
    visualize_distributions
])
def test_binary_predict(function):
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [.2, .3, .4, .1, .4, .7]

    function(y_true, y_pred)


@pytest.mark.parametrize('clf', [
    RidgeClassifier(),
    Lasso(),
    RandomForestClassifier(),

    XGBClassifier(),
    LGBMClassifier(),
    PrePostProcessModel(instance=Lasso())
])
def test_feature_importance(binary_Xy, clf):
    clf.fit(*binary_Xy)

    from vivid.visualize import visualize_feature_importance

    visualize_feature_importance([clf])


def test_not_support_classifier(binary_Xy):
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(*binary_Xy)

    with pytest.raises(NotSupportedError):
        visualize_feature_importance([clf])