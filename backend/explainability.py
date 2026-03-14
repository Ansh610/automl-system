import joblib
import numpy as np
from sklearn.inspection import permutation_importance


def get_feature_importance(X, y):

    model = joblib.load("model.pkl")

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=42
    )

    importance = result.importances_mean

    return importance.tolist()