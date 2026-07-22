import numpy as np
from sklearn.inspection import permutation_importance


def get_feature_importance(model, X, y):
    try:
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )

        importance = np.maximum(result.importances_mean, 0)
        return importance.tolist()

    except Exception:
        try:
            estimator = model.named_steps["model"]

            if hasattr(estimator, "feature_importances_"):
                return estimator.feature_importances_.tolist()

            if hasattr(estimator, "coef_"):
                return np.abs(estimator.coef_[0]).tolist()

        except Exception:
            pass

    return [0] * len(X.columns)