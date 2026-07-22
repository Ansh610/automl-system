import joblib
import numpy as np
from pathlib import Path
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"


def get_feature_importance(X, y):

    if not MODEL_PATH.exists():
        return [0] * len(X.columns)

    model = joblib.load(MODEL_PATH)

    try:

        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=10,
            random_state=42,
            scoring=None,
            n_jobs=-1,
        )

        importance = result.importances_mean

        importance = np.maximum(importance, 0)

        return importance.tolist()

    except Exception:

        try:

            estimator = model.named_steps["model"]

            if hasattr(estimator, "feature_importances_"):

                return estimator.feature_importances_.tolist()

            elif hasattr(estimator, "coef_"):

                return (
                    np.abs(estimator.coef_[0]).tolist()
                )

        except Exception:
            pass

    return [0] * len(X.columns)