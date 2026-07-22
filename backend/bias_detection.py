import numpy as np
from fairlearn.metrics import demographic_parity_difference


def detect_bias(model, X, y, sensitive_feature_index):
    """
    Returns demographic parity difference.
    Lower value = less bias.
    """

    try:

        preds = model.predict(X)

        sensitive_feature = X.iloc[:, sensitive_feature_index]

        bias = demographic_parity_difference(
            y_true=y,
            y_pred=preds,
            sensitive_features=sensitive_feature,
        )

        return round(float(abs(bias)), 4)

    except Exception:

        # If Fairlearn cannot calculate
        return 0.0