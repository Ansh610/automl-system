from fairlearn.metrics import demographic_parity_difference


def detect_bias(model, X, y, sensitive_feature_index):

    preds = model.predict(X)

    # FIX: use iloc for pandas
    sensitive_feature = X.iloc[:, sensitive_feature_index]

    bias = demographic_parity_difference(
        y_true=y,
        y_pred=preds,
        sensitive_features=sensitive_feature
    )

    return float(abs(bias))