from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

# IMPORTANT: backend import for deployment
from backend.preprocessing import build_pipeline


def run_automl(X, y):

    # Prevent training on extremely small datasets
    if len(X) < 10:
        raise ValueError(
            "Dataset is too small. Please upload at least 10 rows."
        )

    models = {

        "LogisticRegression": (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),
            {
                "C": [0.1, 1, 10]
            }
        ),

        "RandomForest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42
            ),
            {
                "n_estimators": [50, 100],
                "max_depth": [5, 10]
            }
        ),

        "SVM": (
            SVC(
                probability=True,
                class_weight="balanced"
            ),
            {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        ),

        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7]
            }
        ),

        "XGBoost": (
            xgb.XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42
            ),
            {
                "n_estimators": [50, 100],
                "max_depth": [3, 6]
            }
        ),
    }

    best_model = None
    best_name = None
    best_score = 0

    scores = {}

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Determine the maximum valid CV folds
    class_counts = Counter(y_train)
    min_class = min(class_counts.values())

    cv = min(5, min_class)

    if cv < 2:
        cv = 2

    print(f"Using {cv}-Fold Cross Validation")

    best_preds = None
    best_proba = None

    for name, (model, params) in models.items():

        pipeline = build_pipeline(model, X_train)

        param_grid = {
            "model__" + key: value
            for key, value in params.items()
        }

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            error_score="raise",
        )

        grid.fit(X_train, y_train)

        tuned_model = grid.best_estimator_

        preds = tuned_model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        scores[name] = round(acc, 4)

        if acc > best_score:

            best_score = acc
            best_model = tuned_model
            best_name = name
            best_preds = preds

            if hasattr(tuned_model, "predict_proba"):
                best_proba = tuned_model.predict_proba(X_test)[:, 1]

            elif hasattr(tuned_model, "decision_function"):

                decision = tuned_model.decision_function(X_test)

                decision = (
                    decision - decision.min()
                ) / (
                    decision.max() - decision.min() + 1e-8
                )

                best_proba = decision

            else:
                best_proba = preds

    metrics = {
        "accuracy": round(
            accuracy_score(y_test, best_preds), 4
        ),
        "precision": round(
            precision_score(
                y_test,
                best_preds,
                zero_division=0,
            ),
            4,
        ),
        "recall": round(
            recall_score(
                y_test,
                best_preds,
                zero_division=0,
            ),
            4,
        ),
        "f1": round(
            f1_score(
                y_test,
                best_preds,
                zero_division=0,
            ),
            4,
        ),
    }

    conf_matrix = confusion_matrix(
        y_test,
        best_preds,
    ).tolist()

    fpr, tpr, _ = roc_curve(
        y_test,
        best_proba,
    )

    roc_auc = auc(
        fpr,
        tpr,
    )

    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": round(roc_auc, 4),
    }

    return (
        best_model,
        best_name,
        best_score,
        scores,
        metrics,
        conf_matrix,
        roc_data,
    )