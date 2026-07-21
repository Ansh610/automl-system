from collections import Counter
import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from backend.preprocessing import build_pipeline


def run_automl(X, y):

    if len(X) < 10:
        raise ValueError(
            "Dataset must contain at least 10 rows."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    class_counts = Counter(y_train)

    min_class = min(class_counts.values())

    cv = max(2, min(5, min_class))

    print(f"Using {cv}-Fold Cross Validation")

    max_neighbors = max(1, len(X_train) - 1)

    knn_values = [
        k
        for k in [1, 3, 5, 7, 9]
        if k <= max_neighbors
    ]

    models = {

        "LogisticRegression": (

            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),

            {
                "C": [0.1, 1, 10]
            }

        ),

        "RandomForest": (

            RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
            ),

            {
                "n_estimators": [50, 100],
                "max_depth": [5, 10],
            }

        ),

        "SVM": (

            SVC(
                probability=True,
                class_weight="balanced",
            ),

            {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
            }

        ),

        "KNN": (

            KNeighborsClassifier(),

            {
                "n_neighbors": knn_values
            }

        ),

        "XGBoost": (

            xgb.XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                use_label_encoder=False,
            ),

            {
                "n_estimators": [50, 100],
                "max_depth": [3, 6],
            }

        ),
    }

    best_model = None
    best_name = None
    best_score = -1

    best_preds = None
    best_proba = None

    scores = {}

    for name, (model, params) in models.items():

        pipeline = build_pipeline(model, X_train)

        param_grid = {
            "model__" + key: value
            for key, value in params.items()
        }

        try:

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                error_score=np.nan,
            )

            grid.fit(X_train, y_train)

            tuned_model = grid.best_estimator_

            preds = tuned_model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            scores[name] = round(acc, 4)

            if hasattr(tuned_model, "predict_proba"):

                proba = tuned_model.predict_proba(X_test)[:, 1]

            elif hasattr(tuned_model, "decision_function"):

                decision = tuned_model.decision_function(X_test)

                decision = (
                    decision - decision.min()
                ) / (
                    decision.max() - decision.min() + 1e-8
                )

                proba = decision

            else:

                proba = preds

            if acc > best_score:

                best_score = acc
                best_model = tuned_model
                best_name = name
                best_preds = preds
                best_proba = proba

        except Exception as e:

            print(f"{name} failed: {e}")

            scores[name] = 0

    if best_model is None:

        raise ValueError(
            "None of the models could be trained on this dataset."
        )

    metrics = {
        "accuracy": round(
            accuracy_score(y_test, best_preds),
            4,
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

    try:

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
            "auc": round(float(roc_auc), 4),
        }

    except Exception:

        roc_data = {
            "fpr": [],
            "tpr": [],
            "auc": 0,
        }

    return (
        best_model,
        best_name,
        round(best_score, 4),
        scores,
        metrics,
        conf_matrix,
        roc_data,
    )