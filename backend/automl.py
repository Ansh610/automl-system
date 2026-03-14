from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

from preprocessing import build_pipeline


def run_automl(X, y):

    models = {

        "LogisticRegression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"C": [0.1, 1, 10]}
        ),

        "RandomForest": (
            RandomForestClassifier(class_weight="balanced"),
            {"n_estimators": [50, 100], "max_depth": [5, 10]}
        ),

        "SVM": (
            SVC(probability=True, class_weight="balanced"),
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
        ),

        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7]}
        ),

        "XGBoost": (
            xgb.XGBClassifier(eval_metric="logloss"),
            {"n_estimators": [50, 100], "max_depth": [3, 6]}
        ),
    }

    best_model = None
    best_name = None
    best_score = 0

    scores = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_preds = None
    best_proba = None

    for name, (model, params) in models.items():

        pipeline = build_pipeline(model, X_train)

        param_grid = {"model__" + k: v for k, v in params.items()}

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
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

            else:
                best_proba = preds

    metrics = {
        "accuracy": accuracy_score(y_test, best_preds),
        "precision": precision_score(y_test, best_preds),
        "recall": recall_score(y_test, best_preds),
        "f1": f1_score(y_test, best_preds)
    }

    conf_matrix = confusion_matrix(y_test, best_preds).tolist()

    fpr, tpr, _ = roc_curve(y_test, best_proba)

    roc_auc = auc(fpr, tpr)

    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": roc_auc
    }

    return best_model, best_name, best_score, scores, metrics, conf_matrix, roc_data