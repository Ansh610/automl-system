import time
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from sklearn.svm import (
    SVC,
    SVR,
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)

from sklearn.base import is_classifier

from preprocessing import build_pipeline
# ======================================================
# Detect Task
# ======================================================

def detect_task(y):

    if pd.api.types.is_numeric_dtype(y):

        unique = y.nunique()

        if unique <= 15:
            return "classification"

        return "regression"

    return "classification"


# ======================================================
# Classification Models
# ======================================================

def get_classification_models(max_neighbors):

    return {

        "Logistic Regression": (

            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),

            {
                "C": [0.1, 1, 10]
            }

        ),

        "Random Forest": (

            RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
            ),

            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
            }

        ),

        "Decision Tree": (

            DecisionTreeClassifier(
                random_state=42,
            ),

            {
                "max_depth": [5, 10, None]
            }

        ),

        "Gradient Boosting": (

            GradientBoostingClassifier(
                random_state=42,
            ),

            {
                "n_estimators": [100],
                "learning_rate": [0.05, 0.1],
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
                "n_neighbors": [
                    k
                    for k in [3,5,7,9]
                    if k <= max_neighbors
                ]
            }

        ),

        "XGBoost": (

            xgb.XGBClassifier(
                eval_metric="logloss",
                random_state=42,
            ),

            {
                "n_estimators":[100],
                "max_depth":[3,6],
            }

        ),

    }


# ======================================================
# Regression Models
# ======================================================

def get_regression_models(max_neighbors):

    return {

        "Linear Regression": (

            LinearRegression(),

            {}

        ),

        "Random Forest": (

            RandomForestRegressor(
                random_state=42,
            ),

            {
                "n_estimators":[100,200],
                "max_depth":[5,10,None],
            }

        ),

        "Decision Tree": (

            DecisionTreeRegressor(
                random_state=42,
            ),

            {
                "max_depth":[5,10,None]
            }

        ),

        "Gradient Boosting": (

            GradientBoostingRegressor(
                random_state=42,
            ),

            {
                "n_estimators":[100],
                "learning_rate":[0.05,0.1],
            }

        ),

        "SVR": (

            SVR(),

            {
                "C":[0.1,1,10]
            }

        ),

        "KNN": (

            KNeighborsRegressor(),

            {
                "n_neighbors":[
                    k
                    for k in [3,5,7,9]
                    if k <= max_neighbors
                ]
            }

        ),

        "XGBoost": (

            xgb.XGBRegressor(
                random_state=42,
            ),

            {
                "n_estimators":[100],
                "max_depth":[3,6],
            }

        ),

    }
# ======================================================
# Main AutoML Function
# ======================================================

def run_automl(X, y):

    if len(X) < 20:
        raise ValueError(
            "Dataset must contain at least 20 rows."
        )

    task = detect_task(y)

    if task == "classification":

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

        models = get_classification_models(
            max_neighbors=max(1, len(X_train)-1)
        )

    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        cv = 5

        models = get_regression_models(
            max_neighbors=max(1, len(X_train)-1)
        )

    leaderboard = []

    best_model = None
    best_name = None
    best_score = -999999

    best_predictions = None
    best_probability = None

    training_start = time.time()

    for name, (model, params) in models.items():

        try:

            pipeline = build_pipeline(model, X_train)

            param_grid = {
                f"model__{k}": v
                for k, v in params.items()
            }

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                n_jobs=-1,
                scoring=(
                    "accuracy"
                    if task=="classification"
                    else "r2"
                ),
                error_score="raise",
            )

            search.fit(X_train, y_train)

            trained_model = search.best_estimator_

            predictions = trained_model.predict(X_test)

            if task == "classification":

                score = accuracy_score(
                    y_test,
                    predictions,
                )

                if hasattr(trained_model, "predict_proba"):

                    probability = trained_model.predict_proba(
                        X_test
                    )[:,1]

                else:

                    probability = None

            else:

                score = r2_score(
                    y_test,
                    predictions,
                )

                probability = None

            leaderboard.append({

                "model": name,
                "score": round(float(score),4)

            })

            if score > best_score:

                best_score = score
                best_model = trained_model
                best_name = name
                best_predictions = predictions
                best_probability = probability

        except Exception as e:

            print(f"{name} failed -> {e}")

    leaderboard = sorted(
        leaderboard,
        key=lambda x: x["score"],
        reverse=True,
    )

    total_time = round(
        time.time()-training_start,
        2,
    )

    if task == "classification":

        metrics = {

            "accuracy": round(
                accuracy_score(
                    y_test,
                    best_predictions,
                ),
                4,
            ),

            "precision": round(
                precision_score(
                    y_test,
                    best_predictions,
                    zero_division=0,
                ),
                4,
            ),

            "recall": round(
                recall_score(
                    y_test,
                    best_predictions,
                    zero_division=0,
                ),
                4,
            ),

            "f1_score": round(
                f1_score(
                    y_test,
                    best_predictions,
                    zero_division=0,
                ),
                4,
            ),

        }

        confusion = confusion_matrix(
            y_test,
            best_predictions,
        ).tolist()

        try:

            if best_probability is not None:

                fpr, tpr, _ = roc_curve(
                    y_test,
                    best_probability,
                )

                roc_data = {

                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": round(
                        auc(fpr,tpr),
                        4,
                    ),

                }

            else:

                roc_data = {
                    "fpr": [],
                    "tpr": [],
                    "auc": 0,
                }

        except:

            roc_data = {
                "fpr": [],
                "tpr": [],
                "auc": 0,
            }

    else:

        metrics = {

            "r2_score": round(
                r2_score(
                    y_test,
                    best_predictions,
                ),
                4,
            ),

            "mae": round(
                mean_absolute_error(
                    y_test,
                    best_predictions,
                ),
                4,
            ),

            "rmse": round(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        best_predictions,
                    )
                ),
                4,
            ),

        }

        confusion = []

        roc_data = {
            "fpr": [],
            "tpr": [],
            "auc": 0,
        }

    return (

        best_model,

        best_name,

        round(float(best_score),4),

        leaderboard,

        metrics,

        confusion,

        roc_data,

        task,

        total_time,

    )