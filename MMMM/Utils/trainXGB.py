from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.stats import randint, uniform
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier, XGBRegressor


def detect_task_type(df):
    y = df.iloc[:, -1]
    if not np.issubdtype(y.dtype, np.number):
        return 0  # Classification

    unique_vals = np.unique(y)
    n_unique = len(unique_vals)

    if np.issubdtype(y.dtype, np.integer):
        return 0 if n_unique <= 50 else 1

    if np.issubdtype(y.dtype, np.floating):
        return 0 if n_unique <= 50 else 1

    return 1  # Fallback to Regression


def compute_columnwise_quantization_params(X):
    # Per ogni colonna trovo il massimo assoluto
    max_vals = np.max(np.abs(X), axis=0)
    # Evitiamo divisioni per zero
    max_vals[max_vals == 0] = 1.0
    # E calcolo il fattore di scala per mappare in int16
    scale = 32766 / max_vals
    return scale


def quantize_columnwise(X, scale):
    # Quantizzo ogni colonna e converto in int16
    X_q = np.round(X * scale).astype(np.int16)
    return X_q


def training(csv_path, n_estimators, max_depth, random_state, test_size):
    csv_path = Path(csv_path)
    joblib_name = (
        f"{csv_path.stem}_XGB_T{n_estimators}_D{max_depth}_RS{random_state}.joblib"
    )
    joblib_path = csv_path.parent / joblib_name

    df = pd.read_csv(csv_path)
    task = detect_task_type(df)

    # For XGBoost classification, classes must be 0..N-1
    if task == 0:
        target_col = df.columns[-1]
        classes = np.unique(df[target_col])
        if not np.array_equal(classes, np.arange(len(classes))):
            mapping = {cls: i for i, cls in enumerate(sorted(classes))}
            df[target_col] = df[target_col].map(mapping)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # --- Quantizzazione ---
    print("Quantization to int16...")
    scale_X = compute_columnwise_quantization_params(X)
    X_q = quantize_columnwise(X, scale_X)

    if task == 1:  # Regression
        scale_y = compute_columnwise_quantization_params(y)
        y_q = quantize_columnwise(y, scale_y)
    else:
        y_q = y  # Le classi rimangono int

    if task == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X_q,
            y_q,
            test_size=0.3,
            random_state=random_state,
            stratify=y_q if len(np.unique(y_q)) > 1 else None,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_q, y_q, test_size=0.3, random_state=random_state
        )

    if joblib_path.exists():
        print(f"Loading existing model from {joblib_path}")
        model = load(joblib_path)
        y_pred = model.predict(X_test)
        if task == 0:
            acc = accuracy_score(y_test, y_pred)
            mae = 0
        else:
            acc = 0
            mae = mean_absolute_error(y_test, y_pred)
        return model, task, joblib_path, X_test.head(test_size), acc, mae

    print("Training new model with Cross Validation...")
    if task == 0:
        n_classes = y.nunique()
        if n_classes > 2:
            xgb = XGBClassifier(
                objective="multi:softprob",
                n_estimators=n_estimators,
                num_class=n_classes,
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
            )
        else:
            xgb = XGBClassifier(
                objective="binary:logistic",
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
            )

        param_grid = {
            "learning_rate": [0.05, 0.1, 0.2],
            "gamma": [0, 0.1, 0.2, 0.5],
            "max_depth": [2, 3, 5],
            "subsample": [0.5, 0.75, 1.0],
            "reg_alpha": [0, 0.1, 0.2],
            "reg_lambda": [1, 2, 3],
        }
        grid_search = GridSearchCV(
            xgb, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        if n_classes > 2:
            model = XGBClassifier(
                n_estimators=n_estimators,
                num_class=n_classes,
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
                **grid_search.best_params_,
            )
        else:
            model = XGBClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
                **grid_search.best_params_,
            )
    else:
        xgb = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
        param_dist = {
            "learning_rate": uniform(0.02, 0.08),
            "max_depth": randint(2, 6),
            "min_child_weight": randint(1, 15),
            "gamma": uniform(0.0, 0.5),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": uniform(0.0, 0.3),
            "reg_lambda": uniform(0.5, 3.0),
        }
        rand_search = RandomizedSearchCV(
            xgb,
            param_dist,
            n_iter=60,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )
        rand_search.fit(X_train, y_train)

        print(f"Best parameters: {rand_search.best_params_}")
        model = XGBRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            **rand_search.best_params_,
        )

    model.fit(X_train, y_train)
    dump(model, joblib_path)

    y_pred = model.predict(X_test)
    if task == 0:
        acc = accuracy_score(y_test, y_pred)
        mae = 0
    else:
        acc = 0
        mae = mean_absolute_error(y_test, y_pred)

    return model, task, joblib_path, X_test.head(test_size), acc, mae
