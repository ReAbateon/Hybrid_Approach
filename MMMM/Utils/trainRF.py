from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split


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
        f"{csv_path.stem}_RF_T{n_estimators}_D{max_depth}_RS{random_state}.joblib"
    )
    joblib_path = csv_path.parent / joblib_name

    df = pd.read_csv(csv_path)
    task = detect_task_type(df)

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
            test_size=0.3,  # Split fisso come originale
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
        model_cv = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        param_grid = {
            "min_samples_split": [i for i in range(2, 11)],
            "min_samples_leaf": [i for i in range(1, 11)],
        }
        grid_search = GridSearchCV(
            model_cv, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
            min_samples_split=grid_search.best_params_["min_samples_split"],
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        model_cv = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        param_grid = {
            "min_samples_split": [i for i in range(2, 11)],
            "min_samples_leaf": [i for i in range(1, 11)],
        }
        grid_search = GridSearchCV(
            model_cv,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
            min_samples_split=grid_search.best_params_["min_samples_split"],
            n_jobs=-1,
            random_state=random_state,
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
