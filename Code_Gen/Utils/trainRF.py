#
# Copyright (c) 2026 Lorenzo Abate <lorenzo.abate@unina.it>.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

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


def compute_columnwise_quantization_params(X, num_bits=16):

    # Per ogni colonna trovo il massimo
    max_vals = np.max(np.abs(X), axis=0)
    print("Max Values:")
    print(max_vals)
    # E calcolo il fattore di scala
    scale = 32766 / max_vals
    print("Scales:")
    print(scale)
    return scale


def quantize_columnwise(X, scale):
    # Quantizzo ogni colonna
    X_q = np.round(X * scale).astype(np.int16)
    return X_q


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


def training(csv_path, number_of_trees, max_depth, random_seed, test_size):
    csv_dir = csv_path.parent
    csv_stem = csv_path.stem

    joblib_name = (
        f"{csv_stem}_RF_T{number_of_trees}_D{max_depth}_RS{random_seed}.joblib"
    )
    joblib_path = csv_dir / joblib_name

    test_name = f"{csv_stem}_test_TS{test_size}_RS{random_seed}.csv"
    test_path = csv_dir / test_name

    joblib_exists = 0

    if joblib_path.exists():
        joblib_exists = 1

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Display basic dataset information
    print("Dataset loaded successfully!")

    task = detect_task_type(df)

    # Define features (X) and target (y)
    print("Extracting features and target variable...")
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]  # Last column

    sample_size = X.shape[1]

    print("Quantization...")
    scale = compute_columnwise_quantization_params(X)

    X_q = quantize_columnwise(X, scale)

    if task == 1:
        scale_reg = compute_columnwise_quantization_params(y)

        y_q = quantize_columnwise(y, scale_reg)

        X_q_train, X_q_test, y_q_train, y_q_test = train_test_split(
            X_q, y_q, test_size=0.3, random_state=random_seed
        )
    else:
        X_q_train, X_q_test, y_q_train, y_q_test = train_test_split(
            X_q, y, test_size=0.3, random_state=random_seed, stratify=y
        )

    if not test_path.exists():
        X_q_test.head(test_size).to_csv(test_path, index=False)

    if joblib_exists == 1:
        print("A Joblib already exists!")
        best_rf = load(joblib_path)
    else:
        if task == 0:
            model_cv = RandomForestClassifier(
                n_estimators=number_of_trees,
                max_depth=max_depth,
                random_state=random_seed,
                n_jobs=-1,
            )
        else:
            model_cv = RandomForestRegressor(
                n_estimators=number_of_trees,
                max_depth=max_depth,
                random_state=random_seed,
                n_jobs=-1,
            )

        param_grid = {
            "min_samples_split": [i for i in range(2, 11)],
            "min_samples_leaf": [i for i in range(1, 11)],
        }

        # Perform grid search with 5-fold cross-validation
        print("Performing grid search with 5-fold cross-validation...")
        if task == 0:
            grid_search = GridSearchCV(
                model_cv, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
            )
        else:
            grid_search = GridSearchCV(
                model_cv,
                param_grid,
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=1,
            )
        grid_search.fit(X_q_train, y_q_train)

        # Best parameters and score
        print(f"Best parameters found: {grid_search.best_params_}")
        if task == 0:
            print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
            print("Training the final model with the best parameters...")
            best_rf = RandomForestClassifier(
                n_estimators=number_of_trees,
                max_depth=max_depth,
                min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                min_samples_split=grid_search.best_params_["min_samples_split"],
                n_jobs=-1,
                random_state=random_seed,
            )
            best_rf.fit(X_q_train, y_q_train)
        else:
            print(f"Best CV MSE: {-grid_search.best_score_:.4f}")
            print("Training the final model with the best parameters...")
            best_rf = RandomForestRegressor(
                n_estimators=number_of_trees,
                max_depth=max_depth,
                min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                min_samples_split=grid_search.best_params_["min_samples_split"],
                n_jobs=-1,
                random_state=random_seed,
            )
            best_rf.fit(X_q_train, y_q_train)

        dump(best_rf, joblib_path)

    y_pred = best_rf.predict(X_q_test)

    accuracy = 0
    mse = 0
    mae = 0
    r2 = 0

    if task == 0:
        accuracy = accuracy_score(y_q_test, y_pred)
    else:
        mse = mean_squared_error(y_q_test, y_pred)
        mae = mean_absolute_error(y_q_test, y_pred)
        r2 = r2_score(y_q_test, y_pred)

    return test_path, joblib_path, sample_size, task, accuracy, mse, mae, r2, csv_stem
