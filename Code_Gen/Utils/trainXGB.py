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

import re
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


def detect_task_type(df, max_classes=50, class_ratio=0.05):
    y = df.iloc[:, -1]  # ultima colonna = target
    n_samples = len(y)

    # Caso 1: target non numerico
    if not np.issubdtype(y.dtype, np.number):
        return 0

    unique_vals = np.unique(y)
    n_unique = len(unique_vals)

    # Caso 2: target numerico ma intero
    if np.issubdtype(y.dtype, np.integer):
        # poche classi rispetto ai campioni
        if n_unique <= max_classes:
            return 0
        else:
            return 1

    # Caso 3: target float
    if np.issubdtype(y.dtype, np.floating):
        # se pochi valori distinti → classificazione
        if n_unique <= max_classes:
            return 0
        else:
            return 1

    # fallback
    return 1


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


def training(csv_path, n_trees, random_seed, test_size):
    mae = 0
    mse = 0
    r2 = 0
    acc = 0

    csv_dir = csv_path.parent
    csv_stem = csv_path.stem

    joblib_name = f"{csv_stem}_XGB_T{n_trees}_RS{random_seed}.joblib"
    joblib_path = csv_dir / joblib_name

    test_name = f"{csv_stem}_test_TS{test_size}_RS{random_seed}.csv"
    class_name = f"{csv_stem}_testclass_TS{test_size}_RS{random_seed}.csv"
    test_path = csv_dir / test_name
    class_path = csv_dir / class_name

    joblib_exists = 0

    if joblib_path.exists():
        joblib_exists = 1

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Display basic dataset information
    print("Dataset loaded successfully!")

    task = detect_task_type(df)

    if task == 0:
        # === NORMALIZZAZIONE CLASSI (target = ultima colonna) ===
        target_col = df.columns[-1]

        y = df[target_col].to_numpy()

        # Classi trovate nel dataset
        classes = np.unique(y)

        # Dovrebbero essere [0, 1, ..., N-1]
        expected = np.arange(len(classes))

        # Se non coincidono, rinumeriamo
        if not np.array_equal(classes, expected):
            print(f"[WARN] Classi trovate: {classes}")
            print(f"[INFO] Le converto automaticamente in range 0..{len(classes) - 1}")

            # Mapping ordinato: classe_originale -> nuova_classe
            mapping = {cls: i for i, cls in enumerate(sorted(classes))}

            # Applichiamo il mapping
            df[target_col] = df[target_col].map(mapping)

            print(f"[OK] Mapping applicato: {mapping}")
        else:
            print("[OK] Le classi sono già nel formato corretto (0..N-1)")

    # Define features (X) and target (y)
    print("Extracting features and target variable...")

    df.columns = range(len(df.columns))

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    sample_size = X.shape[1]

    if task == 0:
        n_classes = y.nunique()
        print(f"Number of classes: {n_classes}")
    else:
        n_classes = 2

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
        y_q_test.head(test_size).to_csv(class_path, index=False)

    if joblib_exists == 1:
        print("A Joblib already exists!")
        f_model = load(joblib_path)
    else:
        if task == 0:
            if n_classes > 2:
                xgb = XGBClassifier(
                    objective="multi:softprob",
                    n_estimators=n_trees,
                    num_class=n_classes,
                    random_state=random_seed,
                    n_jobs=-1,
                    tree_method="hist",
                )
            else:
                xgb = XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=n_trees,
                    random_state=random_seed,
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

            # Perform grid search with 5-fold cross-validation
            print("Performing grid search with 5-fold cross-validation...")
            grid_search = GridSearchCV(
                xgb, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
            )
            grid_search.fit(X_q_train, y_q_train)

            # Best parameters and score
            print(f"Best parameters found: {grid_search.best_params_}")
            print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

            # Train final model using the best parameters
            print("Training the final model with the best parameters...")

            if n_classes > 2:
                f_model = XGBClassifier(
                    n_estimators=n_trees,  # numero di alberi
                    learning_rate=grid_search.best_params_[
                        "learning_rate"
                    ],  # shrinkage
                    max_depth=grid_search.best_params_[
                        "max_depth"
                    ],  # profondità massima degli alberi
                    subsample=grid_search.best_params_[
                        "subsample"
                    ],  # per regolarizzare
                    reg_alpha=grid_search.best_params_["reg_alpha"],
                    gamma=grid_search.best_params_["gamma"],
                    reg_lambda=grid_search.best_params_["reg_lambda"],
                    objective="multi:softprob",
                    n_jobs=-1,
                    num_class=n_classes,
                    random_state=random_seed,
                    tree_method="hist",  # veloce; su GPU puoi fare "gpu_hist"
                )
            else:
                f_model = XGBClassifier(
                    n_estimators=n_trees,  # numero di alberi
                    learning_rate=grid_search.best_params_[
                        "learning_rate"
                    ],  # shrinkage
                    max_depth=grid_search.best_params_[
                        "max_depth"
                    ],  # profondità massima degli alberi
                    subsample=grid_search.best_params_[
                        "subsample"
                    ],  # per regolarizzare
                    reg_alpha=grid_search.best_params_["reg_alpha"],
                    gamma=grid_search.best_params_["gamma"],
                    reg_lambda=grid_search.best_params_["reg_lambda"],
                    objective="binary:logistic",  # classificazione binaria
                    n_jobs=-1,
                    random_state=random_seed,
                    tree_method="hist",  # veloce; su GPU puoi fare "gpu_hist"
                )
            f_model.fit(X_q_train, y_q_train)

            dump(f_model, joblib_path)

        else:
            xgb = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=n_trees,
                random_state=random_seed,
                n_jobs=-1,
                tree_method="hist",
            )

            param_dist = {
                "learning_rate": uniform(0.02, 0.08),  # ~ [0.02, 0.10]
                "max_depth": randint(2, 6),  # {2,3,4,5}
                "min_child_weight": randint(1, 15),  # molto influente
                "gamma": uniform(0.0, 0.5),
                "subsample": uniform(0.6, 0.4),  # ~ [0.6, 1.0]
                "colsample_bytree": uniform(0.6, 0.4),
                "reg_alpha": uniform(0.0, 0.3),
                "reg_lambda": uniform(0.5, 3.0),
            }

            print("Performing RANDOMIZED search with 5-fold CV...")
            rand_search = RandomizedSearchCV(
                estimator=xgb,
                param_distributions=param_dist,
                n_iter=60,  # ← 40–80 è lo sweet spot
                cv=5,
                scoring="neg_mean_squared_error",
                random_state=random_seed,
                n_jobs=-1,
                verbose=1,
            )
            rand_search.fit(X_q_train, y_q_train)
            best_params = rand_search.best_params_
            print(f"Best parameters found: {best_params}")
            print(f"Best cross-validation score: {rand_search.best_score_:.4f}")

            print("Training the final model with the best parameters...")
            f_model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=n_trees,  # stesso valore usato nella grid
                random_state=random_seed,
                n_jobs=-1,
                tree_method="hist",
                **best_params,  # <-- QUI la grid search
            )

            f_model.fit(X_q_train, y_q_train)

            dump(f_model, joblib_path)

    # Evaluate on the test set
    y_pred = f_model.predict(X_q_test)

    if task == 0:
        acc = accuracy_score(y_q_test, y_pred)
    else:
        mae = mean_absolute_error(y_q_test, y_pred)
        mse = mean_squared_error(y_q_test, y_pred)
        r2 = r2_score(y_q_test, y_pred)

    return (
        test_path,
        joblib_path,
        sample_size,
        task,
        acc,
        mse,
        mae,
        r2,
        csv_stem,
        n_classes,
    )
