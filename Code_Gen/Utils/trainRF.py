# 
# Copyright (c) 2026 Lorenzo Abate <lorenzoabate510@gmail.com>.
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

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load

def model_stats(joblib_path, X_q_test, y_q_test, weight=8, fast=0, hyb=0, task=0):
    model = load(joblib_path)

    y_pred = model.predict(X_q_test)
    hybrid = 0
    if task == 0:
        acc = accuracy_score(y_q_test, y_pred)
    else:
        mse = mean_squared_error(y_q_test, y_pred)
        mae = mean_absolute_error(y_q_test, y_pred)
        r2  = r2_score(y_q_test, y_pred)

    print(f"Model Stats:")
    if fast == 0:
        total_nodes = 0
        size_bytes = 0
        size_kb = 0

        counter = 0
        for tree in model.estimators_:
            nodes = tree.tree_.node_count
            total_nodes += nodes

            size_bytes += nodes * weight
            
            if size_bytes > 129000 and hybrid == 0:
                hybrid = counter

            counter += 1
        
        size_kb += size_bytes / 1024

        # Results
        
        print(f"  - Total Nodes: {total_nodes}")
        if hyb == 0:  
            print(f"  - Dimension: {size_bytes} Bytes ({size_kb:.2f} KB)")

    if task == 0:    
        print(f"  - Accuracy: {acc:.4f}")
    else:
        print(f"  - MSE: {mse:.4f}")
        print(f"  - MAE: {mae:.4f}")
        print(f"  - R2:  {r2:.4f}")

    return hybrid

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

def detect_task_type(df, max_classes=50, class_ratio = 0.05):

    y = df.iloc[:, -1]   # ultima colonna = target
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

def training(csv_path, number_of_trees, max_depth, random_seed, test_size, weight=8, fast=0, hyb=0):
    csv_dir = csv_path.parent
    csv_stem = csv_path.stem

    joblib_name = f"{csv_stem}_RF_T{number_of_trees}_D{max_depth}_RS{random_seed}.joblib"
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
    y = df.iloc[:, -1]   # Last column

    sample_size = X.shape[1]

    print("Quantization...")
    scale = compute_columnwise_quantization_params(X)

    X_q = quantize_columnwise(X, scale)
    
    if task == 1:
        scale_reg = compute_columnwise_quantization_params(y)

        y_q = quantize_columnwise(y, scale_reg)

        X_q_train, X_q_test, y_q_train, y_q_test = train_test_split(X_q, y_q, test_size=0.3, random_state=random_seed)
    else:
        X_q_train, X_q_test, y_q_train, y_q_test = train_test_split(X_q, y, test_size=0.3, random_state=random_seed, stratify=y)

    if not test_path.exists():
        X_q_test.head(test_size).to_csv(test_path, index=False)

    if joblib_exists == 1:
        hybrid = 0
        print("A Joblib already exists!")
        hybrid = model_stats(joblib_path, X_q_test, y_q_test, weight, fast, hyb, task)
        return test_path, joblib_path, sample_size, hybrid, csv_stem, task

    if task == 0:
        model_cv = RandomForestClassifier(n_estimators=number_of_trees, max_depth=max_depth, random_state=random_seed, n_jobs=-1)
    else:
        model_cv = RandomForestRegressor(n_estimators=number_of_trees, max_depth=max_depth, random_state=random_seed, n_jobs=-1)

    param_grid = {
        "min_samples_split": [i for i in range(2,11)],
        "min_samples_leaf": [i for i in range (1, 11)]
    }

    # Perform grid search with 5-fold cross-validation
    print("Performing grid search with 5-fold cross-validation...")
    if task == 0:
        grid_search = GridSearchCV(model_cv, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    else:
        grid_search = GridSearchCV(model_cv, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
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
            random_state=random_seed
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
            random_state=random_seed
        )
        best_rf.fit(X_q_train, y_q_train)
    
    dump(best_rf, joblib_path)

    y_pred = best_rf.predict(X_q_test)

    if task == 0:
        accuracy = accuracy_score(y_q_test, y_pred)
    else:
        mse = mean_squared_error(y_q_test, y_pred)
        mae = mean_absolute_error(y_q_test, y_pred)
        r2  = r2_score(y_q_test, y_pred)
    
    print(f"Model Stats:")
    hybrid = 0
    if fast == 0:
        total_nodes = 0
        size_bytes = 0
        size_kb = 0

        counter = 0
        for tree in best_rf.estimators_:
            nodes = tree.tree_.node_count
            total_nodes += nodes

            size_bytes += nodes * weight
            
            if size_bytes > 129000 and hybrid == 0:
                hybrid = counter

            counter += 1

        size_kb += size_bytes / 1024

        # Results
        
        print(f"  - Total Nodes: {total_nodes}")
        if hyb == 0:    
            print(f"  - Dimension: {size_bytes} Bytes ({size_kb:.2f} KB)")
    if task == 0:    
        print(f"  - Accuracy: {accuracy:.4f}")
    else:
        print(f"  - MSE: {mse:.4f}")
        print(f"  - MAE: {mae:.4f}")
        print(f"  - R2:  {r2:.4f}")
    
    return test_path, joblib_path, sample_size, hybrid, csv_stem, task
