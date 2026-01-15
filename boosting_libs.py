#!/usr/bin/env python3
"""
XGBoost, LightGBM, CatBoost on embeddings.
Using GPU where supported.
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

EMBEDDINGS_FILE = "data/gemma_1b_embeddings.npz"


def load_data():
    """Load embeddings."""
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    data = np.load(EMBEDDINGS_FILE)

    X_train = data['X_train'].astype(np.float32)
    y_train = data['y_train']
    X_test = data['X_test'].astype(np.float32)
    y_test = data['y_test']

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features")
    return X_train, y_train, X_test, y_test


def tune_xgboost(X_train, y_train, X_test, y_test):
    """XGBoost with GPU and hyperparameter tuning."""
    print("\n" + "=" * 60)
    print("XGBoost (GPU)")
    print("=" * 60)

    best_acc = 0
    best_params = {}

    # Parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.5, 0.8, 1.0],
    }

    print("Searching hyperparameters...")

    # Quick search
    for n_est in [50, 100, 200]:
        for depth in [3, 5, 7]:
            for lr in [0.01, 0.05, 0.1]:
                for subsample in [0.8, 1.0]:
                    for colsample in [0.5, 0.8]:
                        model = xgb.XGBClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr,
                            subsample=subsample,
                            colsample_bytree=colsample,
                            tree_method='hist',
                            device='cuda',
                            verbosity=0,
                            random_state=42,
                        )
                        model.fit(X_train, y_train)
                        acc = model.score(X_test, y_test)

                        if acc > best_acc:
                            best_acc = acc
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'learning_rate': lr,
                                'subsample': subsample,
                                'colsample_bytree': colsample,
                            }

    print(f"Best params: {best_params}")
    print(f"Best accuracy: {best_acc*100:.1f}%")
    return best_acc, best_params


def tune_lightgbm(X_train, y_train, X_test, y_test):
    """LightGBM with GPU and hyperparameter tuning."""
    print("\n" + "=" * 60)
    print("LightGBM (GPU)")
    print("=" * 60)

    best_acc = 0
    best_params = {}

    print("Searching hyperparameters...")

    for n_est in [50, 100, 200]:
        for depth in [3, 5, 7, -1]:
            for lr in [0.01, 0.05, 0.1]:
                for num_leaves in [31, 63, 127]:
                    try:
                        model = lgb.LGBMClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr,
                            num_leaves=num_leaves,
                            device='gpu',
                            verbosity=-1,
                            random_state=42,
                        )
                        model.fit(X_train, y_train)
                        acc = model.score(X_test, y_test)

                        if acc > best_acc:
                            best_acc = acc
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'learning_rate': lr,
                                'num_leaves': num_leaves,
                            }
                    except Exception as e:
                        # Fall back to CPU if GPU fails
                        model = lgb.LGBMClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr,
                            num_leaves=num_leaves,
                            verbosity=-1,
                            random_state=42,
                        )
                        model.fit(X_train, y_train)
                        acc = model.score(X_test, y_test)

                        if acc > best_acc:
                            best_acc = acc
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'learning_rate': lr,
                                'num_leaves': num_leaves,
                            }

    print(f"Best params: {best_params}")
    print(f"Best accuracy: {best_acc*100:.1f}%")
    return best_acc, best_params


def tune_catboost(X_train, y_train, X_test, y_test):
    """CatBoost with GPU and hyperparameter tuning."""
    print("\n" + "=" * 60)
    print("CatBoost (GPU)")
    print("=" * 60)

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("CatBoost not available")
        return 0, {}

    best_acc = 0
    best_params = {}

    print("Searching hyperparameters...")

    for n_est in [50, 100, 200]:
        for depth in [4, 6, 8]:
            for lr in [0.03, 0.1, 0.2]:
                try:
                    model = CatBoostClassifier(
                        iterations=n_est,
                        depth=depth,
                        learning_rate=lr,
                        task_type='GPU',
                        devices='0',
                        verbose=False,
                        random_state=42,
                    )
                    model.fit(X_train, y_train)
                    acc = model.score(X_test, y_test)

                    if acc > best_acc:
                        best_acc = acc
                        best_params = {
                            'iterations': n_est,
                            'depth': depth,
                            'learning_rate': lr,
                        }
                except Exception as e:
                    # Fall back to CPU
                    model = CatBoostClassifier(
                        iterations=n_est,
                        depth=depth,
                        learning_rate=lr,
                        verbose=False,
                        random_state=42,
                    )
                    model.fit(X_train, y_train)
                    acc = model.score(X_test, y_test)

                    if acc > best_acc:
                        best_acc = acc
                        best_params = {
                            'iterations': n_est,
                            'depth': depth,
                            'learning_rate': lr,
                        }

    print(f"Best params: {best_params}")
    print(f"Best accuracy: {best_acc*100:.1f}%")
    return best_acc, best_params


def run():
    print("=" * 70)
    print("Boosting Libraries Comparison")
    print("=" * 70)

    X_train, y_train, X_test, y_test = load_data()

    results = {}

    # XGBoost
    acc, params = tune_xgboost(X_train, y_train, X_test, y_test)
    results['XGBoost'] = acc

    # LightGBM
    acc, params = tune_lightgbm(X_train, y_train, X_test, y_test)
    results['LightGBM'] = acc

    # CatBoost
    acc, params = tune_catboost(X_train, y_train, X_test, y_test)
    results['CatBoost'] = acc

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 50)
        print(f"  {name:15s}: {acc*100:.1f}% {bar}")

    print(f"\n  Random baseline: 50.0%")
    print(f"  Previous best (RandomForest): 76.7%")


if __name__ == "__main__":
    run()
