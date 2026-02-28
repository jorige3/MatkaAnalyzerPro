
"""
A refactored, statistically rigorous ML predictor with a walk-forward
validation pipeline designed to prevent data leakage.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
from tqdm import tqdm
from typing import Tuple, Dict, Any

class MLPredictor:
    """
    A refactored ML predictor with a strong focus on preventing data leakage
    in a time-series context. It uses a simplified feature set and includes
    a shuffle test for sanity checking the validation pipeline.
    """
    def __init__(self):
        """
        Initializes the MLPredictor with a lightweight XGBoost configuration
        as specified for the refactor.
        """
        self.model_params = {
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'n_jobs': -1
        }
        self.model = XGBClassifier(**self.model_params)

    def _engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Creates a simple, leak-proof feature set based on lag values.
        This function is safe to call on the entire dataset at once because it
        only uses past data (`shift(k)` where k > 0) to create features.
        """
        # Rule A.1: Convert jodi to integer immediately.
        df['jodi'] = pd.to_numeric(df['jodi'], errors='coerce')
        df = df.dropna(subset=['jodi'])
        df['jodi'] = df['jodi'].astype(int)

        feature_df = df.copy()

        # Rule A.2: Create lag features from 1 to 7.
        lag_cols = []
        for k in range(1, 8):
            col_name = f'lag_{k}'
            # Rule A.2 / A.3: Use shift(k) for k > 0.
            feature_df[col_name] = feature_df['jodi'].shift(k)
            lag_cols.append(col_name)

        # Rule A.5: Drop rows with NaN after lag creation.
        feature_df = feature_df.dropna(subset=lag_cols).reset_index(drop=True)
        
        # Rule A.6: Ensure lag columns are numeric.
        feature_df[lag_cols] = feature_df[lag_cols].astype(int)
        
        X = feature_df[lag_cols]
        y = feature_df['jodi']
        
        # Rule A.4: No feature references the current row's jodi.
        # This is inherently true as X only contains lag columns.
        
        return X, y

    def walk_forward_validation(self, df: pd.DataFrame, min_train_size: int = 150) -> Dict[str, Any]:
        """
        Performs a strict, expanding-window walk-forward validation.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
            
        X, y = self._engineer_features(df.copy())
        
        if len(X) <= min_train_size:
            print("Not enough data for walk-forward validation after feature engineering.")
            return {}

        metrics = {'hits': 0, 'top_5_hits': 0, 'log_losses': [], 'skipped': 0}
        
        iterator = tqdm(range(min_train_size, len(X)), desc="Walk-Forward Validation")
        for i in iterator:
            # Rule B: Strict train/test split for this window.
            X_train, y_train = X.iloc[:i], y.iloc[:i]
            X_test, y_test_actual_series = X.iloc[[i]], y.iloc[[i]]
            y_test_actual = y_test_actual_series.iloc[0]

            # Rule C: Per-window label encoding.
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)

            if y_test_actual not in le.classes_:
                metrics['skipped'] += 1
                continue
            
            # Train the model
            model = XGBClassifier(**self.model_params)
            model.fit(X_train, y_train_encoded)
            
            # Predict and evaluate
            probas = model.predict_proba(X_test)
            
            y_pred_encoded = np.argmax(probas, axis=1)
            
            # Top-1 Accuracy
            if le.inverse_transform(y_pred_encoded)[0] == y_test_actual:
                metrics['hits'] += 1
            
            # Top-5 Accuracy
            top_5_preds_encoded = np.argsort(probas, axis=1)[0, -5:]
            top_5_preds = le.inverse_transform(top_5_preds_encoded)
            if y_test_actual in top_5_preds:
                metrics['top_5_hits'] += 1

            # Log Loss
            probas_clipped = np.clip(probas, 1e-15, 1 - 1e-15)
            y_test_encoded = le.transform([y_test_actual])
            loss = log_loss(y_test_encoded, probas_clipped, labels=le.transform(le.classes_))
            metrics['log_losses'].append(loss)
        
        total_predictions = len(X) - min_train_size - metrics['skipped']
        if total_predictions == 0:
            print("No predictions were made (all test labels were unseen in training).")
            return {}

        # --- Rule G: Output Final Metrics ---
        accuracy = metrics['hits'] / total_predictions
        top_5_accuracy = metrics['top_5_hits'] / total_predictions
        avg_log_loss = np.mean(metrics['log_losses'])
        
        # In a 100-class problem, num_classes is effectively 100.
        num_classes = 100 

        print("\n--- Walk-Forward Validation Results ---")
        print(f"Total Predictions Made: {total_predictions}")
        print(f"Skipped Iterations (unseen test label): {metrics['skipped']}")
        print(f"Top-1 Accuracy: {accuracy:.2%}")
        print(f"Top-5 Accuracy: {top_5_accuracy:.2%}")
        print(f"Mean Log Loss: {avg_log_loss:.4f}")
        print("---------------------------------------")
        
        # --- Rule F: Baseline Comparison ---
        print("\n--- Baseline Comparison ---")
        print(f"Random Baseline Top-1: {1/num_classes:.2%}")
        print(f"Random Baseline Top-5: {5/num_classes:.2%}")
        print(f"Random Baseline Log Loss (approx): {np.log(num_classes):.4f}")
        print("---------------------------------------")
        
        return {"accuracy": accuracy}

    def shuffle_test(self, df: pd.DataFrame) -> float:
        """
        Rule E: Runs walk-forward validation on a shuffled target to check for
        pipeline-induced data leakage. The accuracy should be close to random chance.
        """
        print("\n--- Running Shuffle Test for Data Leakage ---")
        df_shuffled = df.copy()
        
        # Shuffle the target 'jodi' column to break any real time-series patterns
        df_shuffled['jodi'] = np.random.permutation(df_shuffled['jodi'].values)
        
        print("Target column 'jodi' has been shuffled.")
        results = self.walk_forward_validation(df_shuffled)
        
        shuffle_accuracy = results.get("accuracy", 0.0)

        print("\n--- Shuffle Test Results ---")
        print(f"Shuffle Test Accuracy: {shuffle_accuracy:.2%}")
        
        # A generous threshold. For a 100-class problem, anything >3% is suspicious.
        if shuffle_accuracy > 0.03:
            print("\nWARNING: LEAKAGE DETECTED!")
            print("Accuracy on shuffled data is unexpectedly high. Review feature engineering.")
        else:
            print("\nSUCCESS: No significant data leakage detected by the shuffle test.")
        print("---------------------------------------")
            
        return shuffle_accuracy
