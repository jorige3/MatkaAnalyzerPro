# engines/ml_predictor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from scoring.utils import validate_df  # Reuse!

class MLPredictor:
    def __init__(self, df_path='data/sridevi.csv'):
        self.df = validate_df(pd.read_csv(df_path))
        self.model = None
        self.le = LabelEncoder()
    
    def engineer_features(self):
        df = self.df.copy()
        df['jodi_int'] = df['Jodi'].astype(int)
        # Ensure Jodi is a two-character string for consistent digit extraction
        df['Jodi_str'] = df['Jodi'].astype(str).str.zfill(2)

        # Assuming 'open' and 'close' columns exist or can be derived from 'Jodi_str'
        # For now, let's derive them from 'Jodi_str' string if they don't exist
        if 'open' not in df.columns:
            df['open'] = df['Jodi_str'].apply(lambda x: int(x[0]))
        if 'close' not in df.columns:
            df['close'] = df['Jodi_str'].apply(lambda x: int(x[1]))

        df['sum'] = df['open'].fillna(0).astype(int) + df['close'].fillna(0).astype(int)
        df['digit1'] = df['jodi_int'] // 10
        df['digit2'] = df['jodi_int'] % 10
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['lag1'] = df['jodi_int'].shift(1)
        return df.dropna()
    
    def train(self):
        df_feat = self.engineer_features()
        features = ['sum', 'digit1', 'digit2', 'dayofweek', 'lag1']
        X = df_feat[features]
        y = self.le.fit_transform(df_feat['Jodi'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = XGBClassifier(random_state=42).fit(X_train, y_train)
        acc = self.model.score(X_test, y_test)
        print(f"ML Accuracy: {acc:.2%}")
        return acc
    
    def predict_top(self, n=10):
        if self.model is None:
            self.train()
        df_feat = self.engineer_features()
        features = ['sum', 'digit1', 'digit2', 'dayofweek', 'lag1']
        
        # Ensure that the latest features DataFrame has the same features
        # as the one used for training.
        latest_data_features = df_feat.iloc[-1:][features]

        # Use predict_proba only if model supports it, else use predict
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(latest_data_features)
            top_idx = np.argsort(probs[0])[::-1][:n]
        else:
            # Fallback for models without predict_proba (though XGBoost has it)
            # This path is less ideal for 'predict_top' as it doesn't give probabilities
            preds = self.model.predict(latest_data_features)
            # If using predict, it would typically return a single prediction,
            # so predicting 'top n' would need a different approach.
            # For simplicity, we'll return the single prediction.
            # This section might need more thought if 'predict_proba' is truly unavailable.
            return self.le.inverse_transform(preds)

        return self.le.inverse_transform(top_idx)
