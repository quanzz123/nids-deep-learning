import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


def load_and_clean_data(train_path, test_path):
    """Load CSV datasets and clean invalid values in the service column."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df['service'] = train_df['service'].replace('-', np.nan)
    train_df = train_df.dropna()
    test_df['service'] = test_df['service'].replace('-', np.nan)
    test_df = test_df.dropna()

    return train_df, test_df


def encode_features(train_df, test_df):
    """Apply One-Hot Encoding on proto, service, state columns."""
    train_len = len(train_df)

    combined = pd.concat([train_df, test_df], axis=0)
    combined = pd.get_dummies(combined, columns=['proto', 'service', 'state'])

    train_encoded = combined.iloc[:train_len].reset_index(drop=True)
    test_encoded = combined.iloc[train_len:].reset_index(drop=True)

    y_train = train_encoded['label'].values
    y_test = test_encoded['label'].values

    X_train = train_encoded.drop(columns=['attack_cat', 'label'])
    X_test = test_encoded.drop(columns=['attack_cat', 'label'])

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Normalize numerical features to [0,1] range using MinMaxScaler."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train):
    """Balance training dataset using SMOTE to achieve 1:1 class ratio."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
