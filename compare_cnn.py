import os
import numpy as np

from src.preprocessing import load_and_clean_data, encode_features, scale_features, apply_smote
from src.model import build_cnn_model
from src.train import compile_and_train
from src.evaluate import (
    evaluate,
    plot_loss_curve,
    plot_confusion_matrix,
    print_metrics_table,
    save_report_txt,
)
import src.evaluate as ev

DATA_DIR = 'dataset'
TRAIN_PATH = os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv')
TEST_PATH = os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv')
REPORTS_DIR = 'reports_cnn'


def main():
    """Comparison Pipeline: 1D-CNN using all features (no feature selection)."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load & clean data
    train_df, test_df = load_and_clean_data(TRAIN_PATH, TEST_PATH)
    X_train, X_test, y_train, y_test = encode_features(train_df, test_df)

    # 2. Scale features (fit on full train, transform test)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 3. Split BEFORE SMOTE — val must be real data, not synthetic
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_train_scaled, y_train,
        test_size=0.33,
        random_state=42,
        stratify=y_train
    )

    # 4. Apply SMOTE ONLY on the 67% training split
    X_train_balanced, y_train_balanced = apply_smote(X_train_raw, y_train_raw)

    # 5. Reshape for 1D-CNN: (samples, features, 1)
    X_train_cnn = np.expand_dims(X_train_balanced, axis=-1)
    X_val_cnn = np.expand_dims(X_val_raw, axis=-1)
    X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)

    # 6. Build model
    input_dim = X_train_cnn.shape[1]
    model = build_cnn_model(input_dim)

    # 7. Train — validation_data uses real pre-SMOTE samples
    print(f"Training 1D-CNN with {input_dim} features...")
    history = compile_and_train(model, X_train_cnn, y_train_balanced, X_val_cnn, y_val_raw)


    # 7. Evaluate on test set
    results, y_pred = evaluate(model, X_test_cnn, y_test)
    print_metrics_table(results)

    # 8. Save plots and report to reports_cnn/
    ev.REPORTS_DIR = REPORTS_DIR
    plot_loss_curve(history)
    plot_confusion_matrix(results['confusion_matrix'])
    save_report_txt(results, [], {})  # No feature selection for CNN
    ev.REPORTS_DIR = 'reports'        # Reset to default

    # 9. Save model
    os.makedirs('models', exist_ok=True)
    model.save(os.path.join('models', 'nids_cnn_model.h5'))
    print(f'CNN Model saved to models/nids_cnn_model.h5. Reports saved in {REPORTS_DIR}/')


if __name__ == '__main__':
    main()
