import os
from sklearn.model_selection import train_test_split

from src.preprocessing import load_and_clean_data, encode_features, scale_features, apply_smote
from src.feature_selection import select_features
from src.model import build_model
from src.train import compile_and_train
from src.evaluate import (
    evaluate,
    plot_loss_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    print_metrics_table,
    save_report_txt,
)


DATA_DIR = 'dataset'
TRAIN_PATH = os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv')
TEST_PATH = os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv')


def main():
    """Orchestrate the full NIDS pipeline: load, preprocess, feature selection, train, evaluate."""
    train_df, test_df = load_and_clean_data(TRAIN_PATH, TEST_PATH)
    X_train, X_test, y_train, y_test = encode_features(train_df, test_df)
    feature_names = list(X_train.columns)
    # Split original training set into train and val
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    # Scale features
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train_raw, X_val_raw)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE only to training split
    X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train_raw)

    # Feature selection (using balanced training data)
    _, selected_features, etc = select_features(
        X_train_balanced, y_train_balanced, feature_names
    )

    # Filter features for all sets
    test_indices = [feature_names.index(name) for name in selected_features]
    X_train_final = X_train_balanced[:, test_indices]
    X_val_final = X_val_scaled[:, test_indices]
    X_test_final = X_test_scaled[:, test_indices]

    model = build_model(X_train_final.shape[1])
    history = compile_and_train(model, X_train_final, y_train_balanced, X_val_final, y_val_raw)
    results, y_pred = evaluate(model, X_test_final, y_test)

    print_metrics_table(results)

    plot_loss_curve(history)
    plot_confusion_matrix(results['confusion_matrix'])

    importances_dict = dict(zip(feature_names, etc.feature_importances_))
    plot_feature_importance(selected_features, importances_dict)

    save_report_txt(results, selected_features, importances_dict)

    os.makedirs('models', exist_ok=True)
    model.save(os.path.join('models', 'nids_model.h5'))
    print('Model saved to models/nids_model.h5')


if __name__ == '__main__':
    main()
