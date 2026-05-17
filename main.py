import os

from src.preprocessing import load_and_split_data, encode_features, scale_features, apply_smote
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
DATA_PATH = os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv')


def main():
    """Orchestrate the full NIDS pipeline using a single dataset (80/20 split)."""
    train_df, test_df = load_and_split_data(DATA_PATH, test_size=0.2)
    X_train, X_test, y_train, y_test = encode_features(train_df, test_df)
    feature_names = list(X_train.columns)

    # Scale features (fit on full train, transform test)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Apply SMOTE on the full training data
    X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)

    # Feature selection (using balanced training data)
    _, selected_features, etc = select_features(
        X_train_balanced, y_train_balanced, feature_names
    )

    # Filter features for all sets
    indices = [feature_names.index(name) for name in selected_features]
    X_train_final = X_train_balanced[:, indices]
    X_test_final = X_test_scaled[:, indices]

    # Train — validation_split=0.33 is handled inside compile_and_train
    model = build_model(X_train_final.shape[1])
    history = compile_and_train(model, X_train_final, y_train_balanced)
    results, y_pred = evaluate(model, X_test_final, y_test)

    print_metrics_table(results)

    plot_loss_curve(history)
    plot_confusion_matrix(results['confusion_matrix'])

    importances_dict = dict(zip(feature_names, etc.feature_importances_))
    plot_feature_importance(selected_features, importances_dict)

    save_report_txt(results, selected_features, importances_dict, model=model)

    os.makedirs('models', exist_ok=True)
    model.save(os.path.join('models', 'nids_model.h5'))
    print('Model saved to models/nids_model.h5')


if __name__ == '__main__':
    main()
