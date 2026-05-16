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
    print_metrics_table
)


DATA_DIR = 'dataset'
TRAIN_PATH = os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv')
TEST_PATH = os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv')


def main():
    """Orchestrate the full NIDS pipeline: load, preprocess, feature selection, train, evaluate."""
    train_df, test_df = load_and_clean_data(TRAIN_PATH, TEST_PATH)
    X_train, X_test, y_train, y_test = encode_features(train_df, test_df)
    feature_names = list(X_train.columns)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)
    X_train_selected, selected_features, etc = select_features(
        X_train_balanced, y_train_balanced, feature_names
    )
    test_indices = [feature_names.index(name) for name in selected_features]
    X_test_selected = X_test_scaled[:, test_indices]
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_selected, y_train_balanced,
        test_size=0.33,
        random_state=42
    )
    model = build_model(X_train_split.shape[1])
    history = compile_and_train(model, X_train_split, y_train_split, X_val_split, y_val_split)
    results, y_pred = evaluate(model, X_test_selected, y_test)

    print_metrics_table(results)

    plot_loss_curve(history)
    plot_confusion_matrix(results['confusion_matrix'])

    importances_dict = dict(zip(feature_names, etc.feature_importances_))
    plot_feature_importance(selected_features, importances_dict)

    print('Selected features:', selected_features)
    print('Feature importance scores:')
    for feat in selected_features:
        print(f'  {feat}: {importances_dict[feat]:.6f}')

    os.makedirs('models', exist_ok=True)
    model.save(os.path.join('models', 'nids_model.h5'))
    print('Model saved to models/nids_model.h5')


if __name__ == '__main__':
    main()
