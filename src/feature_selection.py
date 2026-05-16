from sklearn.ensemble import ExtraTreesClassifier

REQUIRED_FEATURES = ['sttl', 'dttl', 'ct_state_ttl', 'dload', 'swin', 'ct_srv_dst', 'id']


def select_features(X_train, y_train, feature_names):
    """Select top 8 features using Extra Trees Classifier, ensuring required features are included."""
    etc = ExtraTreesClassifier(random_state=42)
    etc.fit(X_train, y_train)

    state_cols = [name for name in feature_names if name.startswith('state_')]

    selected = []
    selected_set = set()

    for feat in REQUIRED_FEATURES:
        if feat == 'state':
            if state_cols:
                best_state = max(
                    state_cols,
                    key=lambda c: etc.feature_importances_[feature_names.index(c)]
                )
                if best_state not in selected_set:
                    selected.append(best_state)
                    selected_set.add(best_state)
        else:
            if feat in feature_names and feat not in selected_set:
                selected.append(feat)
                selected_set.add(feat)

    importances = sorted(
        zip(feature_names, etc.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )

    for name, _ in importances:
        if len(selected) >= 8:
            break
        if name not in selected_set:
            selected.append(name)
            selected_set.add(name)

    indices = [feature_names.index(name) for name in selected]

    X_train_selected = X_train[:, indices]

    return X_train_selected, selected, etc
