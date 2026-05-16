from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def compile_and_train(model, X_train, y_train, X_val, y_val):
    """Compile with Adam (lr=0.001) and train for 100 epochs, batch size 50.

    Validation is performed on real (pre-SMOTE) data passed as validation_data,
    not on the SMOTE-balanced training set, to avoid misleading validation metrics.
    """
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)

    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=100,
        batch_size=50,
        verbose=1
    )

    return history
