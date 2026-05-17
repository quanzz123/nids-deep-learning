from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def compile_and_train(model, X_train, y_train):
    """Compile with Adam (lr=0.001) and train for 100 epochs, batch size 50.
    
    Uses validation_split=0.33 as specified in the pseudo-code of rule.md.
    """
    y_train_cat = to_categorical(y_train, num_classes=2)

    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.33,
        epochs=100,
        batch_size=50,
        verbose=1
    )

    return history
