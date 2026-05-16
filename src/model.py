from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, Flatten,
    BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2


def build_model(input_dim=8):
    """Build a Sequential DNN: 400-800-800-400, ReLU, Dropout 0.2, L2 on first layer, Softmax output."""
    model = Sequential([
        Dense(400, activation='relu', kernel_regularizer=l2(1e-4), input_shape=(input_dim,)),
        Dense(800, activation='relu'),
        Dropout(0.2), 
        Dense(800, activation='relu'),
        Dropout(0.2),
        Dense(400, activation='relu'),
        Dense(2, activation='softmax')
    ])
    return model


def build_cnn_model(input_dim):
    """Build an improved 1D-CNN model for tabular NIDS features.

    Improvements over baseline:
    - padding='same' preserves spatial dimensions across Conv layers
    - BatchNormalization stabilizes training after each Conv block
    - GlobalAveragePooling1D replaces aggressive MaxPooling x2 + Flatten
    - L2 regularization on Conv and Dense layers
    - Intermediate Dense layers reduce large representation jumps
    - Dropout reduced from 0.5 to 0.2-0.3
    """
    model = Sequential([
        # Conv Block 1
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l2(1e-4), input_shape=(input_dim, 1)),
        BatchNormalization(),
        Dropout(0.2),

        # Conv Block 2
        Conv1D(filters=128, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),

        # Global pooling — preserves all spatial info, no dimension loss
        GlobalAveragePooling1D(),

        # Dense head with gradual dimension reduction
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),

        # Output
        Dense(2, activation='softmax')
    ])
    return model
