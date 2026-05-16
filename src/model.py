from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.regularizers import l2


def build_model(input_dim=8):
    """Build a Sequential DNN: 400-800-800-400, ReLU, Dropout 0.2, L2 on first layer, Softmax output."""
    model = Sequential([
        Dense(400, activation='relu', kernel_regularizer=l2(1e-4), input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(800, activation='relu'),
        Dense(800, activation='relu'),
        Dropout(0.2),
        Dense(400, activation='relu'),
        Dense(2, activation='softmax')
    ])
    return model


def build_cnn_model(input_dim):
    """Build a 1D-CNN model for 43-feature input."""
    model = Sequential([
        # Input shape: (43, 1)
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model
