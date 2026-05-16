from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
