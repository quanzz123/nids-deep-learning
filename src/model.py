from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2


def build_model(input_dim):
    """Build a Sequential DNN: 800-800-400 hidden layers, ReLU, Dropout 0.2, L2 reg, Softmax output."""
    model = Sequential([
        Dense(800, activation='relu', kernel_regularizer=l2(1e-4), input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(800, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(400, activation='relu', kernel_regularizer=l2(1e-4)),
        Dense(2, activation='softmax')
    ])
    return model
