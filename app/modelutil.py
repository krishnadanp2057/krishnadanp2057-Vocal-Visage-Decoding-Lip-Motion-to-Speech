import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

def load_model() -> Sequential:
    model = Sequential()

    # 3D Convolutional Layers
    model.add(Conv3D(128, kernel_size=3, padding='same', input_shape=(75, 46, 140, 1)))
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(256, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(75, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    # Flatten time-distributed 3D features to feed into RNN
    model.add(TimeDistributed(Flatten()))

    # BiLSTM Layers
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    # Final Classification Layer
    model.add(Dense(41, activation='softmax', kernel_initializer='he_normal'))

    # Load weights from the checkpoint file
    checkpoint_path = os.path.join('..', 'models', 'checkpoint')
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    return model
