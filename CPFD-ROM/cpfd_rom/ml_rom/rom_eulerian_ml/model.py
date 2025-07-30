from tensorflow.keras import models, layers, callbacks


def build_autoencoder(input_shape):
    input_layer = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(2, padding='same')(x)

    # Decoder
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    decoded = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def train_autoencoder(X_train, X_val, epochs=100, batch_size=16):
    model = build_autoencoder(input_shape=X_train.shape[1:])
    model.summary()
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=[early_stop],
        verbose=1
    )

    return model, history
