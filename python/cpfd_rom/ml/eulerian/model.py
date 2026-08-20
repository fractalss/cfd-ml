# Modified model.py with parameter injection in latent space (concatenation logic)
import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers, callbacks

# Enable XLA acceleration
tf.config.optimizer.set_jit(True)

def build_conditional_autoencoder(input_shape, latent_dim=128):  # Increased latent_dim from 64 to 128
    # CFD field input
    input_field = layers.Input(shape=input_shape, name="field_input")

    # Operating parameter input (e.g., velocity, volume fraction)
    input_param = layers.Input(shape=(1,), name="parameter_input")

    # Encoder
    x = layers.Conv1D(32, 3, activation='elu', padding='same')(input_field)  # Changed activation to 'elu'
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(16, 3, activation='elu', padding='same')(x)  # Changed activation to 'elu'
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(8, 3, activation='elu', padding='same')(x)  # Added one more Conv1D layer

    # Flatten and project to latent space
    flat = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation='elu')(flat)  # Changed activation to 'elu'

    # Inject parameter via concatenation
    param_encoded = layers.Dense(latent_dim, activation='elu')(input_param)  # Changed activation to 'elu'
    conditioned_latent = layers.Concatenate()([latent, param_encoded])
    conditioned_latent = layers.Dense(latent_dim, activation='elu')(conditioned_latent)  # Changed activation to 'elu'

    # Decoder
    x_shape = x.shape[1:]
    latent_units = int(np.prod(x_shape))
    x = layers.Dense(latent_units, activation='elu')(conditioned_latent)  # Changed activation to 'elu'
    x = layers.Reshape(x_shape)(x)
    x = layers.Conv1D(8, 3, activation='elu', padding='same')(x)  # Changed activation to 'elu'
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(16, 3, activation='elu', padding='same')(x)  # Changed activation to 'elu'
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, activation='elu', padding='same')(x)  # Changed activation to 'elu'
    output = layers.Conv1D(1, 3, activation='linear', padding='same')(x)

    model = models.Model(inputs=[input_field, input_param], outputs=output)
    model.compile(optimizer='adam', loss='mse')

    return model

def train_autoencoder(X_train, X_val, P_train, P_val, epochs=100, batch_size=16):
    model = build_conditional_autoencoder(input_shape=X_train.shape[1:])
    model.summary()
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        x={"field_input": X_train, "parameter_input": P_train},
        y=X_train,
        validation_data=({"field_input": X_val, "parameter_input": P_val}, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history
