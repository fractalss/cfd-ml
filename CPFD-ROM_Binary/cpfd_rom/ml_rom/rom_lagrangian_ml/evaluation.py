# evaluation.py for rom_lagrangian_ml

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, GlobalMaxPooling1D, concatenate, Layer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cpfd_rom.util.layers import ExpandDimsLayer, TileLayer


def create_pointnet_model(n_points, n_features, ExpandDimsLayerCls=ExpandDimsLayer, TileLayerCls=TileLayer):
    input_layer = Input(shape=(n_points, n_features))
    x = Conv1D(64, 1, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    skip_connection = x
    x = Conv1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    global_feat = GlobalMaxPooling1D()(x)

    # x_global = Lambda(lambda t: tf.expand_dims(t, axis=1))(global_feat)
    # x_global_tiled = Lambda(lambda t: tf.tile(t, [1, n_points, 1]))(x_global)
    x_global = ExpandDimsLayerCls()(global_feat)
    x_global_tiled = TileLayerCls(n_points)(x_global)
    x_concat = concatenate([input_layer, skip_connection, x_global_tiled])
    x = Conv1D(128, 1, activation='relu')(x_concat)
    x = BatchNormalization()(x)
    x = Conv1D(64, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    output_layer = Conv1D(n_features, 1, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model


def train_pointnet_model(model, data_scaled, epochs=100, batch_size=2):
    X_train, X_val = train_test_split(data_scaled, test_size=0.2, random_state=42)
    es = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_val, X_val), callbacks=[es])
    return model

