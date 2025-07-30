from cpfd_rom.ml_rom.rom_lagrangian_ml.data_loader import load_lagrangian_snapshots
from cpfd_rom.ml_rom.rom_lagrangian_ml.evaluation import create_pointnet_model, train_pointnet_model
from cpfd_rom.ml_rom.rom_lagrangian_ml.visualization_lagrangian import predict_and_plot_rmse, plot_selected_snapshots
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import save_model, load_model
from cpfd_rom.util.layers import ExpandDimsLayer, TileLayer
import joblib
import os
from cpfd_rom.util.output_utils import setup_output_dir
from cpfd_rom.util.model_utils import setup_model_paths
import tensorflow as tf


def run_lagrangian_ml_pipeline(config, log_time):

    physical_gpus = tf.config.list_physical_devices('GPU')
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[INFO] Using {len(physical_gpus)} GPUs with memory growth enabled.")

    model_path = getattr(config, 'model_path_lagrangian', 'cnn_autoencoder_lagrangian.keras')
    # Setup output directory once at start
    setup_output_dir(config)
    setup_model_paths(config)

    model_path = config.model_path_lagrangian
    model = None
    scaler = None

    epochs = getattr(config, 'epochs', 100)
    batch_size = getattr(config, 'batch_size', 2)

    if getattr(config, 'skip_training', False) and os.path.exists(model_path):
        with log_time("Loading pretrained model"):
            print(f"[INFO] Loading pretrained model from {model_path}")
            model = load_model(model_path, custom_objects={
                'ExpandDimsLayer': ExpandDimsLayer,
                'TileLayer': TileLayer
            })
        scaler_path = model_path.replace(".keras", "_scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        with log_time("Loading and preprocessing Lagrangian data"):
            print("[INFO] Loading Lagrangian training data...")
            train_times, train_data = load_lagrangian_snapshots(config.rev_dirs)

            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_data.reshape(-1, 4)).reshape(train_data.shape)

        with log_time("Training PointNet autoencoder"):
            print("[INFO] Training PointNet autoencoder...")
            n_points, n_features = train_scaled.shape[1], train_scaled.shape[2]
            model = create_pointnet_model(n_points, n_features, ExpandDimsLayer, TileLayer)
            model = train_pointnet_model(model, train_scaled, epochs=epochs, batch_size=batch_size)
            save_model(model, model_path)
            joblib.dump(scaler, model_path.replace(".keras", "_scaler.pkl"))

    with log_time("Loading test data and running inference"):
        print("[INFO] Loading test data...")
        test_directory = os.path.join(config.base_data_dir, config.test_dir)
        test_times, test_data = load_lagrangian_snapshots([test_directory])

        test_scaled = scaler.transform(test_data.reshape(-1, 4)).reshape(test_data.shape)

        print("[INFO] Running inference and evaluation...")
        pred_array = predict_and_plot_rmse(model, scaler, test_scaled, test_data, test_times)
        plot_selected_snapshots(test_times, test_data, pred_array)
