# main.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from data_loader import load_full_dataset, print_ram_usage
from model_architecture import create_spatiotemporal_unet, weighted_mse
from visualize_results import create_visualizations

# Constants
REFLECTIVITY_MAX = 70.0
VELOCITY_SCALE = 60.0
TARGET_SHAPE = (128, 128)
TIME_HORIZON = 10
SEQUENCE_LENGTH = 1
NUM_FILTERS = 64
LEARNING_RATE = 0.00043

# Paths
DATA_PATHS = {
    "June": "D:/Processed_Radar_Data/2023/June",
    "July": "D:/Processed_Radar_Data/2023/July",
    "August": "D:/Processed_Radar_Data/2023/August"
}


def main():
    print("==================== DATA LOADING ====================")
    print("Loading and processing June, July, and August radar data...")
    X, Y, input_times, future_times = load_full_dataset(
        folder_paths=DATA_PATHS,
        target_shape=TARGET_SHAPE,
        time_horizon=TIME_HORIZON,
        sequence_length=SEQUENCE_LENGTH,
        reflectivity_max=REFLECTIVITY_MAX,
        velocity_scale=VELOCITY_SCALE
    )

    # Splitting
    train_mask = ((input_times >= "2023-06-01") & (input_times <= "2023-08-20"))
    val_mask = ((input_times > "2023-08-20") & (input_times <= "2023-08-23"))
    test_mask = ((input_times > "2023-08-23") & (input_times <= "2023-08-25"))

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]
    input_test_times = input_times[test_mask]

    print(f"[✔] Training samples: {len(X_train)}")
    print(f"[✔] Validation samples: {len(X_val)}")
    print(f"[✔] Test samples: {len(X_test)}")

    print("==================== MODEL SETUP ====================")
    model = create_spatiotemporal_unet(input_shape=X_train.shape[1:])
    model.compile(optimizer=Adam(LEARNING_RATE), loss=weighted_mse, metrics=['mae'])

    print("==================== TRAINING ====================")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=100,
        batch_size=8,
        callbacks=[
            EarlyStopping(patience=40, restore_best_weights=True),
            ModelCheckpoint("extended_training_model.keras", save_best_only=True)
        ]
    )

    print("==================== TRAINING HISTORY PLOT ====================")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('MAE Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

    print("==================== VISUALIZATION ====================")
    create_visualizations(model, X_test, Y_test, input_test_times)

    print("==================== FINAL EVALUATION ====================")
    train_score = model.evaluate(X_train, Y_train, verbose=0)
    val_score = model.evaluate(X_val, Y_val, verbose=0)
    test_score = model.evaluate(X_test, Y_test, verbose=0)

    print(f"Train  -> MSE: {train_score[0]:.4f}, MAE: {train_score[1]:.4f}")
    print(f"Val    -> MSE: {val_score[0]:.4f}, MAE: {val_score[1]:.4f}")
    print(f"Test   -> MSE: {test_score[0]:.4f}, MAE: {test_score[1]:.4f}")


if __name__ == "__main__":
    main()
