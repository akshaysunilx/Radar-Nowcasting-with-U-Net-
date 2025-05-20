# -*- coding: utf-8 -*-
"""
Created on Wed May  7 19:27:34 2025

@author: admin
"""


# main.py

# -*- coding: utf-8 -*-


import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from data_loader           import load_full_dataset
from model_architecture    import create_spatiotemporal_unet, weighted_mse
from visualize_results     import create_visualizations

# ───────── CONFIG ───────────────────────────────────────────────────────────────
JUNE_PATH   = "D:/Processed_Radar_Data/2023/June"
JULY_PATH   = "D:/Processed_Radar_Data/2023/July"
AUGUST_PATH = "D:/Processed_Radar_Data/2023/August"
FOLDER_PATHS = {"June": JUNE_PATH, "July": JULY_PATH, "August": AUGUST_PATH}

TARGET_SHAPE     = (128, 128)
TIME_HORIZON     = 12        # 12 future frames  (≈ 1 h 31 min if scans ~7.6 min apart)
SEQUENCE_LENGTH  = 1
REFLECTIVITY_MAX = 70.0
VELOCITY_SCALE   = 60.0

LEARNING_RATE = 4.3e-4
BATCH_SIZE    = 8
EPOCHS        = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ───────── MAIN ────────────────────────────────────────────────────────────────
def main():
    print("=== PyTorch Radar Forecasting Pipeline ===")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"Using device    : {device}\n")

    # 1) Load full dataset
    X, Y, input_times, future_times = load_full_dataset(
        folder_paths     = FOLDER_PATHS,
        target_shape     = TARGET_SHAPE,
        time_horizon     = TIME_HORIZON,
        sequence_length  = SEQUENCE_LENGTH,
        reflectivity_max = REFLECTIVITY_MAX,
        velocity_scale   = VELOCITY_SCALE,
    )

    # 2) Simple time split (unchanged logic)
    train_mask = (input_times >= "2023-06-01") & (input_times <= "2023-08-20")
    val_mask   = (input_times >  "2023-08-20") & (input_times <= "2023-08-23")
    test_mask  = (input_times >  "2023-08-23") & (input_times <= "2023-08-25")

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_val,   Y_val   = X[val_mask],   Y[val_mask]
    X_test,  Y_test  = X[test_mask],  Y[test_mask]
    input_times_test = input_times[test_mask]

    print(f"Samples → Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    # 3) NHWC → NCHW tensors
    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2)

    X_train_t, Y_train_t = to_tensor(X_train), to_tensor(Y_train)
    X_val_t,   Y_val_t   = to_tensor(X_val),   to_tensor(Y_val)
    X_test_t,  Y_test_t  = to_tensor(X_test),  to_tensor(Y_test)

    # 4) DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   Y_val_t),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TensorDataset(X_test_t,  Y_test_t),  batch_size=BATCH_SIZE)

    # 5) Model / optimiser / loss
    model     = create_spatiotemporal_unet(X_train_t.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = weighted_mse           # expects (y_true, y_pred)

    best_val_loss = float("inf")
    train_losses, val_losses, train_mae, val_mae = [], [], [], []

    # 6) Training loop (critical fixes marked ★)
    print("Training …")
    for epoch in range(EPOCHS):
        model.train()
        tr_loss_sum, tr_abs_sum, tr_elem = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            y_hat = model(xb)

            # ★ FIX 1: loss uses (y_true, y_pred)
            loss = criterion(yb, y_hat)
            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item()
            # ★ FIX 2: accumulate absolute error correctly
            tr_abs_sum  += torch.abs(y_hat - yb).sum().item()
            tr_elem     += yb.numel()

        model.eval()
        va_loss_sum, va_abs_sum, va_elem = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_hat  = model(xb)
                va_loss_sum += criterion(yb, y_hat).item()    # ★ same arg order
                va_abs_sum  += torch.abs(y_hat - yb).sum().item()
                va_elem     += yb.numel()

        # Averaged metrics
        tr_loss = tr_loss_sum / len(train_loader)
        va_loss = va_loss_sum / len(val_loader)
        tr_mae  = tr_abs_sum / tr_elem
        va_mae  = va_abs_sum / va_elem

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_mae.append(tr_mae)
        val_mae.append(va_mae)

        print(f"Epoch {epoch+1:03d}: "
              f"Train Loss={tr_loss:.4f}  Val Loss={va_loss:.4f}  "
              f"Train MAE={tr_mae:.4f}  Val MAE={va_mae:.4f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), "extended_training_model.pth")

    # 7) Plot history - Enhanced version with multiple visualizations
    plt.figure(figsize=(15, 10))
    
    # Loss subplot
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.title('MSE Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # MAE subplot
    plt.subplot(2, 2, 2)
    plt.plot(train_mae, 'b-', label='Training MAE')
    plt.plot(val_mae, 'r-', label='Validation MAE')
    plt.title('Mean Absolute Error vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # Log-scale loss subplot
    plt.subplot(2, 2, 3)
    plt.semilogy(train_losses, 'b-', label='Training Loss')
    plt.semilogy(val_losses, 'r-', label='Validation Loss')
    plt.title('Log-scale MSE Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # Loss difference subplot
    plt.subplot(2, 2, 4)
    loss_diff = np.array(train_losses) - np.array(val_losses)
    plt.plot(loss_diff, 'g-')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Train-Val Loss Difference\n(Positive = Overfitting)')
    plt.xlabel('Epochs')
    plt.ylabel('Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("training_history_enhanced.png", dpi=150)
    plt.close()
    
    # Also save the original plot format for comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
    plt.title("MSE Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(train_mae, label="Train"); plt.plot(val_mae, label="Val")
    plt.title("Mean Absolute Error"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig("training_history_original.png", dpi=150); plt.close()

    # 8) Visualisations
    model.load_state_dict(torch.load("extended_training_model.pth"))
    print("\nGenerating test visualisations …")
    create_visualizations(model, X_test_t, Y_test_t, input_times_test, device)

    # 9) Final evaluation
    def evaluate(loader):
        model.eval()
        loss_sum, abs_sum, elem = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                y_hat  = model(xb)
                loss_sum += criterion(yb, y_hat).item()       # ★ fixed order
                abs_sum  += torch.abs(y_hat - yb).sum().item()
                elem     += yb.numel()
        return loss_sum / len(loader), abs_sum / elem

    print("\nFinal Evaluation:")
    for tag, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        mse, mae = evaluate(loader)
        print(f"{tag:5s}  MSE={mse:.4f}  MAE={mae:.4f}")

if __name__ == "__main__":
    main()