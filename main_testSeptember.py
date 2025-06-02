# -*- coding: utf-8 -*-
"""
Created on Sat May 31 23:04:30 2025
@author: admin
"""

"""
Spatiotemporal U-Net training script.
* Val set   : Sep 1–15, 2024
* Test set  : Sep 16–30, 2024
* Train set : all data earlier than Sep 2024, excluding April
Updated: 2025-06-02
"""

# ───────── IMPORTS ─────────────────────────────────────────────────────────────
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from data_loader import load_full_dataset
from model_architecture import create_spatiotemporal_unet, weighted_mse
from visualize_results import create_visualizations

# ───────── CONFIG ──────────────────────────────────────────────────────────────
FOLDER_PATHS = {
    # "April":     "D:/Processed_Radar_Data/2024/April",  # ← Skipped due to memory usage
    "May":       "D:/Processed_Radar_Data/2023/May",
    "June":      "D:/Processed_Radar_Data/2023/June",
    "July":      "D:/Processed_Radar_Data/2023/July",
    "August":    "D:/Processed_Radar_Data/2023/August",
    "September": "D:/Processed_Radar_Data/2024/September",
}

TARGET_SHAPE     = (128, 128)
TIME_HORIZON     = 12
SEQUENCE_LENGTH  = 1
REFLECTIVITY_MAX = 70.0
VELOCITY_SCALE   = 60.0

LEARNING_RATE = 4.3e-4
BATCH_SIZE    = 8
EPOCHS        = 500
PATIENCE      = 80

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ───────── MAIN ────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== PyTorch Radar Forecasting Pipeline ===")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"Using device    : {device}\n")
    print("Folders to load:", list(FOLDER_PATHS.keys()))

    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1) ───── Load data
    X, Y, input_times, _ = load_full_dataset(
        folder_paths     = FOLDER_PATHS,
        target_shape     = TARGET_SHAPE,
        time_horizon     = TIME_HORIZON,
        sequence_length  = SEQUENCE_LENGTH,
        reflectivity_max = REFLECTIVITY_MAX,
        velocity_scale   = VELOCITY_SCALE,
    )

    # 2) ───── Sort chronologically
    order = np.argsort(input_times)
    X, Y, input_times = X[order], Y[order], input_times[order]

    # 3) ───── Build masks
    input_dt = input_times.to_numpy().astype("datetime64[D]")

    sep_start = np.datetime64("2024-09-01")
    sep_mid   = np.datetime64("2024-09-15")
    sep_end   = np.datetime64("2024-09-30")

    val_mask   = (input_dt >= sep_start) & (input_dt <= sep_mid)
    test_mask  = (input_dt >  sep_mid)  & (input_dt <= sep_end)
    train_mask =  input_dt <  sep_start

    assert not np.any(val_mask & train_mask)
    assert not np.any(test_mask & train_mask)
    assert not np.any(val_mask & test_mask)

    # 4) ───── Split sets
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_val,   Y_val   = X[val_mask],   Y[val_mask]
    X_test,  Y_test  = X[test_mask],  Y[test_mask]
    input_times_test = input_times[test_mask]

    # 5) ───── Diagnostics
    print("Samples → "
          f"Train: {len(X_train)}  "
          f"Val: {len(X_val)}  "
          f"Test: {len(X_test)}\n")
    print(f"Training window  : {input_times[train_mask][0]} → {input_times[train_mask][-1]}")
    print(f"Validation window: {input_times[val_mask][0]}  → {input_times[val_mask][-1]}")
    print(f"Test window      : {input_times[test_mask][0]} → {input_times[test_mask][-1]}\n")

    # 6) ───── Convert to NCHW tensors
    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2)

    X_train_t, Y_train_t = to_tensor(X_train), to_tensor(Y_train)
    X_val_t,   Y_val_t   = to_tensor(X_val),   to_tensor(Y_val)
    X_test_t,  Y_test_t  = to_tensor(X_test),  to_tensor(Y_test)

    # 7) ───── DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   Y_val_t),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t,  Y_test_t),  batch_size=BATCH_SIZE, shuffle=False)

    # 8) ───── Model and training setup
    model     = create_spatiotemporal_unet(X_train_t.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=PATIENCE, factor=0.5)
    criterion = weighted_mse

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses, train_mae, val_mae = [], [], [], []

    print("Training …")
    total_t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.time()
        model.train()
        tr_loss_sum = tr_abs_sum = tr_elems = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            y_hat = model(xb)
            loss  = criterion(yb, y_hat)
            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item() * yb.numel()
            tr_abs_sum  += torch.abs(y_hat - yb).sum().item()
            tr_elems    += yb.numel()

        model.eval()
        va_loss_sum = va_abs_sum = va_elems = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_hat  = model(xb)
                va_loss_sum += criterion(yb, y_hat).item() * yb.numel()
                va_abs_sum  += torch.abs(y_hat - yb).sum().item()
                va_elems    += yb.numel()

        tr_loss = tr_loss_sum / tr_elems
        va_loss = va_loss_sum / va_elems
        tr_mae  = tr_abs_sum  / tr_elems
        va_mae  = va_abs_sum  / va_elems

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_mae.append(tr_mae)
        val_mae.append(va_mae)
        scheduler.step(va_loss)

        print(f"Epoch {epoch:03d} | "
              f"Train Loss {tr_loss:.4f}  Val Loss {va_loss:.4f}  "
              f"Train MAE {tr_mae:.4f}  Val MAE {va_mae:.4f}  "
              f"Time {time.time() - epoch_t0:.1f}s")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    total_minutes = (time.time() - total_t0) / 60
    print(f"\nTotal training time: {int(total_minutes)}m {int((total_minutes % 1) * 60)}s")

    # 10) ───── Plot metrics
    def plot_metrics() -> None:
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
        plt.title("MSE Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.grid(True, linestyle="--"); plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(train_mae, label="Train"); plt.plot(val_mae, label="Val")
        plt.title("MAE"); plt.xlabel("Epoch"); plt.ylabel("Error")
        plt.grid(True, linestyle="--"); plt.legend()

        plt.subplot(2, 2, 3)
        plt.semilogy(train_losses, label="Train"); plt.semilogy(val_losses, label="Val")
        plt.title("Log-scale Loss"); plt.grid(True, linestyle="--"); plt.legend()

        plt.subplot(2, 2, 4)
        diff = np.array(train_losses) - np.array(val_losses)
        plt.plot(diff); plt.axhline(0, color="r", alpha=0.4)
        plt.title("Train – Val Loss Δ"); plt.grid(True, linestyle="--")

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=150)
        plt.close()

    plot_metrics()

    # 11) ───── Visualizations
    print("\nGenerating test visualisations …")
    model.load_state_dict(torch.load("best_model.pth"))
    create_visualizations(model, X_test_t, Y_test_t, input_times_test, device)

    # 12) ───── Final evaluation
    def evaluate(loader: DataLoader):
        model.eval()
        mse_sum = abs_sum = elems = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                y_hat  = model(xb)
                mse_sum += criterion(yb, y_hat).item() * yb.numel()
                abs_sum += torch.abs(y_hat - yb).sum().item()
                elems   += yb.numel()
        return mse_sum / elems, abs_sum / elems

    print("\nFinal Evaluation:")
    for tag, loader in [("Train", train_loader),
                        ("Val",   val_loader),
                        ("Test",  test_loader)]:
        mse, mae = evaluate(loader)
        print(f"{tag:5s}  MSE={mse:.4f}  MAE={mae:.4f}")

# ───────── RUN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
