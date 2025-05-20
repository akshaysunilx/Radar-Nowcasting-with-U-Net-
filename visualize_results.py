# visualize_results_pytorch.py

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import random
import torch
import matplotlib.animation as animation
from sklearn.metrics import mean_squared_error, r2_score

REFLECTIVITY_MAX = 70.0
TARGET_SHAPE     = (128, 128)
TIME_HORIZON     = 12

def create_visualizations(model,
                          X_test, Y_test, input_times,
                          output_prefix="extended",
                          generate_static=True,
                          generate_gif=True,
                          generate_correlation=True):
    """
    Visualize a random sample from the test set.

    Parameters
    ----------
    model : PyTorch model
    X_test, Y_test : torch.Tensor with shapes
                     [N, C_in, H, W]  and  [N, T_out, H, W]
    input_times    : list of timestamps for X_test
    """
    device = next(model.parameters()).device

    # ---------- choose random sample ----------------------------------------
    idx = random.randint(0, len(X_test) - 1)
    model.eval()
    with torch.no_grad():
        input_tensor = X_test[idx:idx+1].to(device)
        pred_tensor = model(input_tensor).cpu().numpy()[0]  # [T, H, W]
    true_tensor = Y_test[idx].cpu().numpy()  # [T, H, W]

    # Transpose to [H, W, T] for visualization
    pred = np.transpose(pred_tensor, (1, 2, 0))
    true = np.transpose(true_tensor, (1, 2, 0))

    # Denormalize
    pred = pred * (REFLECTIVITY_MAX / 3) + (REFLECTIVITY_MAX / 2)
    true = true * (REFLECTIVITY_MAX / 3) + (REFLECTIVITY_MAX / 2)
    input_img = X_test[idx, 0].cpu().numpy() * (REFLECTIVITY_MAX / 3) + (REFLECTIVITY_MAX / 2)

    vmin, vmax = 0, REFLECTIVITY_MAX

    # ---------- static 4‑panel figure --------------------------------------
    if generate_static:
        fig, ax = plt.subplots(1, 4, figsize=(22, 6))
        ax[0].imshow(input_img, cmap='Reds', vmin=vmin, vmax=vmax)
        ax[1].imshow(true[..., -1], cmap='Reds', vmin=vmin, vmax=vmax)
        ax[2].imshow(pred[..., -1], cmap='Reds', vmin=vmin, vmax=vmax)
        ax[3].imshow(pred[..., -1] - true[..., -1], cmap='seismic', vmin=-20, vmax=20)

        ttl0 = f"Input Reflectivity\n{input_times[idx]}"
        ax[0].set_title(ttl0)
        ax[1].set_title("Ground Truth T+10")
        ax[2].set_title("Predicted T+10")
        ax[3].set_title("Prediction Error")
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(f"comparison_{output_prefix}.png")
        plt.show()

    # ---------- animated GIFs ---------------------------------------------
    if generate_gif:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ims = []
        for t in range(TIME_HORIZON):
            ims.append([
                axes[0].imshow(true[..., t], cmap='Reds', vmin=vmin, vmax=vmax, animated=True),
                axes[1].imshow(pred[..., t], cmap='Reds', vmin=vmin, vmax=vmax, animated=True)
            ])
        for a in axes:
            a.axis('off')
        ani = animation.ArtistAnimation(fig, ims, interval=400, blit=True)
        ani.save(f"gt_vs_pred_{output_prefix}.gif", writer='pillow')

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ims_err = []
        for t in range(TIME_HORIZON):
            ims_err.append([
                ax2.imshow(pred[..., t] - true[..., t], cmap='seismic', vmin=-20, vmax=20, animated=True)
            ])
        ax2.axis('off')
        ani_err = animation.ArtistAnimation(fig2, ims_err, interval=400, blit=True)
        ani_err.save(f"error_maps_{output_prefix}.gif", writer='pillow')

    # ---------- write NetCDF ----------------------------------------------
    ds = xr.Dataset(
        {
            "predicted": (("time", "x", "y"), np.transpose(pred, (2, 0, 1))),
            "actual":    (("time", "x", "y"), np.transpose(true, (2, 0, 1)))
        },
        coords={
            "time": [f"T+{i+1}" for i in range(TIME_HORIZON)],
            "x": np.arange(TARGET_SHAPE[0]),
            "y": np.arange(TARGET_SHAPE[1]),
        }
    )
    ds.to_netcdf(f"output_prediction_{output_prefix}.nc")

    # ---------- scatter / correlation -------------------------------------
    if generate_correlation:
        t_flat = true[..., -1].ravel()
        p_flat = pred[..., -1].ravel()
        msk = (t_flat > 0) & (p_flat > 0)
        t_flat, p_flat = t_flat[msk], p_flat[msk]
        r2 = r2_score(t_flat, p_flat)
        rmse = np.sqrt(mean_squared_error(t_flat, p_flat))

        plt.figure(figsize=(6, 6))
        plt.hexbin(t_flat, p_flat, gridsize=50, cmap='viridis', bins='log')
        plt.plot([0, 70], [0, 70], 'r--')
        plt.xlabel("True Reflectivity (dBZ)")
        plt.ylabel("Predicted Reflectivity (dBZ)")
        plt.title(f"Scatter (R²={r2:.2f}, RMSE={rmse:.2f})")
        plt.colorbar(label='log10(N)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"correlation_plot_T+10_{output_prefix}.png")
        plt.show()
