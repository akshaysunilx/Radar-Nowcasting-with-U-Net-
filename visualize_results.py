# visualize_results.py

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import random
import tensorflow as tf
import matplotlib.animation as animation
from sklearn.metrics import mean_squared_error, r2_score

REFLECTIVITY_MAX = 70.0
TARGET_SHAPE = (128, 128)
TIME_HORIZON = 10


def create_visualizations(model, X_test, Y_test, input_times, output_prefix="extended", generate_static=True, generate_gif=True, generate_correlation=True):
    sample_idx = random.randint(0, len(X_test) - 1)
    enhanced_pred = model.predict(X_test[sample_idx:sample_idx+1])[0]
    true_data = Y_test[sample_idx]

    enhanced_pred = enhanced_pred * (REFLECTIVITY_MAX/3) + (REFLECTIVITY_MAX/2)
    true_data = true_data * (REFLECTIVITY_MAX/3) + (REFLECTIVITY_MAX/2)
    input_plot = X_test[sample_idx][..., 0] * (REFLECTIVITY_MAX/3) + (REFLECTIVITY_MAX/2)

    vmin, vmax = 0, 70

    if generate_static:
        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        axes[0].imshow(input_plot, cmap='Reds', vmin=vmin, vmax=vmax)
        axes[1].imshow(true_data[..., -1], cmap='Reds', vmin=vmin, vmax=vmax)
        axes[2].imshow(enhanced_pred[..., -1], cmap='Reds', vmin=vmin, vmax=vmax)
        axes[3].imshow(enhanced_pred[..., -1] - true_data[..., -1], cmap='seismic', vmin=-20, vmax=20)

        axes[0].set_title(f"Input Reflectivity\n{input_times[sample_idx]}")
        axes[1].set_title("Ground Truth T+10")
        axes[2].set_title("Predicted T+10")
        axes[3].set_title("Prediction Error")
        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"comparison_{output_prefix}.png")
        plt.show()

    if generate_gif:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ims = []
        for t in range(TIME_HORIZON):
            ims.append([
                axes[0].imshow(true_data[..., t], cmap='Reds', vmin=vmin, vmax=vmax, animated=True),
                axes[1].imshow(enhanced_pred[..., t], cmap='Reds', vmin=vmin, vmax=vmax, animated=True)
            ])
        for ax in axes:
            ax.axis('off')

        ani = animation.ArtistAnimation(fig, ims, interval=400, blit=True)
        ani.save(f"gt_vs_pred_{output_prefix}.gif", writer='pillow')

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ims_error = []
        for t in range(TIME_HORIZON):
            ims_error.append([
                ax2.imshow(enhanced_pred[..., t] - true_data[..., t], cmap='seismic', vmin=-20, vmax=20, animated=True)
            ])
        ax2.axis('off')
        ani_err = animation.ArtistAnimation(fig2, ims_error, interval=400, blit=True)
        ani_err.save(f"error_maps_{output_prefix}.gif", writer='pillow')

    ds = xr.Dataset({
        "predicted": ("time", enhanced_pred.transpose(2, 0, 1)),
        "actual": ("time", true_data.transpose(2, 0, 1))
    }, coords={
        "time": [f"T+{i+1}" for i in range(TIME_HORIZON)],
        "x": np.arange(TARGET_SHAPE[0]),
        "y": np.arange(TARGET_SHAPE[1])
    })
    ds.to_netcdf(f"output_prediction_{output_prefix}.nc")

    if generate_correlation:
        true_flat = true_data[..., -1].flatten()
        pred_flat = enhanced_pred[..., -1].flatten()

        mask = (true_flat > 0) & (pred_flat > 0)
        true_flat, pred_flat = true_flat[mask], pred_flat[mask]

        r2 = r2_score(true_flat, pred_flat)
        rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))

        plt.figure(figsize=(6, 6))
        plt.hexbin(true_flat, pred_flat, gridsize=50, cmap="viridis", bins='log')
        plt.plot([0, 70], [0, 70], 'r--')
        plt.xlabel("True Reflectivity (dBZ)")
        plt.ylabel("Predicted Reflectivity (dBZ)")
        plt.title(f"Scatter Plot (R²: {r2:.2f}, RMSE: {rmse:.2f})")
        plt.colorbar(label='log10(N)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"correlation_plot_T+10_{output_prefix}.png")
        plt.show()