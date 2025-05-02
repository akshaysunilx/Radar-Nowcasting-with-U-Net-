Radar Nowcasting using U-Net (Extended Training Version)

This repository contains a deep learning pipeline for radar-based nowcasting using a U-Net architecture. The system is designed to predict reflectivity fields up to 10 time steps ahead using spatiotemporal features derived from multiple elevation scans of Doppler radar data.

🧭 Overview

Model: U-Net with skip connections and convolutional encoder-decoder blocks

Input Features:

Reflectivity

Velocity

Magnitude of velocity

Divergence

Direction

Vorticity

Input Resolution: 128x128 grid, interpolated from raw radar data

Prediction: 10 future frames of maximum reflectivity field

📁 Project Structure

.
├── main.py                    # Central training and evaluation pipeline
├── data_loader.py            # Loads and processes radar data month-wise with RAM tracking
├── model_architecture.py     # U-Net creation and custom loss function
├── visualize_results.py      # Generates comparative plots and prediction GIFs
├── training_history.png      # Saved training loss/MAE curves
├── output_prediction_extended.nc  # Predicted and actual radar fields
├── extended_training_model.keras # Best saved Keras model

📊 Input Data

Place processed .nc radar files in these folders:

D:/Processed_Radar_Data/2023/June/
D:/Processed_Radar_Data/2023/July/
D:/Processed_Radar_Data/2023/August/

Each file should follow the naming convention 2023MONDD_combined.nc and contain variables:

Reflectivity_Horizontal

Horizontal_Velocity

time

To add more months (e.g. April), update the DATA_PATHS dictionary in main.py accordingly.

🚀 Running the Pipeline

Ensure the data folders are correctly populated.

Execute the script:

python main.py

🧠 Model Training

Loss: Custom Weighted MSE (gives more weight to high-reflectivity regions)

Optimizer: Adam (learning rate = 0.00043)

Batch Size: 8

EarlyStopping: Patience of 40 epochs

📈 Outputs

training_history.png: Training and validation curves

comparison_extended_training.png: Input vs Predicted vs Ground Truth

gt_vs_pred_extended.gif: Animated prediction timeline

error_maps_extended.gif: Error progression across lead times

output_prediction_extended.nc: NetCDF output of predictions and actuals

📉 Evaluation Metrics

Final scores are printed at the end for:

Training set

Validation set

Test set

With both MSE and MAE values.

👨‍💻 Developed By

Akshay SunilCentre for Climate Studies, IIT BombayLast updated: May 2025

📌 Notes

RAM usage is tracked and printed for each processed day

Data is loaded in daily chunks to avoid memory overload

Input splitting for training/validation/test is done based on timestamp ranges

