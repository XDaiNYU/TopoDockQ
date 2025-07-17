# TopoDockQ
![Figure](./image/combine_all.jpg)
Co-authored-by: Dr. Rui Wang <rw3594@nyu.edu> <https://github.com/wangru25>

# Data
The protein-peptide interface PDB files could be downloaded from the figshare repository.
The feature extraction algorithm used for generating TopoDockQ features is available at: https://github.com/wangru25/TopoDockQ-Feature.

# Training
The model training example is described in the 01_tutorial_train.ipynb.
Necessary feature CSV files could be downloaded from the Zenodo repository: 10.5281/zenodo.15469415
processed_data.zip in the Zenodo: 
    singlePPD_full_bins_features.csv: generated TopoDockQ features used for the model training, validation and testing.
    singlePPD_DockQ.csv, singlePPD_filtered_DockQ.csv: calculated DockQ scores used for model training, validation and testing.

# Inference
The inference example is described in the 02_tutorial_inference.ipynb.
Necessary feature CSV files and optimal model could be downloaded from the Zenodo repository: 10.5281/zenodo.15469415
        
        
        
        . 
You can either use the model saved by example training script, or the best model(best_model.pth) provided in the trained_model.zip from Zenodo.
