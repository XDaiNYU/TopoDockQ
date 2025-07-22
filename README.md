# TopoDockQ

![Figure](./image/combine_all.jpg)

**Co-first authored by:** Dr. Rui Wang <rw3594@nyu.edu>

**Corresponding Author:** Dr. Yingkai Zhang <yz22@nyu.edu>

## Overview

TopoDockQ is a deep learning model that predicts protein-peptide binding quality using persistent combinatorial Laplacian-based features. The model leverages topological data analysis to extract meaningful features from protein-peptide interfaces and uses a neural network architecture to predict DockQ scores.

## Data

### Protein-Peptide Interface Files
The protein-peptide interface PDB files can be downloaded from the [figshare repository](https://zenodo.org/record/15469415).

### Feature Extraction
The feature extraction algorithm used for generating TopoDockQ features is also available at: [TopoDockQ-Feature](https://github.com/wangru25/TopoDockQ-Feature)

### Training and Inference Data
Download the necessary feature and model files from the [Zenodo repository](https://zenodo.org/record/15469415):

- `processed_data.zip` contains:
  - `singlePPD_full_bins_features.csv`: Generated TopoDockQ features for model training, validation, and testing
  - `singlePPD_DockQ.csv`, `singlePPD_filtered_DockQ.csv`: Calculated DockQ scores for model training, validation, and testing

- `trained_model.zip` contains:
  - `best_model.pth`: Optimal pre-trained model for inference

## Features

- **Persistent Combinatorial Laplacian Features**: Advanced topological features extracted from protein-peptide interfaces
- **Deep Neural Network**: Multi-layer perceptron with batch normalization and dropout for robust predictions
- **Comprehensive Training Pipeline**: Complete workflow from feature extraction to model training and inference
- **Easy-to-use Tutorials**: Jupyter notebooks for training and inference examples

## Requirements

- python=3.8.1
- numpy<=1.24.3
- pandas<=2.2.0
- scikit-learn=1.3.0
- gudhi=3.8.0
- pytorch>=2.0.0

**Note:** You can also use the existing `mlms2023` conda environment which already has PyTorch 2.0.0 and other required packages installed.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/XDaiNYU/TopoDockQ.git
cd TopoDockQ
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate TopoDockQ
```

## Usage

### Feature Extraction

To extract features from protein-peptide interface files:

```bash
python -m main --pdb_id PDBID --model_id MODELID --bins BINS --filtration FILTRATION --file_path FILEPATH --saving_path SAVINGPATH
```

**Parameters:**

- PDBID: A string, the Protein Data Bank ID. For example, 4k38
- MODELID: An int, the ID of the model. For example, 44.
- BINS: A string records bins' starting and ending points in a list format.
- FILTRATION: A string, records various distance-based filtration values. 
- FILEPATH: A string shows the directory of the protein-peptide data.
- SAVINGPATH: A string, indicates where the output feature file will be saved. 

**Example:**
```bash
python -m main --pdb_id 4k38 --model_id 44 --bins "[0, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]" --filtration "[0, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]" --file_path ./data/interface_files --saving_path ./feature
```

**Output:** Features will be saved as `feature_4k38_ranked_44_sp_interface.npy` in the specified saving path.

### Model Training

The model training example is described in `01_tutorial_train.ipynb`. The notebook demonstrates:

- Data preprocessing and feature engineering
- Model architecture configuration
- Training hyperparameters
- Model evaluation and validation

**Key Hyperparameters:**
- Input dimension: 2646
- Hidden layers: 2048 neurons each (4 layers)
- Learning rate: 0.0005
- Batch size: 512
- Dropout: 0.0

### Model Inference

The inference example is described in `02_tutorial_inference.ipynb`. The notebook shows:

- Loading pre-trained models
- Feature preprocessing for inference
- Making predictions on test data
- Model performance evaluation

You can use either:
- The model saved by the training script
- The optimal model (`best_model.pth`) provided in the [Zenodo repository](https://zenodo.org/record/15469415)

## Model Architecture

TopoDockQ uses a deep neural network with the following architecture:

- **Input Layer**: 2646 persistent combinatorial Laplacians features
- **Hidden Layers**: 4 fully connected layers with 2048 neurons each
- **Activation**: ReLU
- **Regularization**: Batch normalization and dropout
- **Output Layer**: Single neuron for DockQ score prediction
- **Loss Function**: Root Mean Squared Error (RMSE)

## Project Structure

```
TopoDockQ/
├── data/
│   └── interface_files/          # Protein-peptide interface PDB files
├── feature/                      # Extracted feature files
├── image/                        # Project images and figures
├── results/                      # Model results and outputs
├── src/                          # Source code
│   ├── model.py                  # Neural network model definition
│   ├── train.py                  # Training utilities
│   └── ...
├── 01_tutorial_train.ipynb       # Training tutorial
├── 02_tutorial_inference.ipynb   # Inference tutorial
├── environment.yaml              # Conda environment specification
└── README.md                     # This file
```

## Citation

If you use TopoDockQ in your research, please cite:

```bibtex
@article{dai2025topological,
  title={Topological Deep Learning for Enhancing Peptide-Protein Complex Prediction},
  author={Dai, Xuhang and Wang, Rui and Zhang, Yingkai},
  journal={in review},
  year={2025}
}
```

## Other Helpful References for Persistent Combinatorial Laplacians:

- R. Wang, R. Zhao, E. Ribando-Gros, J. Chen, Y. Tong, and G.-W. Wei. [HERMES: Persistent spectral graph software](https://www.aimsciences.org/article/doi/10.3934/fods.2021006), _Foundations of Data Science_, 2021.
- R. Wang, D. D. Nguyen, and G.-W. Wei. [Persistent spectral graph](https://users.math.msu.edu/users/weig/paper/p243.pdf), _International Journal for Numerical Methods in Biomedical Engineering_, page e3376, 2020.


## Contact

For feature generation questions and support, please contact:
- Dr. Rui Wang: <rw3594@nyu.edu>

For model training questions and support, please contact:
- Mr. Xuhang Dai: <xd638@nyu.edu>
- Dr. Rui Wang: <rw3594@nyu.edu>
