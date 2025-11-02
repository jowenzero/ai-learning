# Laptop Price Prediction using Vanilla RNN

This project implements a Vanilla Recurrent Neural Network (RNN) using PyTorch to predict laptop prices based on various hardware specifications.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)

## 🎯 Overview

This project demonstrates the application of a vanilla RNN for regression tasks. While RNNs are typically used for sequential data, this implementation treats each laptop's features as a sequence of length 1, showcasing how RNN architecture can be adapted for tabular data prediction.

### Key Features
- Vanilla RNN implementation from scratch using PyTorch
- Custom dataset class for efficient data loading
- Comprehensive preprocessing pipeline
- Training with gradient clipping and learning rate scheduling
- Detailed evaluation metrics and visualizations

## 📊 Dataset

The dataset (`laptop_price.csv`) contains 1000 laptop samples with the following features:

### Features
- **Brand**: Laptop manufacturer (Asus, Acer, Lenovo, HP, Dell)
- **Processor_Speed**: CPU speed in GHz
- **RAM_Size**: RAM capacity in GB (4, 8, 16, 32)
- **Storage_Capacity**: Storage size in GB (256, 512, 1000)
- **Screen_Size**: Display size in inches
- **Weight**: Laptop weight in kg

### Target Variable
- **Price**: Laptop price in USD

### Data Split
- Training: 70% (700 samples)
- Validation: 15% (150 samples)
- Testing: 15% (150 samples)

## 🏗️ Model Architecture

### VanillaRNN Class

```python
VanillaRNN(
  input_size=6,      # Number of input features
  hidden_size=64,    # Number of hidden units
  num_layers=2,      # Number of RNN layers
  output_size=1,     # Price prediction
  dropout=0.2        # Dropout rate
)
```

### Architecture Components

1. **RNN Layers**
   - Type: Vanilla RNN (not LSTM or GRU)
   - Activation: Tanh
   - 2 stacked RNN layers with 64 hidden units each
   - Dropout between layers for regularization

2. **Output Layer**
   - Fully connected layer
   - Maps hidden state to price prediction
   - Additional dropout before final layer

### Model Parameters
- **Total Parameters**: ~12,000 trainable parameters
- **Activation Function**: Tanh (for RNN cells)
- **Output Activation**: None (linear regression)

## 🔧 Implementation Details

### Preprocessing Steps

1. **Label Encoding**
   - Brand names converted to numerical values (0-4)

2. **Feature Scaling**
   - StandardScaler applied to all features
   - Target variable (Price) also normalized
   - Scaling parameters saved for inverse transformation

3. **Data Format**
   - Input shape: `(batch_size, seq_length=1, num_features=6)`
   - Output shape: `(batch_size, 1)`

### Training Configuration

```python
# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 100
hidden_size = 64
num_layers = 2
dropout = 0.2
```

### Optimization Techniques

1. **Optimizer**: Adam
   - Adaptive learning rate
   - Beta values: (0.9, 0.999)

2. **Loss Function**: MSE (Mean Squared Error)
   - Suitable for regression tasks
   - Measures average squared difference

3. **Gradient Clipping**
   - Max norm: 1.0
   - Prevents exploding gradients (common in RNNs)

4. **Learning Rate Scheduler**
   - ReduceLROnPlateau
   - Factor: 0.5
   - Patience: 10 epochs
   - Reduces LR when validation loss plateaus

5. **Early Stopping Strategy**
   - Saves best model based on validation loss
   - Prevents overfitting

## 💻 Installation

### Requirements
```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm
```

### Detailed Dependencies
- Python >= 3.7
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Pandas >= 1.2.0
- Scikit-learn >= 0.24.0
- Matplotlib >= 3.3.0
- tqdm >= 4.60.0

## 🚀 Usage

### Running the Notebook

1. **Ensure data file is present**
   ```bash
   # laptop_price.csv should be in the same directory
   ```

2. **Open Jupyter Notebook**
   ```bash
   jupyter notebook main.ipynb
   ```

3. **Run all cells sequentially**
   - The notebook is designed to run from top to bottom
   - Each section builds on the previous one

### Training Output

The training process will display:
- Epoch progress with train/validation losses
- Learning rate adjustments
- Best model checkpoints

Example output:
```
Epoch [10/100], Train Loss: 0.1234, Val Loss: 0.1456
Epoch [20/100], Train Loss: 0.0987, Val Loss: 0.1123
...
Best validation loss: 0.0856
```

### Model Checkpoint

The best model is saved as:
- **File**: `best_rnn_model.pth`
- **Location**: Same directory as notebook
- **Contents**: Model state dictionary

## 📈 Results

### Performance Metrics

The model is evaluated using:

1. **Mean Absolute Error (MAE)**
   - Average absolute difference between predicted and actual prices
   - Interpretation: Average prediction error in dollars

2. **Root Mean Squared Error (RMSE)**
   - Square root of average squared differences
   - More sensitive to large errors than MAE

3. **R² Score**
   - Coefficient of determination
   - Range: 0 to 1 (1 = perfect predictions)
   - Measures proportion of variance explained

### Visualizations

The notebook generates several plots:

1. **Training History**
   - Train vs Validation loss over epochs
   - Helps identify overfitting/underfitting

2. **Actual vs Predicted**
   - Scatter plot with diagonal reference line
   - Perfect predictions would lie on the line

3. **Residual Plot**
   - Shows prediction errors vs predicted values
   - Helps identify systematic biases

4. **Sample Predictions Table**
   - 10 random samples with actual, predicted, and difference

## 📁 Project Structure

```
laptop-price-prediction/
│
├── laptop_price.csv          # Dataset
├── main.ipynb                # Main implementation notebook
├── README.md                 # This file
└── best_rnn_model.pth       # Saved model (generated after training)
```

## 🔍 Key Concepts Explained

### Why RNN for Tabular Data?

While RNNs are typically used for sequential data (time series, text), this project demonstrates:
- **Flexibility**: RNNs can handle various input formats
- **Feature Relationships**: RNN can capture complex feature interactions
- **Educational Purpose**: Understanding RNN mechanics in a simpler context

### RNN vs Other Architectures

**Advantages of this RNN approach:**
- Handles variable-length sequences (though not utilized here)
- Captures temporal dependencies (if features were sequential)
- Non-linear transformations through tanh activation

**Limitations:**
- Vanishing gradient problem (addressed by gradient clipping)
- More complex than simple feedforward networks for this task
- Slower training than CNNs or simple dense networks

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

- **Purpose**: Prevents exploding gradients in RNNs
- **Method**: Scales gradients if norm exceeds threshold
- **Effect**: More stable training, prevents NaN losses

## 🎓 Learning Outcomes

This project teaches:
1. Implementing vanilla RNN from scratch in PyTorch
2. Custom dataset creation for PyTorch
3. Proper data preprocessing for neural networks
4. Training loop implementation with validation
5. Model evaluation and visualization techniques
6. Gradient clipping and learning rate scheduling
7. Saving and loading PyTorch models

## 🔮 Future Improvements

Potential enhancements:
1. **Try LSTM/GRU**: Compare with advanced RNN variants
2. **Hyperparameter Tuning**: Grid search or random search
3. **Feature Engineering**: Create interaction terms or polynomial features
4. **Ensemble Methods**: Combine multiple models
5. **Cross-Validation**: K-fold validation for robust evaluation
6. **Attention Mechanism**: Add attention layers to RNN
7. **Comparative Analysis**: Compare with MLP, Random Forest, XGBoost

## 📝 Notes

- The model uses `batch_first=True` in RNN layer for easier batch handling
- Features are treated as a single time step (seq_length=1)
- Scaling is crucial for neural network convergence
- Random seed can be set for reproducibility

## 🤝 Contributing

Feel free to:
- Report bugs or issues
- Suggest improvements
- Add new features
- Improve documentation

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created as a demonstration of vanilla RNN implementation for regression tasks using PyTorch.