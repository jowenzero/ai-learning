# Student Pass/Fail Classification using Artificial Neural Network (ANN)

## Overview

This project implements an **Artificial Neural Network (ANN)** using PyTorch's `nn.Linear` layers and `nn.ReLU` activation functions to classify whether a student will pass or fail based on various academic and demographic features.

## Model Architecture

The ANN consists of the following layers:

1. **Input Layer**: 8 features
2. **Hidden Layer 1**: 64 neurons with ReLU activation
3. **Hidden Layer 2**: 32 neurons with ReLU activation
4. **Hidden Layer 3**: 16 neurons with ReLU activation
5. **Output Layer**: 2 neurons (binary classification: Pass/Fail)

### Architecture Diagram

```
Input (8) → Linear(8, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 2) → Output
```

**Total Parameters**: 3,218

### Why 64/32/16 Neurons?

The choice of 64, 32, and 16 neurons for the three hidden layers follows a common neural network design pattern:

#### 1. **Pyramidal/Funnel Structure**

The decreasing pattern (64 → 32 → 16 → 2) creates a **funnel architecture** that progressively compresses information:

- **Layer 1 (64 neurons)**: Expands from 8 input features to 64 neurons, allowing the network to learn more complex feature representations and combinations
- **Layer 2 (32 neurons)**: Begins compression, filtering out less important features while retaining key patterns
- **Layer 3 (16 neurons)**: Further compression, creating high-level abstract representations
- **Output (2 neurons)**: Final compression to binary classification (Pass/Fail)

This mimics how information flows from detailed features to abstract concepts to final decisions.

#### 2. **Power-of-2 Convention**

The numbers 64, 32, and 16 are powers of 2 (2⁶, 2⁵, 2⁴), which is a common practice because:
- Computationally efficient for GPU/CPU operations
- Easy to scale and adjust (halving/doubling)
- Standard convention in deep learning

#### 3. **Balancing Capacity and Overfitting**

- **Too many neurons**: Risk of overfitting (memorizing training data)
- **Too few neurons**: Risk of underfitting (unable to learn patterns)
- The 64/32/16 configuration provides sufficient capacity for this dataset (708 samples, 8 features) without being excessive

#### 4. **Rule of Thumb**

A common heuristic is:
- First hidden layer: 2-3× the input size (8 × 8 = 64 is reasonable)
- Subsequent layers: Gradually decrease toward output size

#### Alternative Architectures to Consider

You could experiment with different configurations:

- **Wider networks**: 128/64/32 (more capacity, but risk overfitting on small datasets)
- **Narrower networks**: 32/16/8 (faster training, less capacity)
- **Uniform width**: 32/32/32 (equal capacity at each layer)
- **Different patterns**: 64/48/24 or 50/25/10

#### How to Choose?

The optimal architecture depends on:

1. **Dataset size**: Larger datasets can support wider networks
2. **Feature complexity**: More complex relationships need more neurons
3. **Empirical testing**: Try different configurations and compare validation accuracy
4. **Computational budget**: More neurons = longer training time

For this dataset with 708 samples and 8 features, the 64/32/16 architecture achieved 99.30% test accuracy, demonstrating it is well-suited for this classification task.

## Dataset Features

The model uses the following 8 features for prediction:

1. **Gender** (Male/Female)
2. **Study_Hours_per_Week** (numerical)
3. **Attendance_Rate** (percentage)
4. **Past_Exam_Scores** (numerical)
5. **Parental_Education_Level** (High School/Bachelors/Masters/PhD)
6. **Internet_Access_at_Home** (Yes/No)
7. **Extracurricular_Activities** (Yes/No)
8. **Final_Exam_Score** (numerical)

**Target Variable**: Pass_Fail (Pass/Fail)

## Results

### Model Performance

- **Test Accuracy**: 99.30%
- **Training Accuracy**: 100.00% (final epoch)

### Classification Report

```
              precision    recall  f1-score   support
        Fail       0.99      1.00      0.99        71
        Pass       1.00      0.99      0.99        71
    accuracy                           0.99       142
   macro avg       0.99      0.99      0.99       142
weighted avg       0.99      0.99      0.99       142
```

### Confusion Matrix

```
[[71  0]
 [ 1 70]]
```

The model correctly classified:
- **71 out of 71** students who failed (100% recall for Fail class)
- **70 out of 71** students who passed (98.6% recall for Pass class)
- Only **1 misclassification** out of 142 test samples

## Files Included

1. **student_classifier.py** - Complete Python script for training the ANN
2. **main.ipynb** - Jupyter notebook with step-by-step implementation
3. **student_ann_model.pth** - Saved trained model weights
4. **preprocessing_objects.pkl** - Saved preprocessing objects (scalers, encoders)
5. **training_metrics.png** - Visualization of training loss and accuracy
6. **confusion_matrix.png** - Confusion matrix heatmap

## How to Use

### Training the Model

Run the Python script:
```bash
python3 student_classifier.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

### Making Predictions

```python
import torch
import pickle

# Load the model
model = StudentANN(input_size=8)
model.load_state_dict(torch.load('student_ann_model.pth'))
model.eval()

# Load preprocessing objects
with open('preprocessing_objects.pkl', 'rb') as f:
    preprocessing_objects = pickle.load(f)

scaler = preprocessing_objects['scaler']
label_encoders = preprocessing_objects['label_encoders']
target_encoder = preprocessing_objects['target_encoder']

# Prepare new data (encode and scale)
# ... (encode categorical variables and scale features)

# Make prediction
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    prediction = target_encoder.inverse_transform(predicted.numpy())[0]
```

## Key Implementation Details

### Activation Function: ReLU

The model uses **ReLU (Rectified Linear Unit)** activation function:
- Formula: `f(x) = max(0, x)`
- Benefits: Prevents vanishing gradient problem, computationally efficient, introduces non-linearity

### Loss Function

**CrossEntropyLoss** is used for binary classification, which combines:
- LogSoftmax activation
- Negative Log Likelihood Loss

### Optimizer

**Adam optimizer** with learning rate of 0.001:
- Adaptive learning rates for each parameter
- Combines advantages of AdaGrad and RMSProp

### Data Preprocessing

1. **Label Encoding**: Categorical variables converted to numerical values
2. **Standardization**: Features scaled using StandardScaler (mean=0, std=1)
3. **Train-Test Split**: 80% training, 20% testing with stratification

## Training Configuration

- **Epochs**: 100
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

## Requirements

```
torch
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Conclusion

The ANN model successfully classifies student pass/fail status with **99.30% accuracy** on the test set. The model architecture using `nn.Linear` layers with `nn.ReLU` activation functions proves to be highly effective for this binary classification task.

