> # Handwritten Digit Recognition using CNN

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) using PyTorch. The model is trained on a custom dataset organized in a specific folder structure.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [CNN Architecture](#cnn-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to build an accurate and efficient handwritten digit classifier. The implementation uses a custom `Dataset` class to load images from a directory structure where each digit has its own folder. The CNN model is built with PyTorch and includes convolutional layers, pooling, batch normalization, and dropout for regularization.

## Folder Structure

The project follows a specific folder structure for the dataset:

```
CNN/handwritten-digits-identifier/
└── archive/
    └── 0/
    │   └── image1.jpg
    │   └── ...
    └── 1/
    │   └── image1.jpg
    │   └── ...
    └── ...
    └── 9/
└── main.ipynb
└── README.md
```

## CNN Architecture

The CNN model has the following architecture:

1.  **Convolutional Block 1**:
    -   `Conv2d` (> # Handwritten Digit Recognition using CNN
    
    This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) using PyTorch. The model is trained on a custom dataset organized in a specific folder structure.
    
    ## Table of Contents
    - [Project Overview](#project-overview)
    - [Folder Structure](#folder-structure)
    - [CNN Architecture](#cnn-architecture)
    - [CNN Architecture Explained](#cnn-architecture-explained)
    - [Getting Started](#getting-started)
      - [Prerequisites](#prerequisites)
      - [Installation](#installation)
      - [Usage](#usage)
    - [Training and Evaluation](#training-and-evaluation)
    - [Results](#results)
    - [Contributing](#contributing)
    - [License](#license)
    
    ## Project Overview
    
    The goal of this project is to build an accurate and efficient handwritten digit classifier. The implementation uses a custom `Dataset` class to load images from a directory structure where each digit has its own folder. The CNN model is built with PyTorch and includes convolutional layers, pooling, batch normalization, and dropout for regularization.
    
    ## Folder Structure
    
    The project follows a specific folder structure for the dataset:
    
    ```
    CNN/handwritten-digits-identifier/
    └── archive/
        └── 0/
        │   └── image1.jpg
        │   └── ...
        └── 1/
        │   └── image1.jpg
        │   └── ...
        └── ...
        └── 9/
    └── main.ipynb
    └── README.md
    ```
    
    ## CNN Architecture
    
    The CNN model has the following architecture:
    
    1.  **Convolutional Block 1**:
        -   `Conv2d` (1 input channel, 32 output channels, 3x3 kernel)
        -   `BatchNorm2d`
        -   `ReLU` activation
        -   `MaxPool2d` (2x2 kernel)
    
    2.  **Convolutional Block 2**:
        -   `Conv2d` (32 input channels, 64 output channels, 3x3 kernel)
        -   `BatchNorm2d`
        -   `ReLU` activation
        -   `MaxPool2d` (2x2 kernel)
    
    3.  **Convolutional Block 3**:
        -   `Conv2d` (64 input channels, 128 output channels, 3x3 kernel)
        -   `BatchNorm2d`
        -   `ReLU` activation
        -   `MaxPool2d` (2x2 kernel)
    
    4.  **Fully Connected Layers**:
        -   `Dropout` (p=0.25)
        -   Flatten
        -   `Linear` (128 * 3 * 3 -> 256 neurons)
        -   `ReLU` activation
        -   `Dropout` (p=0.5)
        -   `Linear` (256 -> 10 neurons for output classes)
    
    ## CNN Architecture Explained
    
    The specific parameter values in the `DigitCNN` class were chosen based on common practices for image classification tasks, balancing model capacity with computational efficiency.
    
    ### Convolutional Layers (`nn.Conv2d`)
    
    -   **`conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)`**
        -   `in_channels=1`: Since the input images are grayscale, there is only one channel.
        -   `out_channels=32`: The first layer learns 32 different feature maps. This is a common starting point for simple image tasks.
        -   `kernel_size=3`: A 3x3 kernel is effective at capturing local patterns like edges and corners.
        -   `padding=1`: This preserves the input dimensions (28x28) after the convolution.
    
    -   **`conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)`**
        -   `in_channels=32`: Matches the output channels of the previous layer.
        -   `out_channels=64`: The number of filters is doubled to allow the model to learn more complex features.
    
    -   **`conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)`**
        -   `in_channels=64`: Matches the output of the second convolutional layer.
        -   `out_channels=128`: The number of filters is doubled again to capture even more abstract and complex patterns.
    
    ### Pooling Layer (`nn.MaxPool2d`)
    
    -   **`pool = nn.MaxPool2d(kernel_size=2, stride=2)`**
        -   `kernel_size=2`, `stride=2`: This downsamples the feature maps by half, reducing the spatial dimensions from 28x28 to 14x14, then 7x7, and finally 3x3. This reduces computation and makes the learned features more robust to small translations.
    
    ### Fully Connected Layers (`nn.Linear`)
    
    -   **`fc1 = nn.Linear(128 * 3 * 3, 256)`**
        -   `128 * 3 * 3`: This is the flattened size of the output from the last pooling layer (128 channels * 3x3 spatial dimension).
        -   `256`: A standard choice for the number of neurons in a hidden fully connected layer, providing a good balance between learning capacity and preventing overfitting.
    
    -   **`fc2 = nn.Linear(256, 10)`**
        -   `256`: Matches the output of the previous fully connected layer.
        -   `10`: The final output must be 10, corresponding to the 10 digit classes (0-9).
    
    ### Regularization and Normalization
    
    -   **`dropout1 = nn.Dropout(0.25)`** and **`dropout2 = nn.Dropout(0.5)`**
        -   Dropout is a regularization technique that randomly sets a fraction of neuron activations to zero during training to prevent overfitting. A 25% dropout is applied after the convolutional blocks, and a higher 50% dropout is used before the final classification layer, where overfitting is more likely.
    
    -   **`bn1`, `bn2`, `bn3` (`nn.BatchNorm2d`)**
        -   Batch normalization is used after each convolutional layer to stabilize and accelerate training by normalizing the activations.
    
    ## Getting Started
    
    ### Prerequisites
    
    - Python 3.x
    - PyTorch
    - TorchVision
    - scikit-learn
    - Matplotlib
    - Seaborn
    - Jupyter Notebook
    
    ### Installation
    
    1.  **Clone the repository**:
    
        ```bash
        git clone https://github.com/your-username/handwritten-digits-identifier.git
        cd handwritten-digits-identifier
        ```
    
    2.  **Install dependencies**:
    
        ```bash
        pip install torch torchvision scikit-learn matplotlib seaborn jupyter
        ```
    
    ### Usage
    
    1.  **Organize your dataset** as described in the [Folder Structure](#folder-structure) section.
    2.  **Open and run the Jupyter Notebook**:
    
        ```bash
        jupyter notebook main.ipynb
        ```
    
    3.  The notebook will train the model, evaluate it, and save the best-performing model as `best_model.pth`.
    
    ## Training and Evaluation
    
    - **Dataset Split**: The dataset is split into 70% for training, 15% for validation, and 15% for testing.
    - **Data Augmentation**: The training data is augmented with random rotations and translations to improve model generalization.
    - **Optimizer**: Adam optimizer with a learning rate of 0.001.
    - **Loss Function**: Cross-Entropy Loss.
    - **Epochs**: 20
    
    ## Results
    
    The model's performance is evaluated using accuracy, a confusion matrix, and a classification report. The training and validation history (loss and accuracy) are plotted to visualize the learning process.
    
    ## Contributing
    
    Contributions are welcome! Please feel free to submit a pull request or open an issue.
    
    ## License
    
    This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
    
    1 input channel, 32 output channels, 3x3 kernel)
    -   `BatchNorm2d`
    -   `ReLU` activation
    -   `MaxPool2d` (2x2 kernel)

2.  **Convolutional Block 2**:
    -   `Conv2d` (32 input channels, 64 output channels, 3x3 kernel)
    -   `BatchNorm2d`
    -   `ReLU` activation
    -   `MaxPool2d` (2x2 kernel)

3.  **Convolutional Block 3**:
    -   `Conv2d` (64 input channels, 128 output channels, 3x3 kernel)
    -   `BatchNorm2d`
    -   `ReLU` activation
    -   `MaxPool2d` (2x2 kernel)

4.  **Fully Connected Layers**:
    -   `Dropout` (p=0.25)
    -   Flatten
    -   `Linear` (128 * 3 * 3 -> 256 neurons)
    -   `ReLU` activation
    -   `Dropout` (p=0.5)
    -   `Linear` (256 -> 10 neurons for output classes)

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- TorchVision
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

### Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/handwritten-digits-identifier.git
    cd handwritten-digits-identifier
    ```

2.  **Install dependencies**:

    ```bash
    pip install torch torchvision scikit-learn matplotlib seaborn jupyter
    ```

### Usage

1.  **Organize your dataset** as described in the [Folder Structure](#folder-structure) section.
2.  **Open and run the Jupyter Notebook**:

    ```bash
    jupyter notebook main.ipynb
    ```

3.  The notebook will train the model, evaluate it, and save the best-performing model as `best_model.pth`.

## Training and Evaluation

- **Dataset Split**: The dataset is split into 70% for training, 15% for validation, and 15% for testing.
- **Data Augmentation**: The training data is augmented with random rotations and translations to improve model generalization.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Loss Function**: Cross-Entropy Loss.
- **Epochs**: 20

## Results

The model's performance is evaluated using accuracy, a confusion matrix, and a classification report. The training and validation history (loss and accuracy) are plotted to visualize the learning process.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

