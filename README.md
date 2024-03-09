# DeepLearning

In this project I use Keras, an open-source and high-level neural networks API written in Python. I use Keras model that is the primary entity that represents the architecture of a neural network. Each file corresponds to a different model 
associated to a specific type of neural networks.

## Datasets

I use the MNIST dataset, which contains 60,000 training images and 10,000 testing images of hadwritten digits (Chiffres écrits à la main)

When we load the dataset, we get 4 numpy arrays:
x_train, y_train, x_test, y_test that correspond to the training and testing data.

- x_train and x_test parts contain greyscale RGB codes (from 0 to 255) 
- y_train and y_test parts contains labels from 0 to 9 which represents which number they actually are.

Datasets of Keras: https://keras.io/api/datasets/

## Test a model

1. Install Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

2. Setup the environment:
    ```bash
    conda create -n p311tf python=3.11 tensorflow numpy black matplotlib -c conda-forge

    conda activate p311tf
    ```

3. Run the model you want to test:
    ```bash
    python <name_of_the_file>.py
    ```

## Models

### 1. Simple Neural Network

This model is a simple neural network with 3 layers:
- **Input layer**
- **Hidden layer**
- **Output layer**

The input layer has 784 neurons (28x28 pixels), the hidden layer has 128 neurons and the output layer has 10 neurons (0-9).

### 2. Convolutional Neural Network

This model is a Convolutional Neural Network (CNN) with 7 layers:
- **Input layer**
- **Convolutional layer** of kernel size 1x1
- **3 Convolutional layers** of kernel size 7x7
- **Pooling layer**
- **Output layer**