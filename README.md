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

**<u>Idea of this model:</u>**<br>
Convolutional layers are more suited for image data as they can extract spatial hierarchies of features from the images.

### 3. Variadic transformation

This model is a Variadic Convolutional Neural Network (VCNN) with almost the same architecture as the CNN model. The difference is that the final dense layer becomes a convolutional layer of kernel size 1x1.

**<u>Idea of this model:</u>**<br>
The purpose of this model is to be capable of handling images of varying sizes.

### 4. Residual transformation

This model is a Convolutional Variadic Residual Neural Network (CVRNN) with the same architecture as the VCNN model. The difference is that the model uses residual transformations. 

**<u>Idea of this model:</u>**<br>
The purpose of this model is to add shortcuts to the network to avoid the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).

### 5. Sequentialy sepatated transformation

This model use the same architecture as the CVRNN model. The difference is that the model uses sequentially separated transformations. 

**<u>Idea of this model:</u>**<br>
This model involve breaking down larger convolutions into smaller, more manageable parts, which can increase depth and reduce parameters.