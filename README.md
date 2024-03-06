# DeepLearning

In this project I use Keras, an open-source and high-level neural networks API written in Python. I use Keras model that is the primary entity that represents the architecture of a neural network.

## Datasets

I use the MNIST dataset, which contains 60,000 training images and 10,000 testing images of hadwritten digits (Chiffres écrits à la main)

When we load the dataset, we get 4 numpy arrays:
x_train, y_train, x_test, y_test that correspond to the training and testing data and labels.

- x_train and x_test parts contain greyscale RGB codes (from 0 to 255) 
- y_train and y_test parts contains labels from 0 to 9 which represents which number they actually are.

Datasets of Keras: https://keras.io/api/datasets/

## Test a model

Setup the environment:
```bash
conda create -n p311tf python=3.11 tensorflow numpy black matplotlib -c conda-forge

conda activate p311tf
```

Run the model:
```bash
python <name_of_the_file>.py
```