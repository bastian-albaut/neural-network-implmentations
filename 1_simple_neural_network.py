# Creation of a simple neural network with Keras to classify images from the MNIST dataset

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load data from MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train.shape: " + str(x_train.shape))
print("x_test.shape: " + str(x_test.shape))
print("y_train.shape: " + str(y_train.shape))
print("y_test.shape: " + str(y_test.shape))
print("y_train[1]: " + str(y_train[1]))

# Plot one of the data to see what’s inside
plt.imshow(x_train[1])
plt.show()

# Normalize the images data to be between 0 and 1 instead of 0 and 255
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the neural network model for classifying images
# The input layer is a 28x28 pixels image
input_layer = tf.keras.layers.Input(name="input_layer", shape=(28, 28))

# The flatten layer transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels
flatten_layer = tf.keras.layers.Flatten()(input_layer)

# Create a hidden layer with 128 neurons
hidden_layer = tf.keras.layers.Dense(units=128)(flatten_layer)

# Apply the ReLU activation function to the output of the hidden layer
hidden_layer = tf.keras.layers.Activation("relu")(hidden_layer)

# Apply dropout to the output of the hidden layer to prevent overfitting (overfitting is when a model is too specific to the training data and does not generalize well to new data)
hidden_layer = tf.keras.layers.Dropout(0.2)(hidden_layer)

# The output layer is a 10 neurons layer (one for each class 0-9)
output_layer = tf.keras.layers.Dense(units=10, name="output_layer")(hidden_layer)

model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

model.summary(150)

# Compile the model with an optimizer, a loss function and a metric
model.compile(
    optimizer = "Adam",
    loss = {
        "output_layer": tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )},
    metrics = ["acc"],
)

# ======== Test the model before training ======== 

# Get the 20 first images from the test set
test_logits = model(x_test[:20, ...])

# Apply the softmax function to the logits to get the probabilities
test_probabilities = tf.keras.activations.softmax(test_logits).numpy()

# Get the class with the highest probability
test_predictions = np.argmax(test_probabilities, axis=-1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=32)

print("=========== RESULTS BEFORE TRAINING ===========")

print(f"[test][pred] : {test_predictions}")
print(f"[test][real] : {y_test[:20]}")
print(f"[test][loss] : {test_loss}")
print(f"[test][accu] : {test_accuracy}")

# Train the model
model.fit(
    x_train,
    y_train,
    batch_size = 32,
    epochs = 5
)

# ======== Test the model after training ======== 

# Get the 20 first images from the test set
test_logits = model(x_test[:20, ...])

# Apply the softmax function to the logits to get the probabilities
test_probabilities = tf.keras.activations.softmax(test_logits).numpy()

# Get the class with the highest probability
test_predictions = np.argmax(test_probabilities, axis=-1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=32)

print("=========== RESULTS AFTER TRAINING ===========")

print(f"[test][pred] : {test_predictions}")
print(f"[test][real] : {y_test[:20]}")
print(f"[test][loss] : {test_loss}")
print(f"[test][accu] : {test_accuracy}")