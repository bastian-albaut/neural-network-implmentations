import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, GlobalAveragePooling2D, Dense, Add, BatchNormalization, Dropout
from tensorflow.keras.datasets import cifar10
import numpy as np

# Load the Cifar-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Updated input shape for Cifar-10
input_layer = Input(name="input_layer", shape=(32, 32, 3))  

shortcut = input_layer

conv_layer = Conv2D(filters=128, kernel_size=(1, 1))(input_layer)
conv_layer = Activation("relu")(conv_layer)

for _ in range(3):
    residual = conv_layer
    
    for _ in range((9-1)//2):
        conv_layer = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation("tanh")(conv_layer)
        conv_layer = Dropout(0.2)(conv_layer)
    
    for _ in range((9-1)//2):
        conv_layer = Conv2D(filters=128, kernel_size=(3, 1), padding="same")(conv_layer)
        conv_layer = Conv2D(filters=128, kernel_size=(1, 3), padding="same")(conv_layer)
        conv_layer = BatchNormalization()(conv_layer) 
        conv_layer = Activation("tanh")(conv_layer) 
        conv_layer = Dropout(0.2)(conv_layer)
    
    conv_layer = Add()([conv_layer, residual])

pooled_layer = GlobalAveragePooling2D()(conv_layer)

# Updated output layer for Cifar-10
output_layer = Dense(units=10, activation="softmax", name="output_layer")(pooled_layer)  

model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

model.summary(150)
model.compile(optimizer="Adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["acc"])

# Training
model.fit(
    x_train[..., np.newaxis],
    y_train,
    validation_data=(x_test[..., np.newaxis], y_test),
    batch_size=32,
    epochs=5
)

# ======== Test the model after training ======== 

# Get the 20 first images from the test set
test_logits = model(x_test[:20, ..., np.newaxis])

# Apply the softmax function to the logits to get the probabilities
test_probabilities = tf.keras.activations.softmax(test_logits).numpy()

# Get the class with the highest probability
test_predictions = np.argmax(test_probabilities, axis=-1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test[..., np.newaxis], y_test, batch_size=32)

print("=========== RESULTS AFTER TRAINING ===========")

print(f"[test][pred] : {test_predictions}")
print(f"[test][real] : {y_test[:20]}")
print(f"[test][loss] : {test_loss}")
print(f"[test][accu] : {test_accuracy}")
