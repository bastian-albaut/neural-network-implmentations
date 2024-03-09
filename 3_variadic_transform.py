import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, GlobalAveragePooling2D, Dense
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Allowing for variability in image size
input_layer = Input(name="input_layer", shape=(None, None, 1))  
conv_layer = Conv2D(filters=128, kernel_size=(1, 1))(input_layer)  
conv_layer = Activation("relu")(conv_layer)

for _ in range(3):
    conv_layer = Conv2D(filters=128, kernel_size=(7, 7), padding="same")(conv_layer)
    conv_layer = Activation("relu")(conv_layer)

pooled_layer = GlobalAveragePooling2D()(conv_layer)

# Change input shape to (None, None, 1)
output_layer = Conv2D(filters=10, kernel_size=(1, 1))(input_layer)
output_layer = GlobalAveragePooling2D()(output_layer)

model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

model.summary(150)
model.compile(optimizer="Adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["acc"])

# Training
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=5
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