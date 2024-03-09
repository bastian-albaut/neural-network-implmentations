import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, GlobalAveragePooling2D, Reshape, Add
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

input_layer = Input(name="input_layer", shape=(None, None, 1))

# Save input before the block
shortcut = input_layer

conv_layer = Conv2D(filters=128, kernel_size=(1, 1))(input_layer)
conv_layer = Activation("relu")(conv_layer)

# Add 3 more convolutional layers with residual connections
for _ in range(3):
    # Save input before the block
    residual = conv_layer
    
    conv_layer = Conv2D(filters=128, kernel_size=(7, 7), padding="same")(conv_layer)
    conv_layer = Activation("relu")(conv_layer)
    
    # Adding the residual connection
    conv_layer = Add()([conv_layer, residual])

pooled_layer = GlobalAveragePooling2D()(conv_layer)

reshaped_layer = Reshape((1, 1, 128))(pooled_layer)

output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", name="output_layer")(reshaped_layer)

model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

model.summary(150)
model.compile(optimizer="Adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["acc"])

# Training
model.fit(
    x_train[..., np.newaxis],  # Add channel dimension
    y_train,
    validation_data=(x_test[..., np.newaxis], y_test),  # Add channel dimension
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