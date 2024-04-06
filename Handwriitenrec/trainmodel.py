
import tensorflow as tf
from tensorflow import keras

# Load MNIST dataset and preprocess
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape input data and define the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten input images from 28x28 to 784
    keras.layers.Dense(128, activation='relu'),  # Hidden dense layer with 128 units
    keras.layers.Dropout(0.5),  # Dropout layer for regularization
    keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units (for digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5)

# Evaluate the model on test data


# Save the model
model.save('mnist_model')
