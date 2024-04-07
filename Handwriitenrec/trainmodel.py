
import tensorflow as tf
from tensorflow import keras

# Load MNIST dataset and preprocess
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

# CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Convolutional layer with 32 filters
    keras.layers.MaxPooling2D((2, 2)),  # Max pooling layer
    keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer with 64 filters
    keras.layers.MaxPooling2D((2, 2)),  # Second max pooling layer
    keras.layers.Flatten(),  # Flatten the output before the dense layers
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

# Save the model
model.save('mnist_model')
