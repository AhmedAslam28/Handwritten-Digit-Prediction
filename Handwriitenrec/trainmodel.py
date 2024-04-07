
import tensorflow as tf
from tensorflow import keras

# Load MNIST dataset and preprocess
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

# CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  
    keras.layers.MaxPooling2D((2, 2)),  
    keras.layers.Conv2D(64, (3, 3), activation='relu'), 
    keras.layers.MaxPooling2D((2, 2)),  
    keras.layers.Flatten(),  
    keras.layers.Dense(128, activation='relu'), 
    keras.layers.Dropout(0.5),  
    keras.layers.Dense(10, activation='softmax')  
]) 


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5)

# Save the model
model.save('mnist_model')
