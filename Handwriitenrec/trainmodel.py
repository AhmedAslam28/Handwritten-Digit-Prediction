
import tensorflow as tf
from tensorflow import keras

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)
    
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the input images
    keras.layers.Dense(128, activation='relu'),  # Add a hidden dense layer with 128 units and ReLU activation
    keras.layers.Dropout(0.5),  # Add dropout layer to reduce overfitting
    keras.layers.Dense(10, activation='softmax')  # Add output layer with 10 units (one for each digit) and softmax activation
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



model.fit(X_train_flattened, y_train, epochs=5)
model.save('mnist_model')