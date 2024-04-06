from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)

# Load your trained deep learning model
model = tf.keras.models.load_model('mnist_model')

@app.route('/')
def index():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image data from the request
    img = request.files['file'].read()
    img = Image.open(io.BytesIO(img)).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to the input size of your model
    img_array = np.array(img) / 255.0  # Normalize the pixel values
    img_array = img_array.reshape(1, 28, 28)    # Convert to grayscale
   # Reshape for model input

    # Make prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    return str(predicted_digit)

if __name__ == '__main__':
    app.run(debug=True)
