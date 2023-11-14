from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np
from io import BytesIO

# Load the model only once when the script is loaded
model_path = 'model/warning_signs_inceptionv3_model.h5'
model = load_model(model_path)

# Custom labels - Replace with your actual class names in the correct order
LABELS = ['caution', 'danger', 'warning']  # Modify this list based on your classes

def load_and_preprocess_image(image_bytes):
    """
    Load and preprocess the image for prediction.
    """
    img = image.load_img(BytesIO(image_bytes), target_size=(299, 299))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

def pred(image_bytes):
    """
    Make prediction on the given image bytes.
    """
    new_image = load_and_preprocess_image(image_bytes)
    predictions = model.predict(new_image)
    predicted_class = np.argmax(predictions, axis=1)
    return LABELS[predicted_class[0]]



