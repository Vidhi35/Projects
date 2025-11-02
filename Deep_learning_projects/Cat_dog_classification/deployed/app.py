import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from PIL import Image

# Build the model architecture
def build_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Create and load the model
print("Building model architecture...")
model = build_model()
print("Loading weights from dog_vs_cat_model.h5...")
model.load_weights('dog_vs_cat_model.h5', by_name=True)

def preprocess_image(img):
    # Convert to RGB if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize and preprocess
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_animal(img):
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Make prediction
    prediction = model.predict(processed_img)
    confidence = float(prediction[0][0])
    
    # Return prediction with confidence
    if confidence > 0.5:
        return f"Dog (confidence: {confidence:.2%})"
    else:
        return f"Cat (confidence: {(1-confidence):.2%})"

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_animal,
    inputs=gr.Image(),  # Accept any size, we'll resize in preprocessing
    outputs="text",
    title="Dog vs Cat Classifier",
    description="Upload an image of a dog or cat to get a prediction. The model will return the predicted class with confidence score.",
    examples=[],  # You can add example images here if you have them
)

# Launch the interface
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model
model = tf.keras.models.load_model('../dog_vs_cat_model.h5')

def preprocess_image(image):
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize the image
    image = image.resize((150, 150))
    
    # Convert to array and preprocess
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def predict_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    
    # Get confidence scores
    dog_confidence = float(prediction[0][0])
    cat_confidence = 1 - dog_confidence
    
    # Create confidence dictionary
    return {
        "Cat": cat_confidence,
        "Dog": dog_confidence
    }

# Create Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    examples=[
        ["../dog_vs_cat/dogvscat/test/dogs/dog.1500.jpg"],
        ["../dog_vs_cat/dogvscat/test/cats/cat.1500.jpg"]
    ],
    title="Dog vs Cat Classifier",
    description="Upload an image of a dog or cat, and the model will predict which one it is!"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()