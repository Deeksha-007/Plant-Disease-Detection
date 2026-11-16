import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Image size used during CNN training
IMG = 128
def build_model(n):
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((IMG, IMG, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n, activation="softmax")
    ])


# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# Load trained CNN model and class names
@st.cache_resource
def load():
    with open("class_names.json", "r") as file:
        names = json.load(file)

    model = build_model(len(names))
    model.load_weights("model.weights.h5")

    return model, names

# Convert class names into readable text
def clean(name):
    return name.replace("_", " - ").replace("_", " ")


# Load model
try:
    model, class_names = load()
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()


# Application heading
st.title("🌱 Plant Disease Detection")
st.write(
    "Upload a leaf image or take a photo using your camera "
    "to identify the plant disease."
)

st.info("Our CNN model can classify images into 38 PlantVillage classes.")


# Select image input method
mode = st.radio(
    "Choose input method:",
    ["Upload image", "Take a photo"],
    horizontal=True
)


# Upload image option
if mode == "Upload image":

    file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

# Camera option
else:

    st.caption(
        "Tip: Place one leaf clearly in good lighting "
        "for better image quality."
    )

    file = st.camera_input("📷 Take a leaf photo")


# Process image and predict
if file is not None:

    try:
        # Read image
        img = Image.open(file).convert("RGB")

        # Display image
        st.image(
            img,
            caption="Selected leaf image",
            use_container_width=True
        )

        # Preprocess image
        resized_img = img.resize((IMG, IMG))

        x = np.array(
            resized_img,
            dtype="float32"
        ) / 255.0

        x = np.expand_dims(x, axis=0)

        # Make prediction
        with st.spinner("Analyzing the leaf image..."):

            probs = model.predict(x, verbose=0)[0]

        # Get top 3 predictions
        top = np.argsort(probs)[::-1][:3]

        best_index = top[0]
        best_class = class_names[best_index]
        confidence = float(probs[best_index]) * 100

        # Display main prediction
        st.subheader("Prediction Result")

        st.success(f"Predicted class: {clean(best_class)}")

        st.write(f"*Confidence: {confidence:.2f}%*")

        # Display healthy or disease status
        if "healthy" in best_class.lower():

            st.success("No disease detected – Healthy.")

        else:

            st.warning("Possible plant disease identified.")

        # Display top 3 predictions
        st.subheader("Top 3 Predictions")

        for i in top:

            st.write(
                f"• {clean(class_names[i])}: "
                f"{probs[i] * 100:.2f}%"
            )

        st.caption(
            "Note: Predictions are based on the PlantVillage "
            "dataset. Results may be less reliable for real-world "
            "field images or unfamiliar plants."
        )

    except Exception as e:

        st.error(f"Error processing image: {e}")

else:

    st.info("Please upload a leaf image or capture one using your camera.")
