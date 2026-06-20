import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json
from PIL import Image

with open("class_names.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')
])

model.load_weights("model_bunga.weights.h5")

st.set_page_config(
    page_title="Klasifikasi Jenis Bunga",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Klasifikasi Jenis Bunga Menggunakan CNN")
st.write("Upload gambar bunga, lalu sistem akan memprediksi jenis bunganya.")

uploaded_file = st.file_uploader(
    "Upload gambar bunga",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    img = image.resize((150, 150))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    st.subheader("Hasil Prediksi")
    st.success(f"Jenis Bunga: {predicted_class}")
    st.info(f"Tingkat Keyakinan: {confidence:.2f}%")

    st.write("Probabilitas setiap kelas:")

    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i] * 100:.2f}%")
