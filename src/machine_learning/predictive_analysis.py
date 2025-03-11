import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from PIL import Image
from src.data_management import load_pkl_file
import logging

logging.basicConfig(level=logging.INFO)

def plot_predictions_probabilities(pred_proba, pred_class):
    class_labels = ['Healthy', 'Infected']
    probabilities = [1 - pred_proba, pred_proba]

    prob_per_class = pd.DataFrame({
        'Condition': class_labels,
        'Probability': probabilities
    })

    fig = px.bar(
        prob_per_class,
        x='Condition',
        y='Probability',
        range_y=[0, 1],
        width=600,
        height=300,
        template='seaborn',
        color='Condition',
        title=f"Prediction Confidence: {pred_class.upper()}"
    )
    st.plotly_chart(fig)

def resize_input_image(img, version):
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    if isinstance(image_shape, tuple):
        image_shape = image_shape[0]
    img_resized = img.resize((image_shape, image_shape), Image.LANCZOS)
    my_image = np.expand_dims(img_resized, axis=0) / 255.0
    return my_image

def load_model_and_predict(my_image, version, model_path):
    try:
        logging.info(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully.")

        if isinstance(my_image, np.ndarray):
            my_image = np.squeeze(my_image)
            if my_image.shape[-1] == 4:
                my_image = Image.fromarray(np.uint8(my_image[:, :, :3]))
            else:
                my_image = Image.fromarray(np.uint8(my_image))
        elif not isinstance(my_image, Image.Image):
            st.error("Invalid image format. Please upload a valid image.")
            return None, None

        if my_image.mode == 'RGBA':
            my_image = my_image.convert('RGB')

        image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
        logging.info(f"Image shape loaded: {image_shape}")
        if isinstance(image_shape, tuple):
            image_shape = image_shape[0]

        logging.info(f"Image shape before resize: {my_image.size}")
        resized_img = my_image.resize((image_shape, image_shape))
        my_image = np.array(resized_img) / 255.0
        logging.info(f"Image shape after resize: {my_image.shape}")
        logging.info(f"Image data type: {my_image.dtype}")

        my_image = np.expand_dims(my_image, axis=0)

        pred_proba = model.predict(my_image)[0, 0]
        logging.info(f"Prediction probability: {pred_proba}")

        pred_class = "Infected" if pred_proba > 0.5 else "Healthy"
        return pred_proba, pred_class

    except FileNotFoundError as e:
        logging.error(f"Error: Model file not found. {e}")
        st.error(f"Error: Model file not found. {e}")
        return None, None

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        st.error(f"An unexpected error occurred: {e}")
        return None, None