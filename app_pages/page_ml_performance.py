import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from src.machine_learning.evaluate_clf import load_test_evaluation
import pickle
import os

def run():
    """
    This function displays the ML performance page.
    """
    st.title("Machine Learning Model Performance")

    st.write("### 📈 Model Evaluation")
    st.info(
        "**Objective:**\n"
        "This page presents the performance metrics of the trained machine learning model.\n"
        "These metrics are crucial for assessing the model's effectiveness in distinguishing"
        " between healthy and mildew-infected cherry leaves."
    )

    # Model versions
    version_v1 = 'v1'
    version_v2 = 'v2'
    selected_version_v1 = f"outputs/{version_v1}"
    selected_version_v2 = f"outputs/{version_v2}"

    # Load metrics for v2
    evaluation = load_test_evaluation(version_v2)

    # --- Display other plots ---
    st.write("---")
    st.write("### Average Image Size in Dataset")
    try:
        average_image_size = plt.imread(f"{selected_version_v2}/avg_img_size.png")
        st.image(average_image_size, caption='Average Image Size')
        st.warning(
            "The average image size in the provided dataset is: \n\n"
            "* Width average: 256px \n"
            "* Height average: 256px"
        )
    except FileNotFoundError:
        st.warning("Average image size plot not found.")
    st.write("---")

    st.write("### Train, Validation, and Test Set: Labels Frequencies")
    try:
        labels_distribution = plt.imread(f"{selected_version_v2}/labels_distribution.png")
        st.image(labels_distribution, caption='Labels Distribution')
        st.success(
            f"* Train - healthy: 1472 images\n"
            f"* Train - powdery_mildew: 1472 images\n"
            f"* Validation - healthy: 210 images\n"
            f"* Validation - powdery_mildew: 210 images\n"
            f"* Test - healthy: 422 images\n"
            f"* Test - powdery_mildew: 422 images\n"
        )
    except FileNotFoundError:
        st.warning("Labels distribution plot not found.")
    st.write("---")

    st.write("### Model History")
    st.info(
        "The model learning curve is used to check the model for "
        "overfitting and underfitting by plotting loss and accuracy."
    )
    col1, col2 = st.columns(2)
    try:
        with col1:
            model_acc = plt.imread(f"outputs/{version_v2}/training_validation_accuracy.png")
            st.image(model_acc, caption='Model Training Accuracy')
        with col2:
            model_loss = plt.imread(f"outputs/{version_v2}/training_validation_loss.png")
            st.image(model_loss, caption='Model Training Losses')
    except FileNotFoundError:
        st.warning("Model history plots not found.")
    st.write("---")

    st.write("### Generalised Performance on Test Set")

    evaluation_file_path = "/workspaces/mildew-detection-in-cherry-leaves/outputs/v2/evaluation.pkl"

    if os.path.exists(evaluation_file_path):
        try:
            with open(evaluation_file_path, 'rb') as f:
                evaluation = pickle.load(f)

            # Extract specific values for Loss and Accuracy
            loss = evaluation['test_loss']
            accuracy = evaluation['test_accuracy']

            # Display Loss and Accuracy
            st.write(f"**Test Loss:** {loss:.4f}")
            st.write(f"**Test Accuracy:** {accuracy * 100:.2f}%")

        except FileNotFoundError:
            st.warning(f"Evaluation file not found at: {evaluation_file_path}")
        except Exception as e:
            st.error(f"Error loading evaluation file: {e}")
    else:
        st.warning(f"Evaluation file does not exist at : {evaluation_file_path}")