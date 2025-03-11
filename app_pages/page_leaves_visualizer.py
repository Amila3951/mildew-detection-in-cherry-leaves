import streamlit as st
import os
import matplotlib.pyplot as plt
import itertools
import random
import seaborn as sns
from PIL import Image

def run():
    st.title("Cherry Leaves Visualizer")

    st.write("### 🖼️ Visual Analysis of Cherry Leaves")
    st.info(
        "**Objective:**\n"
        "To visually explore the differences between healthy and powdery mildew-infected cherry leaves.\n\n"
        "**Key Insights:**\n"
        "- Identify visual patterns and distinguishing features.\n"
        "- Assess the effectiveness of image-based differentiation.\n"
    )

    version = 'v1'

    if st.checkbox("📊 Average and Variability Images"):
        avg_powdery_mildew = plt.imread(
            f"outputs/{version}/average_variability_powdery_mildew.png"
        )
        avg_healthy = plt.imread(f"outputs/{version}/average_variability_healthy.png")

        st.image(avg_powdery_mildew, caption="Infected Leaf - Average and Variability")
        st.image(avg_healthy, caption="Healthy Leaf - Average and Variability")

        st.write(
            "Observations: Analyze the average and variability images for noticeable differences in color, texture, or patterns."
        )
        st.write("---")

    if st.checkbox("🔍 Difference Between Average Images"):
        diff_between_avgs = plt.imread(f"outputs/{version}/class_differences_powdery_mildew_healthy.png")

        st.image(diff_between_avgs, caption="Difference Between Average Images")

        st.write(
            "Observations: Examine the difference image to highlight areas where healthy and infected leaves visually differ."
        )
        st.write("---")

    if st.checkbox("🖼️ Generate Image Montage"):
        st.write("Create a montage of healthy or infected leaves.")

        labels = ['healthy', 'powdery_mildew']
        label_to_display = st.selectbox(
            label="Select Label Category", options=labels, index=0
        )

        if st.button("Create Montage"):
            fig = cached_image_montage(label_to_display=label_to_display,
                                        nrows=4,
                                        ncols=2,
                                        figsize=(8, 12))

            if fig:
                st.pyplot(fig)

    st.write("---")

@st.cache_data
def cached_image_montage(label_to_display, nrows, ncols, target_size=(128, 128), figsize=(10, 25)):
    sns.set_style("white")

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    plot_idx = list(itertools.product(range(nrows), range(ncols)))

    image_paths = []
    if label_to_display == 'healthy':
        image_paths = [os.path.join('subset_data/cherry-leaves/healthy', filename) for filename in os.listdir('subset_data/cherry-leaves/healthy')]
    else:
        image_paths = [os.path.join('subset_data/cherry-leaves/powdery_mildew', filename) for filename in os.listdir('subset_data/cherry-leaves/powdery_mildew')]

    selected_images = random.sample(image_paths, nrows * ncols)

    for i, idx in enumerate(plot_idx):
        try:
            img = Image.open(selected_images[i])
            img = img.resize(target_size)
            axes[idx[0], idx[1]].imshow(img)
            axes[idx[0], idx[1]].set_title(f"Size: {target_size[0]}x{target_size[1]} pixels")
            axes[idx[0], idx[1]].set_xticks([])
            axes[idx[0], idx[1]].set_yticks([])
        except Exception as e:
            st.error(f"Error loading image {selected_images[i]}. Error: {e}")

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    run()