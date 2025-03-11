import streamlit as st
import os
import matplotlib.pyplot as plt
import itertools
import random
import seaborn as sns
from PIL import Image
import shutil

def generate_subset(source_dir, subset_dir, num_images=8):
    """
    Generates a subset of images from the source directory.

    Args:
        source_dir (str): Path to the source directory.
        subset_dir (str): Path to the subset directory.
        num_images (int): Number of images to include in the subset.
    """
    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir) # Creates the directory if it does not exist

    images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] 
    selected_images = random.sample(images, min(num_images, len(images))) # Selects a random number of images

    for img in selected_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join(subset_dir, img)) 

def resize_images(input_dir, output_dir, size=(128, 128)):
    """
    Resizes images in the input directory and saves them to the output directory.

    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
        size (tuple): Target size of the images.
    """
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(input_dir, filename))
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(os.path.join(output_dir, filename))

# Generate subset and resize images
generate_subset('data/cherry-leaves/healthy', 'subset_data/cherry-leaves/healthy')
generate_subset('data/cherry-leaves/powdery_mildew', 'subset_data/cherry-leaves/powdery_mildew')
resize_images('subset_data/cherry-leaves/healthy', 'subset_data/cherry-leaves/healthy')
resize_images('subset_data/cherry-leaves/powdery_mildew', 'subset_data/cherry-leaves/powdery_mildew')

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

    # Section: Average and Variability Images
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

    # Section: Differences between average images
    if st.checkbox("🔍 Difference Between Average Images"):
        diff_between_avgs = plt.imread(f"outputs/{version}/class_differences_powdery_mildew_healthy.png")

        st.image(diff_between_avgs, caption="Difference Between Average Images")

        st.write(
            "Observations: Examine the difference image to highlight areas where healthy and infected leaves visually differ."
        )
        st.write("---")

    # Section: Image Montage
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
    """
    Function to create and display a montage of images.

    Args:
      label_to_display (str): The label of the images to display ('healthy' or 'powdery_mildew').
      nrows (int): The number of rows in the montage.
      ncols (int): The number of columns in the montage.
      target_size (tuple): The target size of each image in the montage.
      figsize (tuple): The figure size of the montage.

    Returns:
      matplotlib.figure.Figure: The figure containing the montage.
    """
    sns.set_style("white")

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    plot_idx = list(itertools.product(range(nrows), range(ncols)))

    # Load image paths from the subset directory
    image_paths = []
    if label_to_display == 'healthy':
        image_paths = [os.path.join('subset_data/cherry-leaves/healthy', filename) for filename in os.listdir('subset_data/cherry-leaves/healthy')]
    else:
        image_paths = [os.path.join('subset_data/cherry-leaves/powdery_mildew', filename) for filename in os.listdir('subset_data/cherry-leaves/powdery_mildew')]

    # Randomly select 8 images
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

    plt.tight_layout() # Improves subplot padding
    return fig

if __name__ == "__main__":
    run()