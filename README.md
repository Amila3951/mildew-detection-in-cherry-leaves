# **Powdery Mildew Detection in Cherry Leaves**

This project introduces a system for detecting powdery mildew on cherry leaves, developed using machine learning techniques. The application allows users to upload photos of cherry leaves and receive an analysis indicating the presence or absence of the disease. The system generates a downloadable report with the analysis results.

**The application is available at [Powdery Mildew Detection in Cherry Leaves]()**

This tool was created with the aim of aiding in the early detection of powdery mildew, a common disease affecting cherry leaves. Using advanced machine learning algorithms, the application analyzes the visual characteristics of the leaves in uploaded photos and classifies them as healthy or infected.

**Key Features:**

* **Simple User Interface:** Enables easy photo uploads.
* **Fast Analysis:** Provides results in a short period of time.
* **Analysis Report:** Generates a detailed, downloadable report of the results.
* **Machine Learning Powered:** Utilizes sophisticated models for accurate detection.


## Dataset Description

The dataset utilized in this project was sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). This dataset was selected to provide a real-world context for applying predictive analytics, simulating a scenario where such techniques could be implemented in a professional setting.

Comprising approximately 4,000 images, the dataset captures cherry leaves collected directly from the client's agricultural fields. These images depict both healthy leaves and those exhibiting symptoms of powdery mildew, a fungal disease that affects a wide range of plant species. Given that the cherry plantation represents a key product line for the corporation, maintaining high product quality is paramount. The presence of powdery mildew poses a significant concern regarding the potential degradation of product standards, necessitating the development of effective detection and mitigation strategies.

## Business Objectives

Farmy & Foods' cherry orchards have experienced outbreaks of powdery mildew, impacting crop yield. Currently, the detection of powdery mildew involves a manual inspection process, where an employee spends approximately 30 minutes per tree, collecting leaf samples and performing visual assessments. If powdery mildew is detected, a treatment is applied, which takes approximately one minute per tree. With thousands of cherry trees distributed across multiple farms nationwide, this manual inspection method is proving to be inefficient and unsustainable.

To address this challenge, the IT department has proposed developing a machine learning-based system that can rapidly analyze leaf images to determine the presence or absence of powdery mildew. This solution aims to significantly reduce inspection time and improve scalability. Furthermore, the success of this system could pave the way for its application to other crops within Farmy & Foods' portfolio, where similar manual pest detection methods are currently employed. The dataset provided for this project comprises images of cherry leaves collected from the company's farms.

**Specific Requirements:**

1.  Develop a system capable of accurately differentiating between cherry leaves affected by powdery mildew and healthy leaves based on visual analysis.
2.  Implement a classification model that can determine whether a given cherry leaf image depicts a healthy leaf or one infected with powdery mildew.

## Hypotheses and Validation Methodology

### ***Hypotheses***

1.  **Distinct Visual Patterns:** Observable pattern differences between images of healthy and powdery mildew-affected cherry leaves can serve as a basis for accurate classification.
2.  **Subtle Feature Variations:** Powdery mildew infection induces subtle variations in color and form within cherry leaves, which are detectable through image analysis.
3.  **High-Accuracy Model Development:** The provided image dataset is sufficient for training a machine learning model capable of classifying cherry leaves with a minimum accuracy of 97% in terms of powdery mildew presence.

### ***Validation Methodology***

1.  **Visual Pattern Analysis:**
    * Conduct exploratory data analysis (EDA) on the image dataset to identify and document distinct visual patterns.
    * Utilize image processing techniques (e.g., feature extraction, edge detection) to quantify and compare these patterns.
    * Employ visualization tools to illustrate the observed differences between healthy and infected leaves.

2.  **Feature Variation Analysis:**
    * Apply color analysis techniques (e.g., histogram analysis, color space transformations) to detect subtle color variations.
    * Implement shape analysis techniques (e.g., contour detection, geometric feature extraction) to identify subtle form variations.
    * Employ statistical methods to assess the significance of these variations.

3.  **Model Performance Evaluation:**
    * Train and evaluate various machine learning models (e.g., Convolutional Neural Networks) using the provided dataset.
    * Utilize appropriate evaluation metrics (e.g., accuracy, precision, recall, F1-score) to assess model performance.
    * Perform cross-validation to ensure model robustness and generalization.
    * Conduct hyperparameter tuning to optimize model performance.
    * Compare results against the 97% accuracy goal.


    ### ***Validation Results***

-   **Visual Differentiation:**
    * A comparative image montage visually demonstrates the discernible differences between healthy cherry leaves and those affected by powdery mildew.
    * [Image montage healthy leaves](/workspaces/mildew-detection-in-cherry-leaves/images/healthy.png)
    * [Image montage mildew leaves](/workspaces/mildew-detection-in-cherry-leaves/images/powdery.png)
    * This visual assessment supports the hypothesis that distinct patterns are observable.

-   **Color and Form Analysis:**
    * Analysis of average color, color difference, and color variability within the central region of each leaf image revealed quantifiable color variations between healthy and infected leaves.
    * [Average Healthy](/workspaces/mildew-detection-in-cherry-leaves/outputs/v1/average_variability_healthy.png)
    * [Average Infected](/workspaces/mildew-detection-in-cherry-leaves/outputs/v1/average_variability_powdery_mildew.png)
    * [Difference](/workspaces/mildew-detection-in-cherry-leaves/outputs/v1/class_differences_powdery_mildew_healthy.png)
    * While color differences were evident, no readily apparent shape-based patterns were identified that could reliably distinguish between the two categories.

-   **Machine Learning Model Performance:**
    * The machine learning pipeline achieved a classification accuracy of 99% in distinguishing between healthy and diseased cherry leaves.
    * [Performance Metrics](/workspaces/mildew-detection-in-cherry-leaves/images/performance_metrics.png)
    * This result validates the hypothesis that a high-accuracy model could be developed using the provided dataset.
    * The performance metrics provided by the image, should be discussed in the text, for example: "The performance metrics displayed in the image, showcase a high level of precision, recall and F1 score, thus validating the model's accuracy."

**Key Changes and Rationale:**

* **"Validation" to "Validation Results":** More specific and descriptive.
* **Structured Presentation:** Use of bullet points to organize findings.
* **Descriptive Language:** More precise and professional phrasing.
* **Image References:** Maintained image links for visual support.
* **Detailed Analysis:** Added context and explanations for each validation point.
* **Clear Connection to Hypotheses:** Explicitly stated how the results support or refute the initial hypotheses.
* **Improved Flow:** Transitions between points are smoother and more logical.
* **More precise language:** Changed "apparent distinction" to "discernible differences" and "supported the notion" to "revealed quantifiable color variations".
* Added the suggestion to describe the performance metric image.
* Added the phrase "This visual assessment supports the hypothesis that distinct patterns are observable." to more clearly connect the image montages to the hypothesis.

These changes aim to present the validation results in a more professional, detailed, and impactful manner.


## Rationale for Mapping Business Objectives to Data Visualizations and Machine Learning Tasks

## Rationale for Mapping Business Objectives to Data Visualizations and Machine Learning Tasks

### ***Business Objective 1: Visual Differentiation Research***

The client requires a visual analysis to distinguish between healthy cherry leaves and those affected by powdery mildew.

> **Data Visualization Strategy**

* **Comparative Analysis of Mean and Standard Deviation Images:** The interactive dashboard will present mean and standard deviation images of both healthy and powdery mildew-infected cherry leaves. This will allow for a direct visual comparison, highlighting characteristic features and variability within each category.
* **Visualization of Average Leaf Difference:** A dedicated visualization will demonstrate the difference between the average healthy leaf and the average powdery mildew-infected leaf. This will clearly illustrate the key visual distinctions that differentiate the two categories.
* **Image Montage for Side-by-Side Comparison:** An image montage featuring representative samples of both healthy and powdery mildew-infected leaves will be displayed. This side-by-side comparison will facilitate the identification of subtle and obvious visual patterns.

### ***Business Objective 2: Image Classification for Powdery Mildew Detection***

The client needs a system to classify individual cherry leaf images as either healthy or infected with powdery mildew.

> **Machine Learning Task: Binary Classification**

* **Development of a Binary Classification Model:** A machine learning model, specifically a binary classifier, will be developed to analyze input cherry leaf images and accurately predict whether each leaf is healthy or infected with powdery mildew. This model will provide an automated and efficient solution for image classification.
* **Model Performance Evaluation:** The model's performance will be rigorously evaluated by analyzing loss and accuracy metrics. This will ensure the model's reliability and effectiveness in classifying leaf images.
* **Prediction Report Generation and Download:** Functionality will be implemented to allow users to generate and download a comprehensive prediction report for uploaded photos. This report will provide detailed classification results, enhancing the user experience and facilitating data sharing.

**Key Changes and Rationale:**

* **"Data Visualization" to "Data Visualization Strategy" and "Classification" to "Machine Learning Task: Binary Classification":** More descriptive and accurate.
* **"The dashboard will show" to "The interactive dashboard will present":** Added "interactive" to convey a more dynamic user experience.
* **Added Explanations:** More detailed descriptions of the purpose and value of each visualization and machine learning task.
* **"Build a binary classifier ML model" to "Development of a Binary Classification Model":** More formal and precise.
* **"Evaluate the Ml model's loss and accuracy" to "Model Performance Evaluation":** More professional and encompassing.
* **"Add an option for users to generate and download a prediction report for uploadable photos" to "Prediction Report Generation and Download":** More concise and descriptive.
* **Added "This will allow for a direct visual comparison, highlighting characteristic features and variability within each category." and other similar explanations:** This helps to clarify the purpose of each step.
* **Added "This will clearly illustrate the key visual distinctions that differentiate the two categories.":** To better explain the value of the average leaf difference visualization.
* **Added "This side-by-side comparison will facilitate the identification of subtle and obvious visual patterns.":** To better explain the value of the image montage.
* **Added "This model will provide an automated and efficient solution for image classification.":** to explain the outcome of the ML task.
* **Added "This will ensure the model's reliability and effectiveness in classifying leaf images.":** To explain the value of the model performance evaluation.
* **Added "This report will provide detailed classification results, enhancing the user experience and facilitating data sharing.":** To explain the value of the prediction report.

These changes provide a more comprehensive and professional overview of the rationale behind mapping business objectives to specific project tasks.


## Cross-Industry Standard Process for Data Mining (CRISP-DM) Methodology

This project followed the Cross-Industry Standard Process for Data Mining (CRISP-DM) methodology to ensure a structured and comprehensive approach to development.

![Crisp- DM image](/workspaces/mildew-detection-in-cherry-leaves/images/crisp.png)

**1. Business Understanding**

The project addresses two key business objectives:

> **Business Objective 1: Visual Differentiation Research**

* Conduct a visual analysis to distinguish between healthy cherry leaves and those affected by powdery mildew.
    * Analyze average images and variability images for each class (healthy or powdery mildew).
    * Examine the differences between average healthy and average powdery mildew cherry leaves.
    * Create an image montage for each class.

> **Business Objective 2: Image Classification for Powdery Mildew Detection**

* Develop an ML system capable of predicting whether a cherry leaf is healthy or contains powdery mildew.

**2. Data Understanding**

The project utilizes the [Kaggle dataset](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) provided by Code Institute, containing over 4,000 images of healthy and affected cherry leaves.

* **Data Collection:** Retrieve data from the Kaggle dataset and store it as raw data.

**3. Data Preparation**

* **Data Cleaning:** Clean the data by checking for and removing any non-image files.
* **Dataset Splitting:** Split the data into training, validation, and test sets.
* **Image Standardization:** Define and apply consistent image shapes for processing.
* **Image Analysis:**
    * Calculate the average and variability of images for each class (healthy and powdery mildew).
    * Load image shapes and labels into an array.
    * Plot and save the mean variability of images for each class.
    * Calculate and visualize the difference between the average healthy and powdery mildew-infected leaf.
    * Create image montages for visual comparison.
* **Data Augmentation:** Apply image data augmentation techniques to increase dataset diversity and model robustness.

**4. Modeling**

* **Model Selection:** Choose an appropriate machine learning model, such as a Convolutional Neural Network (CNN), for image classification.
* **Model Training:** Train the selected ML model using the training dataset.
* **Model Saving:** Save the trained model for future use and deployment.

**5. Evaluation**

* **Performance Visualization:** Plot the model's learning curve to visualize training loss and accuracy over time.
* **Model Testing:** Evaluate the trained model's performance on the test dataset using appropriate metrics (e.g., accuracy, precision, recall, F1-score).
* **Prediction Testing:**
    * Load a random image for prediction.
    * Convert the image to an array and prepare it for model input.
    * Predict class probabilities and evaluate the results.

**6. Deployment**

* **Model Deployment:** Deploy the trained and evaluated model into a production environment for real-world use.

**Key Changes and Rationale:**

* **"Business Requirement" to "Business Objective":** Consistent terminology.
* **Added "Visual Differentiation Research" and "Image Classification for Powdery Mildew Detection" :** To more clearly describe the business objectives.
* **Added "Data Collection:" :** To more clearly describe the data collection step.
* **Added "Data Cleaning:" :** To more clearly describe the data cleaning step.
* **Added "Dataset Splitting:" :** To more clearly describe the dataset splitting step.
* **Added "Image Standardization:" :** To more clearly describe the image standardization step.
* **Added "Image Analysis:" :** To more clearly describe the image analysis step.
* **Added "Data Augmentation:" :** To more clearly describe the data augmentation step.
* **Added "Model Selection:" :** To more clearly describe the model selection step.
* **Added "Model Training:" :** To more clearly describe the model training step.
* **Added "Model Saving:" :** To more clearly describe the model saving step.
* **Added "Performance Visualization:" :** To more clearly describe the performance visualization step.
* **Added "Model Testing:" :** To more clearly describe the model testing step.
* **Added "Prediction Testing:" :** To more clearly describe the prediction testing step.
* **Added "Model Deployment:" :** To more clearly describe the model deployment step.

These changes aim to provide a more detailed and comprehensive overview of the CRISP-DM process applied in this project, enhancing clarity and professionalism.


## Machine Learning Business Case

### Business Case Assessment

* **Business Objectives:**
    * Conduct a visual analysis to distinguish between healthy cherry leaves and those affected by powdery mildew.
    * Develop a system to classify individual cherry leaf images as either healthy or infected with powdery mildew.

* **Conventional Data Analysis Applicability:**
    * Yes, traditional data analysis techniques can be employed for the visual differentiation research objective.

* **Dashboard or API Endpoint:**
    * The client requires an interactive dashboard for visual analysis and model interaction.

* **Project Success Criteria:**
    * Successful completion of the visual differentiation study, providing clear insights into distinguishing features.
    * Development of a machine learning model capable of accurately predicting the health status of cherry leaves.

* **Epic and User Story Breakdown:**

    1.  **Data Acquisition and Preparation:**
        * Gather and prepare the cherry leaf image dataset.
        * Perform data cleaning and preprocessing.

    2.  **Visual Analysis and Exploration:**
        * Conduct exploratory data analysis (EDA) to understand the dataset.
        * Create visualizations to compare healthy and infected leaves.

    3.  **Model Development and Evaluation:**
        * Develop and train a machine learning model for image classification.
        * Evaluate model performance and optimize parameters.

    4.  **Dashboard Design and Implementation:**
        * Design and develop an interactive dashboard for data visualization and model interaction.

    5.  **Deployment and Release:**
        * Deploy the dashboard and make it accessible to the client.

* **Ethical and Privacy Considerations:**
    * The client provided the data under a non-disclosure agreement (NDA).
    * Data access and sharing will be strictly limited to authorized personnel involved in the project.

* **Model Selection:**
    * The project requires a binary classification model to predict the health status of cherry leaves.

* **Model Inputs and Outputs:**
    * Input: Cherry leaf image.
    * Output: Prediction of whether the cherry leaf is healthy or infected with powdery mildew.

* **Performance Goal:**
    * Achieve a minimum accuracy of 97% in classifying cherry leaf images.

* **Client Benefits:**
    * Enhanced ability to detect and manage powdery mildew in cherry orchards.
    * Improved crop yield and product quality.
    * Reduced reliance on time-consuming and inefficient manual inspection methods.
    * Potential for application to other crops and broader pest management strategies.


    ### ***Actions to Fulfill the Business Case***

> Using the dataset of images provided by the client, build a supervised, binary classification ML model to predict if a leaf is healthy or infected with powdery mildew.

**1. Data Collection:**

* Obtain the image dataset of healthy and powdery mildew-infected cherry leaves provided by the client.

**2. Data Preprocessing:**

* Clean the dataset by removing any non-image files.
* Analyze and standardize image sizes for consistency.
* Split the data into training, validation, and test sets to ensure robust model evaluation.

**3. Feature Extraction:**

* Utilize a Convolutional Neural Network (CNN) to automatically extract relevant features from the images. This will enable the model to learn complex patterns and representations.

**4. Model Selection:**

* Employ a CNN-based machine learning model specifically designed for binary classification tasks. This architecture is well-suited for image data and can effectively distinguish between healthy and infected leaves.

**5. Model Training:**

* Train the selected CNN model using the training dataset.
* Continuously monitor and validate model performance using the validation set during training. This iterative process ensures optimal model development.

**6. Model Evaluation:**

* Evaluate the trained model's performance using various metrics, including accuracy, precision, recall, and F1-score.
* Visualize the model's learning curve to assess its training progress and identify potential issues.

**7. Model Testing:**

* Test the model's generalization capabilities by evaluating its performance on different test datasets. This ensures the model can accurately classify new, unseen images.

**8. Deployment:**

* Deploy the trained and optimized model into a production environment.
* Integrate the model into an application or system where users can:
    * Upload cherry leaf images.
    * Receive predictions on leaf health status.
    * Download comprehensive prediction reports for further analysis and record-keeping.

---

## Model Details

The machine learning model employed in this project is a Convolutional Neural Network (CNN) designed for binary image classification. 

* **Architecture:** The model utilizes a sequential architecture, where layers are arranged in a linear sequence.
* **Layers:** It consists of four convolutional layers for feature extraction and a dense layer with 128 neurons for further processing.
* **Regularization:** Dropout layers and early stopping are implemented to mitigate overfitting and enhance generalization.
* **Output Layer:** The final layer has a single neuron with a sigmoid activation function, producing a probability score for classification.
* **Loss Function:** Binary cross-entropy is used as the loss function, suitable for binary classification problems.
* **Optimizer:** The Adam optimizer is employed for efficient model training.

![Model Summary]()

## Dashboard Design

The interactive dashboard provides a user-friendly interface for exploring the data, visualizing model results, and performing predictions. It comprises the following pages:

**1. Navigation**

A consistent navigation bar ensures easy access to all dashboard pages.

<details>
<summary>Navigation Image</summary>

![Navigation]()

</details>

**2. Project Summary**

This page provides an overview of the project, including:

* **Farmy & Foods Logo:** Company branding.
* **Project Summary:** Concise description of the project's objectives and scope.
* **Project Dataset:** Information about the dataset used, including source and characteristics.
* **Business Requirements:** Clearly defined business objectives.
* **Link to README:** Hyperlink to this README file for detailed documentation.

<details>
<summary>Project Summary Page Image</summary>

![Navigation]()

</details>

**3. Leaf Visualizer**

This page facilitates visual exploration of the cherry leaf dataset:

* **Business Requirement 1:** Restatement of the visual differentiation objective.
* **Interactive Visualizations:**
    * **Checkbox 1: Difference between average and variability image:** Displays the average and variability images for each class (healthy and powdery mildew), along with an explanation of the visualization.
    * **Checkbox 2: Differences between the average image of healthy and powdery mildew infected leaves:** Highlights the differences between the average images of healthy and infected leaves, with detailed explanations.
    * **Checkbox 3: Image montage:** Allows users to create and view image montages for each class, with instructions and explanations provided.

<details>
<summary>Leaf Visualizer Page Image</summary>

![Leaf Visualizer Page]()
![Leaf Visualizer Page]()
![Leaf Visualizer Page]()

</details>

**4. Powdery Mildew Detector**

This page enables users to perform predictions on new cherry leaf images:

* **Business Requirement 2:** Restatement of the image classification objective.
* **Prediction Interface:**
    * **Live prediction info and hyperlink to download images:** Provides information about live prediction status and a link to download sample cherry leaf images.
    * **File uploader:** Allows users to upload cherry leaf images for prediction.
    * **Prediction results:** Displays the uploaded image, the predicted diagnosis (healthy or infected), the probability score, and a bar plot visualization.
    * **Analysis report table:** Presents a table summarizing the prediction results for all uploaded images.
    * **Download report link:** Enables users to download a comprehensive prediction report.

<details>
<summary>Powdery Mildew Detector Page Image</summary>

![Powdery Mildew Detector Page]()
![Powdery Mildew Detector Page]()

</details>

**5. Project Hypotheses and Validation**

This page documents the project's hypotheses and validation results:

* **Powdery Mildew disease detailed explanation:** Provides background information on powdery mildew.
* **Hypotheses:** Clearly states the project's hypotheses.
* **Validation:** Presents the validation results and their implications.
* **Business Requirements:** Reiterates the business objectives.

<details>
<summary>Project Hypotheses and Validation Page Image</summary>

![Project Hypotheses and Validation Page]()

</details>

**6. ML Performance Metrics**

This page presents detailed information about the machine learning model's performance:

* **Average Image size in dataset:** Displays the average image size in the dataset with a visual representation and explanation.
* **Train, Validation and Test Set: Label Frequencies:** Shows the distribution of labels in each dataset split with a plot and explanation.
* **Model History:** Provides insights into the model's training process, including accuracy and loss over time.
* **Generalized Performance on Test Set:** Presents the model's performance on the test set using key metrics.
* **Model accuracy percentage:** Highlights the final model accuracy achieved.

<details>
<summary>ML Performance Metrics Image</summary>

![ML Performance Metrics Page]()
![ML Performance Metrics Page]()

</details>

## Forking and Cloning the Repository

**Forking:**

To create a personal copy of this repository on your GitHub account, follow these steps:

1. Navigate to the repository you wish to fork.
2. Click the "Fork" button located in the upper right corner of the page.

**Cloning:**

To download a copy of the repository to your local machine, follow these steps:

1. Navigate to the repository you wish to clone.
2. Click the green "<> Code" button and copy the provided URL for your preferred cloning method (HTTPS, SSH, or GitHub CLI).
3. Open your terminal or command prompt.
4. Navigate to the directory where you want to store the cloned repository using the `cd` command.
5. Execute the command `git clone` followed by the copied URL.
6. Press Enter to create a local clone of the repository.

## Core Technologies and Libraries

This project leverages a variety of technologies and libraries to achieve its objectives:

**Programming Language:**

* Python

**Libraries:**

* **NumPy:** Used for numerical computing and array manipulation, particularly for converting images into arrays for model input.
* **Pandas:** Employed for data manipulation and analysis, enabling efficient handling of tabular data.
* **Matplotlib:** Utilized for creating static, interactive, and animated visualizations in Python.
* **Seaborn:** Built on top of Matplotlib, Seaborn provides a high-level interface for creating informative and visually appealing statistical graphics.
* **Plotly:** Used for generating interactive charts and visualizations, enhancing data exploration and presentation.
* **Streamlit:** Employed to build the interactive dashboard, providing a user-friendly interface for data visualization and model interaction.
* **Scikit-learn:** A comprehensive machine learning library used for data preprocessing, model selection, training, and evaluation.
* **TensorFlow:** A powerful open-source library for numerical computation and large-scale machine learning, used for building and training the CNN model.
* **Keras:** A high-level API for building and training neural networks, providing a user-friendly interface for TensorFlow.
* **PIL (Pillow):** Used for image processing tasks, such as opening, manipulating, and saving different image file formats.

**Platforms and Tools:**

* **GitHub:** Used for version control, collaboration, and code storage.
* **Gitpod:** Provided the development environment for this project, enabling efficient and reproducible coding.
* **Kaggle:** The source of the cherry leaf image dataset used for model training and evaluation.
* **Render:** Used for deploying the web application, making it accessible to users.

## Credits and Acknowledgements

**Content and Inspiration:**

* **Malaria Detector Project:** Served as a valuable reference and learning resource. [Link to Malaria Detector project]
* **Churnometer Project:** Provided foundational knowledge and code examples for this project. [Link to Churnometer project]
* **SerraKD's README:** Used as a reference for README structure and powdery mildew information. [Link to SerraKD's README]
* **ChatGPT:** Assisted with troubleshooting and provided additional information.

**Acknowledgements:**

* **Code Institute Slack Channel:** A valuable platform for community support and knowledge sharing.

---




