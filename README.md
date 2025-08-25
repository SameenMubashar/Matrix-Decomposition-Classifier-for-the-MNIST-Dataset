## Project Overview

This repository contains a **Matrix Decomposition Classifier for the MNIST Dataset**. It is a technical project demonstrating hand-written digit recognition using classical machine learning techniques. In a landscape often dominated by deep learning, this project showcases the elegance and effectiveness of **Principal Component Analysis (PCA)**—a form of matrix decomposition—for feature extraction, coupled with a **k-Nearest Neighbors (k-NN)** classifier for accurate digit identification.

This project serves as a compelling example of fundamental computer vision principles, offering a transparent and interpretable alternative to complex neural networks for this foundational task.

-----

## Key Features

  * **Robust Digit Recognition:** Achieves an impressive **97.27% accuracy** on the MNIST dataset.
  * **Principal Component Analysis (PCA):** Leverages PCA to automatically extract the most significant features ("eigen-digits") from the image data, significantly reducing dimensionality while preserving crucial information.
  * **k-Nearest Neighbors (k-NN) Classifier:** Employs a simple yet effective k-NN algorithm to classify the PCA-transformed digit features.
  * **Clear and Interpretable:** Provides a clear understanding of the feature extraction and classification process, unlike the "black box" nature of some deep learning models.
  * **Comprehensive Evaluation:** Includes detailed performance metrics such as accuracy, precision, recall, F1-score, and a confusion matrix.
  * **Well-Documented Code:** The Python scripts are clearly commented and easy to follow.

-----

## Visual Insights

### Explained Variance by PCA Components

This plot illustrates the amount of variance captured by each principal component. We can observe that a relatively small number of components can explain a significant portion of the data's variance, enabling effective dimensionality reduction.
![Explained Variance Plot](https://github.com/SameenMubashar/Matrix-Decomposition-Classifier-for-the-MNIST-Dataset/blob/main/assets/explained_variance.png?raw=true)

### Confusion Matrix

The confusion matrix provides a detailed view of the model's performance across different digit classes, highlighting where misclassifications occur.
![Confusion Matrix](https://github.com/SameenMubashar/Matrix-Decomposition-Classifier-for-the-MNIST-Dataset/blob/main/assets/confusion_matrix.png?raw=true)

### Model Predictions on Unseen Images

These examples showcase the model's ability to correctly classify handwritten digits it has never encountered before.
![Model Predictions](https://github.com/SameenMubashar/Matrix-Decomposition-Classifier-for-the-MNIST-Dataset/blob/main/assets/predictions.png?raw=true)

-----

## Getting Started

Follow these simple steps to run the project on your local machine.

### Prerequisites

  * **Python 3.7+**
  * **pip** (Python package installer)
  * **Jupyter Notebook** (for converting the notebook)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SameenMubashar/Matrix-Decomposition-Classifier-for-the-MNIST-Dataset.git
    cd Matrix-Decomposition-Classifier-for-the-MNIST-Dataset
    ```

2.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

The entire project workflow, from data loading to model evaluation, is contained within the digit_recognizer.ipynb Jupyter Notebook.

#### Launch Jupyter Notebook
 Open your terminal or command prompt, navigate to the project's root directory, and run the following command:

```bash
jupyter notebook
```

#### Launch Jupyter Notebook
A new tab will open in your web browser. From the file list, click on 
```bash
digit_recognizer.ipynb
```

#### Execute the Code
Once the notebook is open, you can run all the code cells sequentially by selecting Cell > Run All from the top menu bar.

The script will:

  * Load the MNIST dataset.
  * Preprocess the image data.
  * Apply PCA for dimensionality reduction.
  * Train a k-NN classifier.
  * Evaluate the model and display the results and plots.

-----

## Project Structure

```
Matrix-Decomposition-Classifier-for-the-MNIST-Dataset/
├── digit_recognizer.ipynb      # Jupyter Notebook with the original analysis
├── requirements.txt            # List of required Python libraries
├── assets/                     # Directory for images used in the README
│   ├── explained_variance.png
│   ├── confusion_matrix.png
│   └── predictions.png
├── README.md                   # This file
└── LICENSE
```

-----

## Concepts Covered

This project delves into fundamental concepts in Computer Vision and Machine Learning:

  * **Image Preprocessing:** Flattening and normalization of image data.
  * **Matrix Decomposition:** Using PCA (via Eigenvalue Decomposition) as a technique for matrix factorization.
  * **Dimensionality Reduction:** Utilizing PCA to extract salient features and reduce data complexity.
  * **Classical Machine Learning:** Implementing and evaluating a k-NN classification algorithm.
  * **Evaluation Metrics:** Interpreting accuracy, precision, recall, F1-score, and confusion matrices.

-----
