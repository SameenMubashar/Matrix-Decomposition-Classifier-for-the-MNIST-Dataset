# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_preprocess_data():
    """
    Loads the MNIST dataset, flattens the images, and normalizes pixel values.
    """
    print("Step 1: Loading and preprocessing data...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    X_train_norm = X_train_flat.astype('float32') / 255.0
    X_test_norm = X_test_flat.astype('float32') / 255.0
    
    print("Data loading and preprocessing complete.\n")
    return X_train_norm, y_train, X_test_norm, y_test, X_test

def perform_pca(X_train, X_test, n_components=100):
    """
    Applies PCA to reduce the dimensionality of the data.
    """
    print(f"Step 2: Applying PCA to reduce dimensions to {n_components} components...")
    pca = PCA(n_components=n_components)
    
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Original data shape: {X_train.shape}")
    print(f"Data shape after PCA: {X_train_pca.shape}\n")
    return X_train_pca, X_test_pca, pca

def train_classifier(X_train_pca, y_train, n_neighbors=5):
    """
    Trains a k-Nearest Neighbors classifier on the PCA-reduced data.
    """
    print(f"Step 3: Training the k-NN classifier with k={n_neighbors}...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_pca, y_train)
    
    print("Classifier training complete.\n")
    return knn

def evaluate_model(model, X_test_pca, y_test):
    """
    Evaluates the trained model and prints performance metrics.
    """
    print("Step 4: Evaluating the model on the test set...")
    y_pred = model.predict(X_test_pca)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def visualize_predictions(model, pca, X_test_original, y_test, num_images=10):
    """
    Selects random images from the test set and visualizes the model's predictions.
    """
    print("\nStep 5: Visualizing model predictions on unseen images...")
    
    random_indices = np.random.choice(len(X_test_original), num_images, replace=False)
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i, idx in enumerate(random_indices):
        img = X_test_original[idx]
        true_label = y_test[idx]
        
        img_flat_norm = img.reshape(1, -1).astype('float32') / 255.0
        img_pca = pca.transform(img_flat_norm)
        
        prediction = model.predict(img_pca)[0]
        
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        title_color = 'g' if prediction == true_label else 'r'
        ax.set_title(f"Pred: {prediction}\nTrue: {true_label}", color=title_color)
        
    plt.suptitle("Model Predictions on Unseen Images")
    plt.show()

def main():
    """
    Main function to orchestrate the digit recognition workflow.
    """
    N_COMPONENTS = 100
    K_NEIGHBORS = 5
    
    # 1. Load and preprocess data
    X_train_norm, y_train, X_test_norm, y_test, X_test_original = load_and_preprocess_data()
    
    # 2. Apply PCA
    X_train_pca, X_test_pca, pca_model = perform_pca(X_train_norm, X_test_norm, n_components=N_COMPONENTS)
    
    # 3. Train the k-NN classifier
    knn_model = train_classifier(X_train_pca, y_train, n_neighbors=K_NEIGHBORS)
    
    # 4. Evaluate the model
    evaluate_model(knn_model, X_test_pca, y_test)
    
    # 5. Visualize predictions
    visualize_predictions(knn_model, pca_model, X_test_original, y_test)

if __name__ == "__main__":
    main()