"""
Example: How to use the preprocessed AT&T Faces dataset for machine learning.

This script demonstrates loading and using the preprocessed data for various tasks.
"""
import pickle
import numpy as np


def load_data():
    """Load the preprocessed dataset."""
    print("Loading preprocessed data...")
    with open('processed_data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    mean_image = data['mean_image']
    
    print(f"✓ Loaded training data: {X_train.shape}")
    print(f"✓ Loaded testing data: {X_test.shape}")
    print(f"✓ Loaded labels: train={y_train.shape}, test={y_test.shape}")
    print(f"✓ Loaded mean image: {mean_image.shape}\n")
    
    return X_train, X_test, y_train, y_test, mean_image, data


def example_basic_statistics(X_train, X_test, y_train, y_test):
    """Display basic statistics about the data."""
    print("="*60)
    print("BASIC STATISTICS")
    print("="*60)
    
    print(f"Training set:")
    print(f"  Shape: {X_train.shape}")
    print(f"  Mean: {X_train.mean():.6f} (centered, so ~0)")
    print(f"  Std: {X_train.std():.6f}")
    print(f"  Min: {X_train.min():.3f}")
    print(f"  Max: {X_train.max():.3f}")
    
    print(f"\nTesting set:")
    print(f"  Shape: {X_test.shape}")
    print(f"  Mean: {X_test.mean():.6f}")
    print(f"  Std: {X_test.std():.6f}")
    print(f"  Min: {X_test.min():.3f}")
    print(f"  Max: {X_test.max():.3f}")
    
    print(f"\nLabels:")
    print(f"  Classes: {len(np.unique(y_train))}")
    print(f"  Range: {y_train.min()} to {y_train.max()}")
    print()


def example_reconstruct_image(X_train, mean_image):
    """Demonstrate how to reconstruct an image."""
    print("="*60)
    print("IMAGE RECONSTRUCTION EXAMPLE")
    print("="*60)
    
    # Take first training sample (centered)
    centered_sample = X_train[0]
    print(f"Centered sample shape: {centered_sample.shape}")
    print(f"Centered sample range: [{centered_sample.min():.3f}, {centered_sample.max():.3f}]")
    
    # Add back the mean to reconstruct original normalized image
    reconstructed = centered_sample + mean_image
    print(f"\nReconstructed sample range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # Reshape to 2D image
    image_2d = reconstructed.reshape(112, 92)
    print(f"Reshaped to 2D: {image_2d.shape}")
    
    print("\nTo visualize this image:")
    print("```python")
    print("import matplotlib.pyplot as plt")
    print("plt.imshow(image_2d, cmap='gray')")
    print("plt.title('Reconstructed Face')")
    print("plt.axis('off')")
    print("plt.show()")
    print("```\n")


def example_compute_pca_dimensions(X_train):
    """Show how many dimensions capture most variance (useful for PCA)."""
    print("="*60)
    print("DIMENSIONALITY ANALYSIS")
    print("="*60)
    
    print(f"Original dimensions: {X_train.shape[1]}")
    print(f"Number of samples: {X_train.shape[0]}")
    print(f"\nMaximum PCA components possible: {min(X_train.shape)} (min of samples and features)")
    print(f"\nNote: With 280 training samples, you can extract at most 280 principal components.")
    print(f"Typical choices: 50-150 components capture most facial variations.\n")


def example_ready_for_ml(X_train, X_test, y_train, y_test):
    """Show that data is ready for machine learning."""
    print("="*60)
    print("READY FOR MACHINE LEARNING")
    print("="*60)
    
    print("The data is now ready to use with:")
    print("\n1. Principal Component Analysis (PCA)")
    print("   from sklearn.decomposition import PCA")
    print("   pca = PCA(n_components=150)")
    print("   X_train_pca = pca.fit_transform(X_train)")
    print("   X_test_pca = pca.transform(X_test)")
    
    print("\n2. Linear Discriminant Analysis (LDA)")
    print("   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis")
    print("   lda = LinearDiscriminantAnalysis(n_components=39)")  # max 39 for 40 classes
    print("   X_train_lda = lda.fit_transform(X_train, y_train)")
    print("   X_test_lda = lda.transform(X_test)")
    
    print("\n3. Support Vector Machine (SVM)")
    print("   from sklearn.svm import SVC")
    print("   svm = SVC(kernel='rbf', C=1.0, gamma='scale')")
    print("   svm.fit(X_train, y_train)")
    print("   accuracy = svm.score(X_test, y_test)")
    
    print("\n4. k-Nearest Neighbors (k-NN)")
    print("   from sklearn.neighbors import KNeighborsClassifier")
    print("   knn = KNeighborsClassifier(n_neighbors=5)")
    print("   knn.fit(X_train, y_train)")
    print("   accuracy = knn.score(X_test, y_test)")
    
    print("\n5. Neural Network")
    print("   from sklearn.neural_network import MLPClassifier")
    print("   mlp = MLPClassifier(hidden_layer_sizes=(256, 128))")
    print("   mlp.fit(X_train, y_train)")
    print("   accuracy = mlp.score(X_test, y_test)")
    
    print("\nAll data is:")
    print("  ✓ Properly shaped (n_samples, n_features)")
    print("  ✓ Normalized to [0, 1] (before centering)")
    print("  ✓ Centered (mean = 0)")
    print("  ✓ Split into train/test")
    print("  ✓ Stratified (balanced classes)")
    print()


def main():
    """Main function demonstrating data usage."""
    print("\n" + "="*60)
    print("PREPROCESSED AT&T FACES DATASET - USAGE EXAMPLES")
    print("="*60 + "\n")
    
    # Load data
    X_train, X_test, y_train, y_test, mean_image, data = load_data()
    
    # Run examples
    example_basic_statistics(X_train, X_test, y_train, y_test)
    example_reconstruct_image(X_train, mean_image)
    example_compute_pca_dimensions(X_train)
    example_ready_for_ml(X_train, X_test, y_train, y_test)
    
    print("="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
