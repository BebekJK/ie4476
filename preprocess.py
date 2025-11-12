import os
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle


class ImagePreprocessor:
    """
    Preprocessor for AT&T Faces dataset with exact preprocessing steps:
    1. Convert to grayscale
    2. Normalize pixel values to [0, 1]
    3. Flatten to 1D vectors (10304 dimensions from 112x92 pixels)
    4. Split into 70% training and 30% testing (stratified)
    5. Center data by subtracting mean training image
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the preprocessor.
        
        Args:
            dataset_path: Path to the dataset folder
        """
        self.dataset_path = Path(dataset_path)
        self.images = []
        self.labels = []
        self.class_names = []
        self.mean_image = None
        
    def load_dataset(self):
        """
        Step 1 & 2: Load all images, convert to grayscale, and normalize to [0, 1].
        """
        print("Step 1 & 2: Loading dataset, converting to grayscale, and normalizing...")
        
        # Get sorted list of class folders (s1, s2, ..., s40)
        class_folders = sorted([d for d in self.dataset_path.iterdir() 
                               if d.is_dir()], 
                              key=lambda x: int(x.name[1:]))
        
        self.class_names = [folder.name for folder in class_folders]
        
        for class_idx, class_folder in enumerate(class_folders):
            print(f"  Processing class {class_folder.name} ({class_idx + 1}/{len(class_folders)})")
            
            # Get all PGM files in the folder, sorted by number
            image_files = sorted(class_folder.glob('*.pgm'), 
                               key=lambda x: int(x.stem))
            
            for img_file in image_files:
                try:
                    # Load image
                    img = Image.open(img_file)
                    
                    # Convert to grayscale (images are already grayscale, but ensure mode 'L')
                    if img.mode != 'L':
                        img = img.convert('L')
                    
                    # Convert to numpy array
                    img_array = np.array(img, dtype=np.float32)
                    
                    # Normalize pixel values to [0, 1]
                    img_array = img_array / 255.0
                    
                    self.images.append(img_array)
                    self.labels.append(class_idx)
                    
                except Exception as e:
                    print(f"    Error loading {img_file}: {e}")
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"\n  Dataset loaded successfully!")
        print(f"  Total images: {len(self.images)}")
        print(f"  Number of classes: {len(self.class_names)}")
        print(f"  Image shape: {self.images[0].shape} (Height x Width)")
        print(f"  Pixel value range: [{self.images.min():.3f}, {self.images.max():.3f}]")
        
        return self.images, self.labels
    
    def flatten_images(self, images):
        """
        Step 3: Flatten 2D images into 1D vectors (112 x 92 = 10304 dimensions).
        
        Args:
            images: Array of 2D images with shape (n_samples, height, width)
            
        Returns:
            Flattened images with shape (n_samples, 10304)
        """
        n_samples = images.shape[0]
        flattened = images.reshape(n_samples, -1)
        return flattened
    
    def split_dataset(self, test_size=0.3, random_state=42):
        """
        Step 4: Split dataset into 70% training and 30% testing sets (stratified).
        
        Args:
            test_size: Proportion of dataset for testing (default: 0.3 for 30%)
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test (all flattened)
        """
        if len(self.images) == 0:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print(f"\nStep 3: Flattening images to 1D vectors...")
        # Flatten images before splitting
        X_flat = self.flatten_images(self.images)
        print(f"  Flattened shape: {X_flat.shape} ({X_flat.shape[1]} dimensions)")
        
        print(f"\nStep 4: Splitting dataset into 70% training and 30% testing...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels
        )
        
        print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X_flat)*100:.1f}%)")
        print(f"  Testing samples: {len(X_test)} ({len(X_test)/len(X_flat)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def center_data(self, X_train, X_test):
        """
        Step 5: Center the data by subtracting the mean training image from all samples.
        
        Args:
            X_train: Training data (flattened)
            X_test: Testing data (flattened)
            
        Returns:
            X_train_centered, X_test_centered, mean_image
        """
        print(f"\nStep 5: Centering data by subtracting mean training image...")
        
        # Calculate mean training image
        self.mean_image = np.mean(X_train, axis=0)
        
        # Center training data
        X_train_centered = X_train - self.mean_image
        
        # Center testing data using the same mean
        X_test_centered = X_test - self.mean_image
        
        print(f"  Mean image shape: {self.mean_image.shape}")
        print(f"  Centered training data - Mean: {X_train_centered.mean():.6f}, Std: {X_train_centered.std():.6f}")
        print(f"  Centered testing data - Mean: {X_test_centered.mean():.6f}, Std: {X_test_centered.std():.6f}")
        
        return X_train_centered, X_test_centered, self.mean_image

def main():
    # Set paths
    dataset_path = "dataset"
    output_path = "processed_data"
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(dataset_path=dataset_path)
    
    # Step 1 & 2: Load dataset, convert to grayscale, and normalize
    X, y = preprocessor.load_dataset()
    
    # Step 3 & 4: Flatten images and split into train/test (70/30)
    X_train, X_test, y_train, y_test = preprocessor.split_dataset(
        test_size=0.3,      # 30% for testing, 70% for training
        random_state=42
    )
    
    # Step 5: Center data by subtracting mean training image
    X_train_centered, X_test_centered, mean_image = preprocessor.center_data(
        X_train, X_test
    )
    
    # Save processed data
    preprocessor.save_processed_data(
        output_path, 
        X_train_centered, X_test_centered, 
        y_train, y_test,
        mean_image
    )
    
    # Print summary
    preprocessor.print_summary(X_train_centered, X_test_centered, y_train, y_test)
    
    # Additional information
    print(f"\nTo load the processed data:")
    print(f"```python")
    print(f"import pickle")
    print(f"with open('processed_data/processed_data.pkl', 'rb') as f:")
    print(f"    data = pickle.load(f)")
    print(f"X_train = data['X_train']  # Centered, flattened training data")
    print(f"X_test = data['X_test']    # Centered, flattened testing data")
    print(f"y_train = data['y_train']  # Training labels (0-39)")
    print(f"y_test = data['y_test']    # Testing labels (0-39)")
    print(f"mean_image = data['mean_image']  # Mean training image")
    print(f"```\n")


if __name__ == "__main__":
    main()
