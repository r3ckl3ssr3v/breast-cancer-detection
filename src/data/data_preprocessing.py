import os
import numpy as np
import pandas as pd
import cv2
import pydicom
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import glob

class DataPreprocessor:
    def __init__(self, data_dir, output_dir, img_size=(224, 224)):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir: Directory containing the raw data
            output_dir: Directory to save processed data
            img_size: Target image size for resizing
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.img_size = img_size
        
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'normal'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'benign'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'malignant'), exist_ok=True)
        
    def load_dicom(self, filepath):
        """Load and preprocess a DICOM file."""
        try:
            dicom = pydicom.dcmread(filepath)
            img = dicom.pixel_array
            
            # Convert to float and normalize
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
            img = img.astype(np.uint8)
            
            return img
        except Exception as e:
            print(f"Error loading DICOM file {filepath}: {e}")
            return None
    
    def load_image(self, filepath):
        """Load and preprocess an image file (PNG, JPG, etc.)."""
        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            return img
        except Exception as e:
            print(f"Error loading image file {filepath}: {e}")
            return None
    
    def preprocess_image(self, img):
        """Apply preprocessing steps to an image."""
        if img is None:
            return None
        
        # Resize image
        img_resized = cv2.resize(img, self.img_size)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_resized)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_enhanced / 255.0
        
        return img_normalized
    
    def process_dataset(self, dataset_name, label_file=None):
        """
        Process a specific dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'CBIS-DDSM', 'MIAS')
            label_file: Path to the CSV file containing labels (if applicable)
        """
        print(f"Processing {dataset_name} dataset...")
        
        if dataset_name == 'CBIS-DDSM':
            self._process_cbis_ddsm(label_file)
        elif dataset_name == 'MIAS':
            self._process_mias()
        elif dataset_name == 'INbreast':
            self._process_inbreast(label_file)
        else:
            print(f"Dataset {dataset_name} not supported yet.")
    
    def _process_cbis_ddsm(self, label_file):
        """Process CBIS-DDSM dataset."""
        # Implementation specific to CBIS-DDSM
        # This is a placeholder - you'll need to adapt to the actual data structure
        if label_file and os.path.exists(label_file):
            labels_df = pd.read_csv(label_file)
            
            for _, row in labels_df.iterrows():
                img_path = os.path.join(self.data_dir, row['image_path'])
                if not os.path.exists(img_path):
                    continue
                
                # Load and preprocess image
                if img_path.endswith('.dcm'):
                    img = self.load_dicom(img_path)
                else:
                    img = self.load_image(img_path)
                
                img_processed = self.preprocess_image(img)
                if img_processed is None:
                    continue
                
                # Determine class folder based on pathology
                if row['pathology'] == 'MALIGNANT':
                    class_folder = 'malignant'
                elif row['pathology'] == 'BENIGN':
                    class_folder = 'benign'
                else:
                    class_folder = 'normal'
                
                # Save processed image
                output_path = os.path.join(self.output_dir, class_folder, os.path.basename(img_path).replace('.dcm', '.png'))
                plt.imsave(output_path, img_processed, cmap='gray')
    
    def _process_mias(self):
        """Process MIAS dataset."""
        # Implementation specific to MIAS
        # This is a placeholder - you'll need to adapt to the actual data structure
        info_file = os.path.join(self.data_dir, 'info.txt')
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                img_id = parts[0]
                img_path = os.path.join(self.data_dir, f"{img_id}.pgm")
                if not os.path.exists(img_path):
                    continue
                
                # Load and preprocess image
                img = self.load_image(img_path)
                img_processed = self.preprocess_image(img)
                if img_processed is None:
                    continue
                
                # Determine class folder based on abnormality
                if 'NORM' in parts[1]:
                    class_folder = 'normal'
                elif 'CALC' in parts[1] or 'CIRC' in parts[1] or 'SPIC' in parts[1] or 'MISC' in parts[1]:
                    if 'B' in parts[2]:
                        class_folder = 'benign'
                    elif 'M' in parts[2]:
                        class_folder = 'malignant'
                    else:
                        class_folder = 'normal'
                else:
                    class_folder = 'normal'
                
                # Save processed image
                output_path = os.path.join(self.output_dir, class_folder, f"{img_id}.png")
                plt.imsave(output_path, img_processed, cmap='gray')
    
    def _process_inbreast(self, label_file):
        """Process INbreast dataset."""
        # Implementation specific to INbreast
        # This is a placeholder - you'll need to adapt to the actual data structure
        pass
    
    def create_dataset_splits(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Create train/validation/test splits and save metadata.
        
        Args:
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
        """
        # Get all processed images
        normal_imgs = glob.glob(os.path.join(self.output_dir, 'normal', '*.png'))
        benign_imgs = glob.glob(os.path.join(self.output_dir, 'benign', '*.png'))
        malignant_imgs = glob.glob(os.path.join(self.output_dir, 'malignant', '*.png'))
        
        # Create dataframe with image paths and labels
        data = []
        for img_path in normal_imgs:
            data.append({'image_path': img_path, 'label': 0})  # 0 for normal
        for img_path in benign_imgs:
            data.append({'image_path': img_path, 'label': 1})  # 1 for benign
        for img_path in malignant_imgs:
            data.append({'image_path': img_path, 'label': 2})  # 2 for malignant
        
        df = pd.DataFrame(data)
        
        # Split into train and test
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)
        
        # Split train into train and validation
        train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df['label'], random_state=random_state)
        
        # Save splits to CSV
        train_df.to_csv(os.path.join(self.output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.output_dir, 'test.csv'), index=False)
        
        print(f"Dataset splits created: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
        
        # Print class distribution
        print("\nClass distribution:")
        print("Training set:")
        print(train_df['label'].value_counts())
        print("\nValidation set:")
        print(val_df['label'].value_counts())
        print("\nTest set:")
        print(test_df['label'].value_counts())

# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        data_dir='c:/SDP_Project_P3/data/raw',
        output_dir='c:/SDP_Project_P3/data/processed',
        img_size=(224, 224)
    )
    
    # Process datasets
    # preprocessor.process_dataset('CBIS-DDSM', label_file='c:/SDP_Project_P3/data/raw/CBIS-DDSM/metadata.csv')
    # preprocessor.process_dataset('MIAS')
    
    # Create dataset splits
    preprocessor.create_dataset_splits()