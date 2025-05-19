import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import glob

def preprocess_mini_ddsm(raw_data_dir, processed_data_dir, image_size=(256, 256)):
    """
    Preprocess the mini-DDSM dataset.
    
    Args:
        raw_data_dir: Path to the raw mini-DDSM data directory
        processed_data_dir: Path to save the processed data
        image_size: Target size for the images (height, width)
    """
    print("Starting mini-DDSM dataset preprocessing...")
    
    # Create directories if they don't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, 'benign'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, 'malignant'), exist_ok=True)
    
    # Since the CSV parsing is failing, let's directly process the image directories
    # Check for Normal, Benign, and Cancer/Malignant directories
    normal_dir = os.path.join(raw_data_dir, 'Normal')
    benign_dir = os.path.join(raw_data_dir, 'Benign')
    cancer_dir = os.path.join(raw_data_dir, 'Cancer')
    
    # Alternative directory names
    if not os.path.exists(cancer_dir):
        cancer_dir = os.path.join(raw_data_dir, 'Malignant')
    
    # Process images from each directory
    processed_count = 0
    
    # Process normal images
    if os.path.exists(normal_dir):
        print(f"Processing normal images from {normal_dir}")
        processed_count += process_images_from_dir(normal_dir, 
                                                 os.path.join(processed_data_dir, 'normal'),
                                                 image_size)
    else:
        print(f"Warning: Normal directory not found at {normal_dir}")
    
    # Process benign images
    if os.path.exists(benign_dir):
        print(f"Processing benign images from {benign_dir}")
        processed_count += process_images_from_dir(benign_dir, 
                                                 os.path.join(processed_data_dir, 'benign'),
                                                 image_size)
    else:
        print(f"Warning: Benign directory not found at {benign_dir}")
    
    # Process cancer/malignant images
    if os.path.exists(cancer_dir):
        print(f"Processing malignant images from {cancer_dir}")
        processed_count += process_images_from_dir(cancer_dir, 
                                                 os.path.join(processed_data_dir, 'malignant'),
                                                 image_size)
    else:
        print(f"Warning: Cancer/Malignant directory not found at {cancer_dir}")
    
    print(f"Successfully processed {processed_count} images")
    
    # Create train/val/test split files
    create_data_splits(processed_data_dir)
    
    print("mini-DDSM preprocessing completed!")

def process_images_from_dir(input_dir, output_dir, image_size):
    """
    Process all images from a directory.
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save processed images
        image_size: Target size for the images
    
    Returns:
        Number of successfully processed images
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.pgm']:
        image_files.extend(glob.glob(os.path.join(input_dir, f'**/*{ext}'), recursive=True))
        image_files.extend(glob.glob(os.path.join(input_dir, f'**/*{ext.upper()}'), recursive=True))
    
    print(f"Found {len(image_files)} image files in {input_dir}")
    
    # Process each image
    processed_count = 0
    for image_path in tqdm(image_files, desc=f"Processing images from {os.path.basename(input_dir)}"):
        try:
            # Read image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not read image: {image_path}")
                continue
            
            # Resize image
            image = cv2.resize(image, image_size)
            
            # Generate output filename
            # Use the original filename but ensure it has a .png extension
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save processed image
            cv2.imwrite(output_path, image)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
    
    return processed_count

def create_data_splits(processed_data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/validation/test splits.
    
    Args:
        processed_data_dir: Directory with processed images
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
    """
    classes = ['normal', 'benign', 'malignant']
    
    train_files = []
    val_files = []
    test_files = []
    
    for class_name in classes:
        class_dir = os.path.join(processed_data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
            
        files = [os.path.join(class_name, f) for f in os.listdir(class_dir) if f.endswith('.png')]
        
        if not files:
            print(f"Warning: No files found in {class_dir}")
            continue
            
        np.random.shuffle(files)
        
        n_files = len(files)
        n_train = int(train_ratio * n_files)
        n_val = int(val_ratio * n_files)
        
        train_files.extend([(f, class_name) for f in files[:n_train]])
        val_files.extend([(f, class_name) for f in files[n_train:n_train+n_val]])
        test_files.extend([(f, class_name) for f in files[n_train+n_val:]])
    
    # Save splits to CSV
    splits_dir = os.path.join(processed_data_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    pd.DataFrame(train_files, columns=['filepath', 'class']).to_csv(
        os.path.join(splits_dir, 'train.csv'), index=False)
    pd.DataFrame(val_files, columns=['filepath', 'class']).to_csv(
        os.path.join(splits_dir, 'val.csv'), index=False)
    pd.DataFrame(test_files, columns=['filepath', 'class']).to_csv(
        os.path.join(splits_dir, 'test.csv'), index=False)
    
    print(f"Created data splits: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")