import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Modify the mask creation part to handle empty values
def preprocess_mias(raw_data_dir, processed_data_dir, image_size=(256, 256)):
    """
    Preprocess the MIAS dataset.
    
    Args:
        raw_data_dir: Path to the raw MIAS data directory
        processed_data_dir: Path to save the processed data
        image_size: Target size for the images (height, width)
    """
    print("Starting MIAS dataset preprocessing...")
    
    # Create directories if they don't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, 'benign'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, 'malignant'), exist_ok=True)
    
    # Load the CSV file
    csv_path = os.path.join(raw_data_dir, 'mias_info.csv')
    
    try:
        # Try to read the CSV file with different separators
        try:
            df = pd.read_csv(csv_path, sep='|', skiprows=1, header=None)
            print(f"Successfully read CSV with '|' separator. Shape: {df.shape}")
        except Exception as e1:
            print(f"Failed with '|' separator: {str(e1)}")
            try:
                df = pd.read_csv(csv_path, sep=',')
                print(f"Successfully read CSV with ',' separator. Shape: {df.shape}")
            except Exception as e2:
                print(f"Failed with ',' separator: {str(e2)}")
                # As a last resort, try to infer the separator
                df = pd.read_csv(csv_path, sep=None, engine='python')
                print(f"Successfully read CSV with inferred separator. Shape: {df.shape}")
        
        # If we have a single column that contains all data, try to split it
        if len(df.columns) == 1:
            col_name = df.columns[0]
            if '|' in df[col_name].iloc[0]:
                # Split the single column by '|'
                print("Detected '|' in the data, splitting manually...")
                df = df[col_name].str.split('|', expand=True)
                print(f"After splitting: {df.shape}")
            elif ',' in df[col_name].iloc[0]:
                # Split the single column by ','
                print("Detected ',' in the data, splitting manually...")
                df = df[col_name].str.split(',', expand=True)
                print(f"After splitting: {df.shape}")
        
        # Assign column names
        column_names = ['REFNUM', 'BG', 'CLASS', 'SEVERITY', 'X', 'Y', 'RADIUS']
        if df.shape[1] == len(column_names):
            df.columns = column_names
        else:
            print(f"Warning: Expected {len(column_names)} columns but found {df.shape[1]}. Using default column names.")
            # Use default column names for available columns
            for i, name in enumerate(column_names):
                if i < df.shape[1]:
                    df = df.rename(columns={i: name})
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        print("Attempting to read the file line by line...")
        # Manual parsing as a last resort
        lines = []
        with open(csv_path, 'r') as f:
            for line in f:
                if '|' in line:
                    parts = line.strip().split('|')
                    if len(parts) > 1:
                        parts = parts[1].split(',')
                        lines.append(parts)
                else:
                    parts = line.strip().split(',')
                    lines.append(parts)
        
        # Create DataFrame from parsed lines
        df = pd.DataFrame(lines[1:], columns=column_names)
        print(f"Manually parsed CSV. Shape: {df.shape}")
    
    # Clean the dataframe
    df = df.dropna(subset=['REFNUM'])
    df = df[df['REFNUM'] != '']
    
    # Map severity to classes
    def map_severity_to_class(row):
        if pd.isna(row['SEVERITY']) or row['SEVERITY'] == '':
            if row['CLASS'] == 'NORM':
                return 'normal'
            else:
                return None
        elif row['SEVERITY'] == 'B':
            return 'benign'
        elif row['SEVERITY'] == 'M':
            return 'malignant'
        else:
            return None
    
    df['class'] = df.apply(map_severity_to_class, axis=1)
    df = df.dropna(subset=['class'])
    
    # Process each image
    images_dir = os.path.join(raw_data_dir, 'images')
    
    # Check if the directory exists, if not try to find the correct directory
    if not os.path.exists(images_dir):
        print(f"Warning: Directory {images_dir} not found. Searching for image directories...")
        # Look for possible image directories
        for dir_name in os.listdir(raw_data_dir):
            if os.path.isdir(os.path.join(raw_data_dir, dir_name)) and dir_name != 'splits':
                potential_dir = os.path.join(raw_data_dir, dir_name)
                # Check if this directory contains image files
                image_files = [f for f in os.listdir(potential_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm'))]
                if image_files:
                    images_dir = potential_dir
                    print(f"Found image directory: {images_dir}")
                    break
    
    # If we still don't have an images directory, use the raw data directory itself
    if not os.path.exists(images_dir):
        images_dir = raw_data_dir
        print(f"Using raw data directory for images: {images_dir}")
    
    processed_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        try:
            # Get image path - try different extensions
            ref_num = str(row['REFNUM']).strip()
            
            # Try different file extensions
            extensions = ['.pgm', '.png', '.jpg', '.jpeg']
            image_path = None
            
            for ext in extensions:
                potential_path = os.path.join(images_dir, ref_num + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            # If not found, try searching for the file in the directory
            if image_path is None:
                found = False
                for img_file in os.listdir(images_dir):
                    if ref_num in img_file:
                        image_path = os.path.join(images_dir, img_file)
                        found = True
                        break
                
                if not found:
                    print(f"Warning: Image file not found for {ref_num}")
                    continue
            
            # Read image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not read image: {image_path}")
                continue
            
            # Resize image
            image = cv2.resize(image, image_size)
            
            # Save processed image
            class_label = row['class']
            output_filename = f"{ref_num}.png"
            output_path = os.path.join(processed_data_dir, class_label, output_filename)
            
            cv2.imwrite(output_path, image)
            processed_count += 1
            
            # If coordinates and radius are available, create a mask image
            # Fix this part to properly check for empty values
            has_x = not pd.isna(row['X']) and row['X'] != ''
            has_y = not pd.isna(row['Y']) and row['Y'] != ''
            has_radius = not pd.isna(row['RADIUS']) and row['RADIUS'] != ''
            
            if has_x and has_y and has_radius:
                try:
                    # Create a mask image
                    mask = np.zeros(image.shape, dtype=np.uint8)
                    
                    # Get coordinates and radius
                    x = float(row['X'])
                    y = float(row['Y'])
                    radius = float(row['RADIUS'])
                    
                    # Scale coordinates to match resized image
                    orig_height, orig_width = image.shape
                    scale_x = image_size[0] / orig_width
                    scale_y = image_size[1] / orig_height
                    
                    x_scaled = int(x * scale_x)
                    y_scaled = int(y * scale_y)
                    radius_scaled = int(radius * min(scale_x, scale_y))
                    
                    # Draw circle on mask
                    cv2.circle(mask, (x_scaled, y_scaled), radius_scaled, 255, -1)
                    
                    # Save mask
                    mask_dir = os.path.join(processed_data_dir, 'masks')
                    os.makedirs(mask_dir, exist_ok=True)
                    mask_path = os.path.join(mask_dir, f"{ref_num}_mask.png")
                    cv2.imwrite(mask_path, mask)
                except Exception as e:
                    print(f"Error creating mask for {ref_num}: {str(e)}")
            
        except Exception as e:
            print(f"Error processing image {ref_num}: {str(e)}")
    
    print(f"Successfully processed {processed_count} images")
    
    # Create train/val/test split files
    create_data_splits(processed_data_dir)
    
    print("MIAS preprocessing completed!")

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
            continue
            
        files = [os.path.join(class_name, f) for f in os.listdir(class_dir) if f.endswith('.png')]
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