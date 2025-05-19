import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

def preprocess_inbreast(raw_data_dir, processed_data_dir, image_size=(256, 256)):
    """
    Preprocess the INbreast dataset.
    
    Args:
        raw_data_dir: Path to the raw INbreast data directory
        processed_data_dir: Path to save the processed data
        image_size: Target size for the images (height, width)
    """
    print("Starting INbreast dataset preprocessing...")
    
    # Create directories if they don't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, 'benign'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, 'malignant'), exist_ok=True)
    
    # Load the CSV file
    csv_path = os.path.join(raw_data_dir, 'INbreast.csv')
    
    # Try different approaches to read the CSV file
    try:
        # First, try to read the file to inspect its content
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()
            print(f"First line of CSV: {first_line}")
            
        # Try reading with different separators or engines
        try:
            df = pd.read_csv(csv_path, sep='|', engine='python')
            print(f"Successfully read CSV with '|' separator. Columns: {df.columns.tolist()}")
        except Exception as e1:
            print(f"Failed with '|' separator: {str(e1)}")
            try:
                df = pd.read_csv(csv_path, sep=',')
                print(f"Successfully read CSV with ',' separator. Columns: {df.columns.tolist()}")
            except Exception as e2:
                print(f"Failed with ',' separator: {str(e2)}")
                # As a last resort, try to infer the separator
                df = pd.read_csv(csv_path, sep=None, engine='python')
                print(f"Successfully read CSV with inferred separator. Columns: {df.columns.tolist()}")
        
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
    
        # Define column names based on the first row or use default names
        if df.shape[1] == 17:  # If we have the expected number of columns
            column_names = [
                'Patient ID', 'Patient age', 'Laterality', 'View', 'Acquisition date', 
                'File Name', 'ACR', 'Bi-Rads', 'Mass', 'Micros', 'Distortion', 
                'Asymmetry', 'Findings Notes', 'Other Notes', 'Lesion Annotation Status', 
                'Pectoral Muscle Annotation', 'Other Annotations'
            ]
            df.columns = column_names
        else:
            print(f"Warning: Expected 17 columns but found {df.shape[1]}. Using default column names.")
            # Use the first row as header if it looks like a header
            if 'Patient' in str(df.iloc[0,0]) or 'ID' in str(df.iloc[0,0]):
                df.columns = df.iloc[0]
                df = df.drop(0)
            # Ensure 'File Name' and 'Bi-Rads' columns exist for further processing
            if 'File Name' not in df.columns and df.shape[1] >= 6:
                df = df.rename(columns={5: 'File Name'})
            if 'Bi-Rads' not in df.columns and df.shape[1] >= 8:
                df = df.rename(columns={7: 'Bi-Rads'})
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        print("Attempting to read the file line by line...")
        # Manual parsing as a last resort
        lines = []
        with open(csv_path, 'r') as f:
            for line in f:
                lines.append(line.strip().split('|'))
        
        # Create DataFrame from parsed lines
        df = pd.DataFrame(lines[1:], columns=lines[0])
        print(f"Manually parsed CSV. Shape: {df.shape}")
    
    # Clean the dataframe
    df = df.dropna(subset=['File Name', 'Bi-Rads'])
    df = df[df['File Name'] != '']
    
    # Map BI-RADS to classes
    # 1-2: Normal/Benign
    # 3-4: Benign with higher suspicion
    # 4-5: Suspicious/Highly suspicious
    # 6: Known malignancy
    
    def map_birads_to_class(birads):
        try:
            birads = str(birads).strip()
            if birads == '1' or birads == '2':
                return 'normal'
            elif birads == '3' or birads in ['4a', '4A']:
                return 'benign'
            elif birads in ['4b', '4B', '4c', '4C', '5']:
                return 'malignant'
            elif birads == '6':
                return 'malignant'
            else:
                return None
        except:
            return None
    
    df['class'] = df['Bi-Rads'].apply(map_birads_to_class)
    df = df.dropna(subset=['class'])
    
    # Process each image - MODIFIED to use PNG files instead of DICOM
    images_dir = os.path.join(raw_data_dir, 'AllImages')  # Changed from 'AllDICOMs' to 'AllImages'
    
    # Check if the directory exists, if not try to find the correct directory
    if not os.path.exists(images_dir):
        print(f"Warning: Directory {images_dir} not found. Searching for image directories...")
        # Look for possible image directories
        for dir_name in os.listdir(raw_data_dir):
            if os.path.isdir(os.path.join(raw_data_dir, dir_name)) and dir_name != 'splits':
                potential_dir = os.path.join(raw_data_dir, dir_name)
                # Check if this directory contains PNG files
                png_files = [f for f in os.listdir(potential_dir) if f.lower().endswith('.png')]
                if png_files:
                    images_dir = potential_dir
                    print(f"Found image directory: {images_dir}")
                    break
    
    processed_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        try:
            # Get image path - try both .png and without extension
            file_name = str(row['File Name'])
            image_file_png = file_name + '.png'
            image_path_png = os.path.join(images_dir, image_file_png)
            
            # Try alternative paths if the first one doesn't exist
            if not os.path.exists(image_path_png):
                # Try without extension
                image_path = os.path.join(images_dir, file_name)
                if not os.path.exists(image_path):
                    # Try searching for the file in the directory
                    found = False
                    for img_file in os.listdir(images_dir):
                        if file_name in img_file:
                            image_path = os.path.join(images_dir, img_file)
                            found = True
                            break
                    
                    if not found:
                        print(f"Warning: Image file not found for {file_name}")
                        continue
                else:
                    image_path = image_path
            else:
                image_path = image_path_png
            
            # Read image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not read image: {image_path}")
                continue
            
            # Resize image
            image = cv2.resize(image, image_size)
            
            # Save processed image
            class_label = row['class']
            output_filename = f"{file_name}.png"
            output_path = os.path.join(processed_data_dir, class_label, output_filename)
            
            cv2.imwrite(output_path, image)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing image {row['File Name']}: {str(e)}")
    
    print(f"Successfully processed {processed_count} images")
    
    # Create train/val/test split files
    create_data_splits(processed_data_dir)
    
    print("INbreast preprocessing completed!")

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