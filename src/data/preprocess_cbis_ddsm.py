import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import glob
import re
import pydicom  # Add this import for DICOM handling

def preprocess_cbis_ddsm(raw_data_dir, processed_data_dir, image_size=(256, 256)):
    """
    Preprocess the CBIS-DDSM dataset.
    
    Args:
        raw_data_dir: Path to the raw CBIS-DDSM data directory
        processed_data_dir: Path to save the processed data
        image_size: Target size for the images (height, width)
    """
    print("Starting CBIS-DDSM dataset preprocessing...")
    
    # Create directories if they don't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, 'benign'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, 'malignant'), exist_ok=True)
    
    # Search for CSV files recursively
    print("Searching for CSV files recursively...")
    csv_files_found = glob.glob(os.path.join(raw_data_dir, '**/*.csv'), recursive=True)
    print(f"Found {len(csv_files_found)} CSV files:")
    for csv_file in csv_files_found[:10]:  # Show first 10 files
        print(f"  - {os.path.basename(csv_file)}")
    if len(csv_files_found) > 10:
        print(f"  - ... and {len(csv_files_found) - 10} more")
    
    # Look for mass and calcification description files (search recursively)
    mass_train_csv = find_csv_file(csv_files_found, 'mass_case_description_train_set.csv')
    mass_test_csv = find_csv_file(csv_files_found, 'mass_case_description_test_set.csv')
    calc_train_csv = find_csv_file(csv_files_found, 'calc_case_description_train_set.csv')
    calc_test_csv = find_csv_file(csv_files_found, 'calc_case_description_test_set.csv')
    
    # Check if any of these files exist
    description_files = [
        (mass_train_csv, 'mass_train'),
        (mass_test_csv, 'mass_test'),
        (calc_train_csv, 'calc_train'),
        (calc_test_csv, 'calc_test')
    ]
    
    found_csv = False
    for csv_path, csv_type in description_files:
        if csv_path:
            found_csv = True
            print(f"Found {csv_type} CSV file: {csv_path}")
            try:
                df = pd.read_csv(csv_path)
                print(f"Columns in {csv_type}: {df.columns.tolist()}")
                process_description_csv(df, raw_data_dir, processed_data_dir, image_size, csv_type)
            except Exception as e:
                print(f"Error processing {csv_path}: {str(e)}")
    
    # If no description CSVs found, try dicom_info.csv or meta.csv
    if not found_csv:
        dicom_info_path = find_csv_file(csv_files_found, 'dicom_info.csv')
        meta_path = find_csv_file(csv_files_found, 'meta.csv')
        
        if meta_path:
            print(f"Using metadata from {meta_path}")
            try:
                meta_df = pd.read_csv(meta_path)
                print(f"Meta.csv columns: {meta_df.columns.tolist()}")
                process_from_meta_csv(meta_df, raw_data_dir, processed_data_dir, image_size)
            except Exception as e:
                print(f"Error processing meta.csv: {str(e)}")
                process_from_files(raw_data_dir, processed_data_dir, image_size)
        elif dicom_info_path:
            print(f"Using DICOM info from {dicom_info_path}")
            try:
                dicom_df = pd.read_csv(dicom_info_path)
                print(f"Dicom_info.csv columns: {dicom_df.columns.tolist()}")
                process_from_dicom_info(dicom_df, raw_data_dir, processed_data_dir, image_size)
            except Exception as e:
                print(f"Error processing dicom_info.csv: {str(e)}")
                process_from_files(raw_data_dir, processed_data_dir, image_size)
        else:
            # If no CSV files are available, process directly from files
            print("No metadata CSV files found. Processing directly from image files.")
            process_from_files(raw_data_dir, processed_data_dir, image_size)
    
    # Create train/val/test split files
    create_data_splits(processed_data_dir)
    
    print("CBIS-DDSM preprocessing completed!")

# Add this new helper function to find CSV files
def find_csv_file(csv_files, filename):
    """
    Find a CSV file in the list of found CSV files.
    
    Args:
        csv_files: List of CSV files found
        filename: Name of the file to find
    
    Returns:
        Full path to the file or None if not found
    """
    for csv_file in csv_files:
        if os.path.basename(csv_file).lower() == filename.lower():
            return csv_file
    return None

def process_description_csv(df, raw_data_dir, processed_data_dir, image_size, csv_type):
    """Process images using the CBIS-DDSM description CSV files"""
    print(f"Processing from {csv_type} with {len(df)} entries")
    
    # Check for pathology column
    if 'pathology' not in df.columns:
        print(f"Warning: 'pathology' column not found in {csv_type}. Available columns: {df.columns.tolist()}")
        return
    
    # Process each row
    processed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_type}"):
        try:
            # Get pathology and map to class
            pathology = str(row['pathology']).upper()
            if 'MALIGNANT' in pathology:
                class_label = 'malignant'
            elif 'BENIGN' in pathology:
                class_label = 'benign'
            else:
                class_label = 'normal'
            
            # Get patient ID and image info
            patient_id = str(row.get('patient_id', '')).strip()
            if not patient_id:
                patient_id = f"patient_{idx}"
            
            # Get image file path if available
            image_path = None
            for col in ['image file path', 'cropped image file path', 'ROI mask file path']:
                if col in df.columns and pd.notna(row.get(col)):
                    image_path = str(row.get(col)).strip()
                    break
            
            if not image_path:
                continue
            
            # Clean up the image path
            if ',' in image_path:
                image_path = image_path.split(',')[0].strip()
            
            # Find the image file
            full_path = find_image_file(raw_data_dir, image_path)
            if not full_path:
                print(f"Warning: Could not find image file for {image_path}")
                continue
            
            # Process the image
            try:
                if full_path.lower().endswith('.dcm'):
                    # Process DICOM file
                    ds = pydicom.dcmread(full_path)
                    image = ds.pixel_array
                    
                    # Normalize to 0-255
                    image = image - np.min(image)
                    if np.max(image) > 0:
                        image = (image / np.max(image) * 255).astype(np.uint8)
                else:
                    # Process regular image file
                    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"Warning: Could not read image: {full_path}")
                        continue
                
                # Resize image
                image = cv2.resize(image, image_size)
                
                # Create a unique filename
                breast_side = str(row.get('left or right breast', '')).strip()
                view = str(row.get('image view', '')).strip()
                abnormality_id = str(row.get('abnormality id', '1')).strip()
                
                output_filename = f"{patient_id}_{breast_side}_{view}_{abnormality_id}.png"
                output_path = os.path.join(processed_data_dir, class_label, output_filename)
                
                # Save processed image
                cv2.imwrite(output_path, image)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing image {full_path}: {str(e)}")
                
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
    
    print(f"Successfully processed {processed_count} images from {csv_type}")

def find_image_file(base_dir, relative_path):
    """
    Find an image file based on the relative path in the CSV.
    
    Args:
        base_dir: Base directory to search from
        relative_path: Relative path from the CSV
    
    Returns:
        Full path to the image file or None if not found
    """
    # Clean up the relative path
    relative_path = relative_path.replace('"', '').strip()
    
    # Try direct path first
    direct_path = os.path.join(base_dir, relative_path)
    if os.path.exists(direct_path):
        return direct_path
    
    # Try to find by filename
    filename = os.path.basename(relative_path)
    
    # Search for the file recursively
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    
    # If still not found, try to find any file with a similar name
    pattern = re.compile(r'.*' + re.escape(os.path.splitext(filename)[0]) + r'.*', re.IGNORECASE)
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    
    # If all else fails, look for any image file in a directory with a similar name
    dir_parts = os.path.dirname(relative_path).split('/')
    if dir_parts:
        last_dir = dir_parts[-1]
        pattern = re.compile(r'.*' + re.escape(last_dir) + r'.*', re.IGNORECASE)
        
        for root, dirs, files in os.walk(base_dir):
            if pattern.search(root):
                for file in files:
                    if file.lower().endswith(('.dcm', '.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        return os.path.join(root, file)
    
    return None

def process_from_meta_csv(meta_df, raw_data_dir, processed_data_dir, image_size):
    """Process images using meta.csv"""
    print(f"Processing from meta.csv with {len(meta_df)} entries")
    
    # Find all image files in the dataset
    print("Searching for image files...")
    image_files = []
    for ext in ['.dcm', '.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        image_files.extend(glob.glob(os.path.join(raw_data_dir, f'**/*{ext}'), recursive=True))
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image file
    processed_count = 0
    
    # Patterns to identify benign and malignant cases
    benign_pattern = re.compile(r'.*benign.*', re.IGNORECASE)
    malignant_pattern = re.compile(r'.*(malignant|cancer).*', re.IGNORECASE)
    
    for image_path in tqdm(image_files, desc="Processing images from meta.csv"):
        try:
            # Determine class based on file path
            if malignant_pattern.search(image_path):
                class_label = 'malignant'
            elif benign_pattern.search(image_path):
                class_label = 'benign'
            else:
                # If can't determine, default to normal
                class_label = 'normal'
            
            # Process the image
            if image_path.lower().endswith('.dcm'):
                # Process DICOM file
                try:
                    ds = pydicom.dcmread(image_path)
                    image = ds.pixel_array
                    
                    # Normalize to 0-255
                    image = image - np.min(image)
                    if np.max(image) > 0:
                        image = (image / np.max(image) * 255).astype(np.uint8)
                except Exception as e:
                    print(f"Error reading DICOM file {image_path}: {str(e)}")
                    continue
            else:
                # Process regular image file
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Could not read image: {image_path}")
                    continue
            
            # Resize image
            image = cv2.resize(image, image_size)
            
            # Extract patient info from filename if possible
            parts = Path(image_path).parts
            patient_id = "unknown"
            view = "unknown"
            side = "unknown"
            
            # Try to extract patient ID and view from path
            for part in parts:
                if 'P_' in part or 'patient' in part.lower():
                    patient_id = part
                if any(v in part for v in ['CC', 'MLO', 'AT']):
                    view = part
                if any(s in part.upper() for s in ['LEFT', 'RIGHT', 'L_', 'R_']):
                    side = part
            
            # Create a unique filename
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            
            output_filename = f"{patient_id}_{side}_{view}_{base_name}.png"
            output_path = os.path.join(processed_data_dir, class_label, output_filename)
            
            # Save processed image
            cv2.imwrite(output_path, image)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
    
    print(f"Successfully processed {processed_count} images from meta.csv")

def process_from_dicom_info(dicom_df, raw_data_dir, processed_data_dir, image_size):
    """Process images using the dicom_info.csv file"""
    print(f"Processing from dicom_info.csv with {len(dicom_df)} entries")
    
    # Check for required columns
    if 'filename' not in dicom_df.columns:
        print(f"Warning: dicom_info.csv is missing required columns. Available columns: {dicom_df.columns.tolist()}")
        return
    
    # Try to determine pathology from filename or other columns
    processed_count = 0
    
    for idx, row in tqdm(dicom_df.iterrows(), total=len(dicom_df), desc="Processing from dicom_info.csv"):
        try:
            # Get filename
            filename = row['filename']
            if not isinstance(filename, str):
                continue
            
            # Determine class from filename or other columns
            filename_lower = filename.lower()
            if 'malignant' in filename_lower or 'cancer' in filename_lower:
                class_label = 'malignant'
            elif 'benign' in filename_lower:
                class_label = 'benign'
            else:
                # Try to get from other columns if available
                if 'pathology' in dicom_df.columns:
                    pathology = str(row['pathology']).upper()
                    if 'MALIGNANT' in pathology:
                        class_label = 'malignant'
                    elif 'BENIGN' in pathology:
                        class_label = 'benign'
                    else:
                        class_label = 'normal'
                else:
                    class_label = 'normal'
            
            # Find the image file
            full_path = find_image_file(raw_data_dir, filename)
            if not full_path:
                print(f"Warning: Could not find image file for {filename}")
                continue
            
            # Process the image
            if full_path.lower().endswith('.dcm'):
                # Process DICOM file
                try:
                    ds = pydicom.dcmread(full_path)
                    image = ds.pixel_array
                    
                    # Normalize to 0-255
                    image = image - np.min(image)
                    if np.max(image) > 0:
                        image = (image / np.max(image) * 255).astype(np.uint8)
                except Exception as e:
                    print(f"Error reading DICOM file {full_path}: {str(e)}")
                    continue
            else:
                # Process regular image file
                image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Could not read image: {full_path}")
                    continue
            
            # Resize image
            image = cv2.resize(image, image_size)
            
            # Extract patient info from filename if possible
            parts = Path(filename).parts
            patient_id = "unknown"
            view = "unknown"
            side = "unknown"
            
            # Try to extract patient ID and view from path
            for part in parts:
                if 'P_' in part:
                    patient_id = part
                if any(v in part for v in ['CC', 'MLO', 'AT']):
                    view = part
                if any(s in part.upper() for s in ['LEFT', 'RIGHT']):
                    side = part
            
            output_filename = f"{patient_id}_{side}_{view}_{idx}.png"
            output_path = os.path.join(processed_data_dir, class_label, output_filename)
            
            # Save processed image
            cv2.imwrite(output_path, image)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
    
    print(f"Successfully processed {processed_count} images from dicom_info.csv")

def process_from_files(raw_data_dir, processed_data_dir, image_size):
    """Process images directly from files without CSV metadata"""
    print("Processing directly from image files")
    
    # Find all image files recursively
    image_files = []
    for ext in ['.dcm', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
        image_files.extend(glob.glob(os.path.join(raw_data_dir, f'**/*{ext}'), recursive=True))
        image_files.extend(glob.glob(os.path.join(raw_data_dir, f'**/*{ext.upper()}'), recursive=True))
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    processed_count = 0
    normal_count = 0    # Initialize counter variables
    benign_count = 0    # Initialize counter variables
    malignant_count = 0 # Initialize counter variables
    
    # More comprehensive patterns to identify benign and malignant cases
    benign_patterns = [
        r'.*benign.*', 
        r'.*mass_B.*',
        r'.*calc_B.*',
        r'.*B_\d+.*',
        r'.*_B_.*',
        r'.*_benign_.*',
        r'.*benign[_\s].*',
        r'.*without_callback.*'
    ]
    
    malignant_patterns = [
        r'.*(malignant|cancer).*',
        r'.*mass_M.*',
        r'.*calc_M.*',
        r'.*M_\d+.*',
        r'.*_M_.*',
        r'.*_malignant_.*',
        r'.*malignant[_\s].*',
        r'.*with_callback.*'
    ]
    
    # Compile all patterns
    benign_regex = re.compile('|'.join(benign_patterns), re.IGNORECASE)
    malignant_regex = re.compile('|'.join(malignant_patterns), re.IGNORECASE)
    
    # Check if we have a folder structure that indicates classes
    benign_folders = ['Benign', 'benign', 'BENIGN', 'without_callback']
    malignant_folders = ['Malignant', 'malignant', 'MALIGNANT', 'Cancer', 'cancer', 'CANCER', 'with_callback']
    
    # Sample a few images to determine if we need to adjust our classification strategy
    sample_images = image_files[:min(100, len(image_files))]
    benign_matches = 0
    malignant_matches = 0
    
    for image_path in sample_images:
        path_parts = Path(image_path).parts
        if any(part in malignant_folders for part in path_parts) or malignant_regex.search(image_path):
            malignant_matches += 1
        elif any(part in benign_folders for part in path_parts) or benign_regex.search(image_path):
            benign_matches += 1
    
    print(f"Sample classification: {benign_matches} benign, {malignant_matches} malignant")
    
    # If we're not finding any matches with our patterns, try a different approach
    if benign_matches == 0 and malignant_matches == 0:
        print("No matches found with standard patterns. Using alternative classification approach.")
        # For CBIS-DDSM, we can try to classify based on directory structure
        # Typically, CBIS-DDSM has Mass-Training, Mass-Test, Calc-Training, Calc-Test folders
        
        # Process each image
        for image_path in tqdm(image_files, desc="Processing CBIS-DDSM images"):
            try:
                # For CBIS-DDSM, we'll check if the image is in a specific directory structure
                path_str = str(image_path).lower()
                
                # Try to determine if this is a benign or malignant case
                # For CBIS-DDSM, we might need to check parent directories
                parent_dir = os.path.dirname(image_path)
                parent_name = os.path.basename(parent_dir).lower()
                
                # Check if parent directory contains classification info
                if 'benign' in parent_name or 'without_callback' in parent_name:
                    class_label = 'benign'
                    benign_count += 1
                elif 'malignant' in parent_name or 'cancer' in parent_name or 'with_callback' in parent_name:
                    class_label = 'malignant'
                    malignant_count += 1
                else:
                    # If we can't determine from parent, check grandparent
                    grandparent_dir = os.path.dirname(parent_dir)
                    grandparent_name = os.path.basename(grandparent_dir).lower()
                    
                    if 'benign' in grandparent_name or 'without_callback' in grandparent_name:
                        class_label = 'benign'
                        benign_count += 1
                    elif 'malignant' in grandparent_name or 'cancer' in grandparent_name or 'with_callback' in grandparent_name:
                        class_label = 'malignant'
                        malignant_count += 1
                    else:
                        # If still can't determine, use the filename
                        filename = os.path.basename(image_path).lower()
                        
                        if 'benign' in filename or 'without_callback' in filename:
                            class_label = 'benign'
                            benign_count += 1
                        elif 'malignant' in filename or 'cancer' in filename or 'with_callback' in filename:
                            class_label = 'malignant'
                            malignant_count += 1
                        else:
                            # If we still can't determine, default to normal
                            class_label = 'normal'
                            normal_count += 1
                
                # Read image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Could not read image: {image_path}")
                    continue
                
                # Resize image
                image = cv2.resize(image, image_size)
                
                # Extract patient info from filename if possible
                filename = os.path.basename(image_path)
                base_name = os.path.splitext(filename)[0]
                
                # Try to extract patient ID and view from path
                path_parts = Path(image_path).parts
                patient_id = "unknown"
                view = "unknown"
                side = "unknown"
                
                for part in path_parts:
                    part_lower = part.lower()
                    # Look for patient ID patterns
                    if 'patient' in part_lower or 'p_' in part_lower or re.match(r'p\d+', part_lower):
                        patient_id = part
                    # Look for view patterns
                    if any(v in part_lower for v in ['cc', 'mlo', 'at']):
                        view = part
                    # Look for side patterns
                    if any(s in part_lower for s in ['left', 'right', 'l_', 'r_']):
                        side = part
                
                output_filename = f"{patient_id}_{side}_{view}_{base_name}.png"
                output_path = os.path.join(processed_data_dir, class_label, output_filename)
                
                # Save processed image
                cv2.imwrite(output_path, image)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
    else:
        # Use our original approach if patterns are working
        for image_path in tqdm(image_files, desc="Processing CBIS-DDSM images"):
            try:
                # Determine class based on file path or folder structure
                path_parts = Path(image_path).parts
                
                # Check if any part of the path matches our patterns
                if any(part in malignant_folders for part in path_parts) or malignant_regex.search(image_path):
                    class_label = 'malignant'
                    malignant_count += 1
                elif any(part in benign_folders for part in path_parts) or benign_regex.search(image_path):
                    class_label = 'benign'
                    benign_count += 1
                else:
                    # If can't determine, default to normal
                    class_label = 'normal'
                    normal_count += 1
                
                # Read image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Could not read image: {image_path}")
                    continue
                
                # Resize image
                image = cv2.resize(image, image_size)
                
                # Extract patient info from filename if possible
                filename = os.path.basename(image_path)
                base_name = os.path.splitext(filename)[0]
                
                # Try to extract patient ID and view from path
                patient_id = "unknown"
                view = "unknown"
                side = "unknown"
                
                for part in path_parts:
                    part_lower = part.lower()
                    # Look for patient ID patterns
                    if 'patient' in part_lower or 'p_' in part_lower or re.match(r'p\d+', part_lower):
                        patient_id = part
                    # Look for view patterns
                    if any(v in part_lower for v in ['cc', 'mlo', 'at']):
                        view = part
                    # Look for side patterns
                    if any(s in part_lower for s in ['left', 'right', 'l_', 'r_']):
                        side = part
                
                output_filename = f"{patient_id}_{side}_{view}_{base_name}.png"
                output_path = os.path.join(processed_data_dir, class_label, output_filename)
                
                # Save processed image
                cv2.imwrite(output_path, image)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
    
    print(f"Successfully processed {processed_count} images:")
    print(f"  - Normal: {normal_count}")
    print(f"  - Benign: {benign_count}")
    print(f"  - Malignant: {malignant_count}")
    
    # Create train/val/test split files
    create_data_splits(processed_data_dir)
    
    print("CBIS-DDSM preprocessing completed!")

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