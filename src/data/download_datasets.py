import os
import argparse
import requests
import zipfile
import tarfile
import gzip
import shutil
from tqdm import tqdm

def download_file(url, destination):
    """
    Download a file from a URL to a destination with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def extract_archive(archive_path, extract_dir):
    """
    Extract an archive file.
    
    Args:
        archive_path: Path to the archive file
        extract_dir: Directory to extract to
    """
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting {archive_path} to {extract_dir}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_dir)
    elif archive_path.endswith('.gz') and not archive_path.endswith('.tar.gz'):
        output_path = archive_path[:-3]  # Remove .gz extension
        with gzip.open(archive_path, 'rb') as f_in:
            with open(os.path.join(extract_dir, os.path.basename(output_path)), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print(f"Unsupported archive format: {archive_path}")

def download_cbis_ddsm():
    """Download CBIS-DDSM dataset."""
    base_url = "https://wiki.cancerimagingarchive.net/download/attachments/22516629/"
    files = [
        "mass_case_description_train_set.csv",
        "calc_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "calc_case_description_test_set.csv"
    ]
    
    data_dir = "c:\\SDP_Project_P3\\data\\raw\\CBIS-DDSM"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading CBIS-DDSM metadata...")
    for file in files:
        download_file(base_url + file, os.path.join(data_dir, file))
    
    print("CBIS-DDSM metadata downloaded.")
    print("Note: The actual CBIS-DDSM images are very large (>100GB).")
    print("Please download them manually from:")
    print("https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM")
    print("or use a smaller subset for development.")

def download_mias():
    """Download MIAS dataset."""
    data_dir = "c:\\SDP_Project_P3\\data\\raw\\MIAS"
    os.makedirs(data_dir, exist_ok=True)
    
    # Download images
    images_url = "http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz"
    images_path = os.path.join(data_dir, "all-mias.tar.gz")
    
    print("Downloading MIAS dataset...")
    download_file(images_url, images_path)
    
    # Download info file
    info_url = "http://peipa.essex.ac.uk/info/mias/info.txt"
    info_path = os.path.join(data_dir, "info.txt")
    download_file(info_url, info_path)
    
    # Extract images
    extract_archive(images_path, data_dir)
    
    print("MIAS dataset downloaded and extracted.")

def download_inbreast():
    """Download INbreast dataset."""
    data_dir = "c:\\SDP_Project_P3\\data\\raw\\INbreast"
    os.makedirs(data_dir, exist_ok=True)
    
    print("INbreast dataset requires registration.")
    print("Please download it manually from:")
    print("http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database")
    print("and place the files in:", data_dir)

def download_mini_ddsm():
    """Download mini-DDSM dataset (a smaller subset for development)."""
    data_dir = "c:\\SDP_Project_P3\\data\\raw\\mini-DDSM"
    os.makedirs(data_dir, exist_ok=True)
    
    # This is a placeholder for a smaller subset of DDSM
    # In a real scenario, you would create or find a smaller subset
    print("Creating mini-DDSM dataset for development...")
    
    # For demonstration, we'll create a small synthetic dataset
    import numpy as np
    from PIL import Image
    
    # Create directories
    os.makedirs(os.path.join(data_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "benign"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "malignant"), exist_ok=True)
    
    # Create synthetic images (just for demonstration)
    for class_name, pattern in [("normal", "uniform"), ("benign", "circular"), ("malignant", "irregular")]:
        for i in range(10):  # 10 images per class
            img = np.zeros((224, 224), dtype=np.uint8)
            
            # Add some pattern based on class
            if pattern == "uniform":
                img = np.random.randint(100, 180, (224, 224), dtype=np.uint8)
            elif pattern == "circular":
                img = np.random.randint(100, 180, (224, 224), dtype=np.uint8)
                center_x, center_y = np.random.randint(50, 174, 2)
                radius = np.random.randint(10, 30)
                for x in range(224):
                    for y in range(224):
                        if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                            img[y, x] = np.random.randint(180, 220)
            elif pattern == "irregular":
                img = np.random.randint(100, 180, (224, 224), dtype=np.uint8)
                for _ in range(3):  # Multiple irregular regions
                    center_x, center_y = np.random.randint(50, 174, 2)
                    radius = np.random.randint(10, 30)
                    for x in range(224):
                        for y in range(224):
                            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                            if dist < radius + np.random.randint(-10, 10):
                                img[y, x] = np.random.randint(180, 220)
            
            # Save image
            Image.fromarray(img).save(os.path.join(data_dir, class_name, f"{class_name}_{i+1}.png"))
    
    # Create metadata file
    with open(os.path.join(data_dir, "metadata.csv"), "w") as f:
        f.write("image_path,class\n")
        for class_name, class_id in [("normal", 0), ("benign", 1), ("malignant", 2)]:
            for i in range(10):
                f.write(f"{class_name}/{class_name}_{i+1}.png,{class_id}\n")
    
    print("Mini-DDSM dataset created for development.")

def main():
    parser = argparse.ArgumentParser(description="Download breast cancer datasets")
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        choices=["all", "cbis-ddsm", "mias", "inbreast", "mini-ddsm"],
                        help="Datasets to download")
    
    args = parser.parse_args()
    
    if "all" in args.datasets or "cbis-ddsm" in args.datasets:
        download_cbis_ddsm()
    
    if "all" in args.datasets or "mias" in args.datasets:
        download_mias()
    
    if "all" in args.datasets or "inbreast" in args.datasets:
        download_inbreast()
    
    if "all" in args.datasets or "mini-ddsm" in args.datasets:
        download_mini_ddsm()
    
    print("Dataset download complete.")

if __name__ == "__main__":
    main()