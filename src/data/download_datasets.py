import os
import argparse
import zipfile
import tarfile
import gzip
import shutil
from tqdm import tqdm

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

def setup_cbis_ddsm():
    """Setup directory for CBIS-DDSM dataset."""
    data_dir = "c:\\SDP_Project_P3\\data\\raw\\CBIS-DDSM"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"CBIS-DDSM directory created at: {data_dir}")
    print("Please download the CBIS-DDSM dataset from Kaggle and place it in this directory.")
    print("After downloading, extract any archive files to this location.")
    print("Expected structure:")
    print(f"{data_dir}\\")
    print("  ├── mass_case_description_train_set.csv")
    print("  ├── calc_case_description_train_set.csv")
    print("  ├── mass_case_description_test_set.csv")
    print("  ├── calc_case_description_test_set.csv")
    print("  └── images/")

def setup_mias():
    """Setup directory for MIAS dataset."""
    data_dir = "c:\\SDP_Project_P3\\data\\raw\\MIAS"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"MIAS directory created at: {data_dir}")
    print("Please download the MIAS dataset from Kaggle and place it in this directory.")
    print("After downloading, extract any archive files to this location.")
    print("Expected structure:")
    print(f"{data_dir}\\")
    print("  ├── info.txt (metadata file)")
    print("  └── images/ (containing .pgm files)")

def setup_inbreast():
    """Setup directory for INbreast dataset."""
    data_dir = "c:\\SDP_Project_P3\\data\\raw\\INbreast"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"INbreast directory created at: {data_dir}")
    print("Please download the INbreast dataset from Kaggle and place it in this directory.")
    print("After downloading, extract any archive files to this location.")
    print("Expected structure:")
    print(f"{data_dir}\\")
    print("  ├── metadata.csv")
    print("  └── images/")

def setup_mini_ddsm():
    """Setup directory for mini-DDSM dataset."""
    data_dir = "c:\\SDP_Project_P3\\data\\raw\\mini-DDSM"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"mini-DDSM directory created at: {data_dir}")
    print("Please download the mini-DDSM dataset from Kaggle and place it in this directory.")
    print("After downloading, extract any archive files to this location.")
    print("Expected structure:")
    print(f"{data_dir}\\")
    print("  ├── normal/")
    print("  ├── benign/")
    print("  ├── malignant/")
    print("  └── metadata.csv")

def process_downloaded_data(dataset_name, archive_path=None):
    """
    Process already downloaded dataset archives.
    
    Args:
        dataset_name: Name of the dataset
        archive_path: Path to the downloaded archive file
    """
    if not archive_path:
        print(f"No archive path provided for {dataset_name}. Skipping extraction.")
        return
    
    if not os.path.exists(archive_path):
        print(f"Archive file not found: {archive_path}")
        return
    
    data_dir = f"c:\\SDP_Project_P3\\data\\raw\\{dataset_name}"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Processing downloaded {dataset_name} dataset...")
    extract_archive(archive_path, data_dir)
    print(f"{dataset_name} dataset extracted to {data_dir}")

def main():
    parser = argparse.ArgumentParser(description="Setup directories for breast cancer datasets")
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        choices=["all", "cbis-ddsm", "mias", "inbreast", "mini-ddsm"],
                        help="Datasets to setup directories for")
    parser.add_argument("--process", action="store_true",
                        help="Process already downloaded archives")
    parser.add_argument("--cbis-ddsm-archive", type=str, default=None,
                        help="Path to downloaded CBIS-DDSM archive")
    parser.add_argument("--mias-archive", type=str, default=None,
                        help="Path to downloaded MIAS archive")
    parser.add_argument("--inbreast-archive", type=str, default=None,
                        help="Path to downloaded INbreast archive")
    parser.add_argument("--mini-ddsm-archive", type=str, default=None,
                        help="Path to downloaded mini-DDSM archive")
    
    args = parser.parse_args()
    
    if args.process:
        # Process already downloaded archives
        if "all" in args.datasets or "cbis-ddsm" in args.datasets:
            process_downloaded_data("CBIS-DDSM", args.cbis_ddsm_archive)
        
        if "all" in args.datasets or "mias" in args.datasets:
            process_downloaded_data("MIAS", args.mias_archive)
        
        if "all" in args.datasets or "inbreast" in args.datasets:
            process_downloaded_data("INbreast", args.inbreast_archive)
        
        if "all" in args.datasets or "mini-ddsm" in args.datasets:
            process_downloaded_data("mini-DDSM", args.mini_ddsm_archive)
    else:
        # Setup directories and provide instructions
        if "all" in args.datasets or "cbis-ddsm" in args.datasets:
            setup_cbis_ddsm()
        
        if "all" in args.datasets or "mias" in args.datasets:
            setup_mias()
        
        if "all" in args.datasets or "inbreast" in args.datasets:
            setup_inbreast()
        
        if "all" in args.datasets or "mini-ddsm" in args.datasets:
            setup_mini_ddsm()
    
    print("\nSetup complete. Please download the datasets from Kaggle and place them in the appropriate directories.")
    print("After downloading, you can use the --process flag with the archive paths to extract them automatically.")
    print("Example:")
    print("python src\\data\\download_datasets.py --process --mias-archive=\"path\\to\\mias.zip\"")

if __name__ == "__main__":
    main()