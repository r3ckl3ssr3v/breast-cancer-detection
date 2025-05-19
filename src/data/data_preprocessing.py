import os
import argparse
from preprocess_inbreast import preprocess_inbreast
from preprocess_mias import preprocess_mias
from preprocess_mini_ddsm import preprocess_mini_ddsm
from preprocess_cbis_ddsm import preprocess_cbis_ddsm

def main():
    parser = argparse.ArgumentParser(description='Preprocess mammogram datasets')
    parser.add_argument('--dataset', type=str, default='all', 
                        choices=['all', 'MIAS', 'DDSM', 'CBIS-DDSM', 'INbreast', 'mini-ddsm'],
                        help='Dataset to preprocess')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images to (square)')
    
    args = parser.parse_args()
    
    # Define paths - Fix the path construction
    base_dir = os.path.abspath(os.path.join('e:\\', 'SDP_Project_P3'))  # Fixed path with proper backslash
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')
    
    # Create processed data directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Process selected dataset(s)
    if args.dataset == 'all' or args.dataset == 'INbreast':
        inbreast_raw_dir = os.path.join(raw_data_dir, 'INbreast')
        inbreast_processed_dir = os.path.join(processed_data_dir, 'INbreast')
        preprocess_inbreast(inbreast_raw_dir, inbreast_processed_dir, 
                           image_size=(args.image_size, args.image_size))
    
    # Process MIAS dataset
    if args.dataset == 'all' or args.dataset == 'MIAS':
        mias_raw_dir = os.path.join(raw_data_dir, 'MIAS')
        mias_processed_dir = os.path.join(processed_data_dir, 'MIAS')
        preprocess_mias(mias_raw_dir, mias_processed_dir, 
                       image_size=(args.image_size, args.image_size))
    
    # Process mini-DDSM dataset
    if args.dataset == 'all' or args.dataset == 'mini-ddsm':
        mini_ddsm_raw_dir = os.path.join(raw_data_dir, 'mini-DDSM')
        mini_ddsm_processed_dir = os.path.join(processed_data_dir, 'mini-DDSM')
        preprocess_mini_ddsm(mini_ddsm_raw_dir, mini_ddsm_processed_dir, 
                            image_size=(args.image_size, args.image_size))
    
    # Process CBIS-DDSM dataset
    if args.dataset == 'all' or args.dataset == 'CBIS-DDSM':
        cbis_ddsm_raw_dir = os.path.join(raw_data_dir, 'CBIS-DDSM')
        cbis_ddsm_processed_dir = os.path.join(processed_data_dir, 'CBIS-DDSM')
        preprocess_cbis_ddsm(cbis_ddsm_raw_dir, cbis_ddsm_processed_dir, 
                            image_size=(args.image_size, args.image_size))
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()