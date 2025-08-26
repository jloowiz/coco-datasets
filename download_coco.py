#!/usr/bin/env python3
"""
Script to download COCO 2017 dataset files
"""
import os
import urllib.request
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    print(f"Downloading {filename}...")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100 / total_size
            print(f"\rProgress: {percent:.1f}% ({downloaded:,} / {total_size:,} bytes)", end='')
    
    urllib.request.urlretrieve(url, filename, progress_hook)
    print(f"\nDownloaded {filename}")

def main():
    # Create directories
    os.makedirs("annotations", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    
    # URLs for COCO 2017 dataset
    urls = {
        "annotations/annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "images/train2017.zip": "http://images.cocodataset.org/zips/train2017.zip"
    }
    
    # Download files
    for filename, url in urls.items():
        if not os.path.exists(filename):
            download_file(url, filename)
        else:
            print(f"{filename} already exists, skipping download")
    
    # Extract annotations
    if os.path.exists("annotations/annotations_trainval2017.zip"):
        print("Extracting annotations...")
        with zipfile.ZipFile("annotations/annotations_trainval2017.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Annotations extracted")
    
    # Extract images (this will take a while - ~19GB)
    if os.path.exists("images/train2017.zip"):
        print("Extracting training images (this may take a while - ~19GB)...")
        with zipfile.ZipFile("images/train2017.zip", 'r') as zip_ref:
            zip_ref.extractall("images/")
        print("Training images extracted")
    
    print("\nDataset download and extraction complete!")
    print("You can now run: python datasets.py")

if __name__ == "__main__":
    main()
