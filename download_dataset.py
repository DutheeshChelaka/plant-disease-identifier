"""
PlantVillage Dataset Downloader
================================

This script downloads the PlantVillage dataset from Kaggle.
Dataset: 54,000+ images of plant leaves with diseases.

Usage:
    python download_dataset.py
"""

import os
from pathlib import Path


def create_data_folders():
    """Create necessary folder structure for dataset."""
    print("📁 Creating data folder structure...")
    
    folders = [
        'data/raw',
        'data/processed',
        'data/train',
        'data/val',
        'data/test'
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {folder}")
    
    print("✅ Folder structure created!\n")


def check_kaggle_setup():
    """Check if Kaggle API is set up."""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    if kaggle_json.exists():
        print("✅ Kaggle API credentials found!\n")
        return True
    else:
        print("❌ Kaggle API credentials not found!\n")
        print("📝 To download the dataset automatically:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. Move downloaded kaggle.json to ~/.kaggle/")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json\n")
        print("📌 Manual Download:")
        print("Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("Download and extract to: data/raw/PlantVillage/\n")
        return False


def download_from_kaggle():
    """Download dataset using Kaggle API."""
    try:
        import kaggle
        
        print("📥 Downloading PlantVillage dataset from Kaggle...")
        print("   (This may take 5-10 minutes)\n")
        
        kaggle.api.dataset_download_files(
            'abdallahalidev/plantvillage-dataset',
            path='data/raw',
            unzip=True
        )
        
        print("\n✅ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        return False


def verify_dataset():
    """Verify that dataset was downloaded correctly."""
    dataset_path = Path('data/raw/PlantVillage')
    
    if not dataset_path.exists():
        print("❌ Dataset folder not found at data/raw/PlantVillage")
        return False
    
    # Count classes (should be 38)
    classes = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(classes) == 0:
        print("❌ No class folders found in dataset")
        return False
    
    print(f"✅ Dataset verified!")
    print(f"   Found {len(classes)} disease classes")
    
    # Count total images
    total_images = 0
    for class_dir in classes:
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG'))
        total_images += len(images)
    
    print(f"   Found {total_images:,} total images\n")
    return True


def main():
    """Main function to orchestrate dataset download."""
    print("🌱 PlantVillage Dataset Downloader")
    print("=" * 50)
    print()
    
    # Step 1: Create folders
    create_data_folders()
    
    # Step 2: Check if dataset already exists
    if verify_dataset():
        print("🎉 Dataset already downloaded and verified!")
        return
    
    # Step 3: Check Kaggle setup
    if check_kaggle_setup():
        # Step 4: Download from Kaggle
        if download_from_kaggle():
            # Step 5: Verify download
            verify_dataset()
    else:
        print("⏸️  Please set up Kaggle API or download manually.")


if __name__ == "__main__":
    main()