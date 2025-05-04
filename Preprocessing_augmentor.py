# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 15:29:34 2025

@author: gopia
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

# ================================
# Configuration
# ================================
SOURCE_DIR = '/content/drive/MyDrive/Embryo data/Train'
TARGET_IMAGE_COUNT = 500  # Minimum images per class after augmentation
IMAGE_SIZE = (224, 224)

# ================================
# Data Augmentation Settings
# ================================
augmentor = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.1,
    fill_mode='nearest'
)

# ================================
# Function to Augment a Class Folder
# ================================
def augment_class_images(folder_path, target_count):
    """
    Augments images in a given folder until it reaches the target count.
    
    Args:
        folder_path (str): Path to the image class folder.
        target_count (int): Target number of images after augmentation.
    """
    image_files = os.listdir(folder_path)
    current_count = len(image_files)
    print(f"[{folder_path}] - Current: {current_count}")

    i = 0
    while current_count + i < target_count:
        img_name = np.random.choice(image_files)
        img_path = os.path.join(folder_path, img_name)

        try:
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img)
            img_array = img_array.reshape((1,) + img_array.shape)

            for batch in augmentor.flow(img_array, batch_size=1):
                save_name = f"aug_{i}_{img_name}"
                save_path = os.path.join(folder_path, save_name)
                array_to_img(batch[0]).save(save_path)
                i += 1
                break  # Generate one augmented image per loop
        except Exception as e:
            print(f"Skipped invalid image: {img_path} | Error: {str(e)}")

# ================================
# Function to Process All Folders
# ================================
def apply_augmentation_to_dataset(base_dir, target_count):
    """
    Applies augmentation across all class folders, including nested ones.

    Args:
        base_dir (str): Base directory containing class folders.
        target_count (int): Minimum number of images per class after augmentation.
    """
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        # Handle Error folder directly
        if folder.lower() == 'error':
            augment_class_images(folder_path, target_count)
            continue

        # Handle nested class folders (e.g., Day3/GradeA)
        for grade in os.listdir(folder_path):
            grade_path = os.path.join(folder_path, grade)
            if os.path.isdir(grade_path):
                augment_class_images(grade_path, target_count)

# ================================
# Main Execution
# ================================
if __name__ == "__main__":
    # Step 1: EDA (already included previously)
    from eda_module import count_images_per_class, plot_class_distribution, show_samples_from_nested_folders
    counts = count_images_per_class(SOURCE_DIR)
    plot_class_distribution(counts)
    show_samples_from_nested_folders(SOURCE_DIR)

    # Step 2: Data Augmentation
    apply_augmentation_to_dataset(SOURCE_DIR, TARGET_IMAGE_COUNT)

    # Step 3: Re-check distribution after augmentation (optional)
    updated_counts = count_images_per_class(SOURCE_DIR)
    plot_class_distribution(updated_counts)
