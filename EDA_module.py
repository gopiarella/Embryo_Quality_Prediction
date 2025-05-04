# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 15:28:06 2025

@author: gopia
"""

import os
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import random

# ================================
# Configuration
# ================================
BASE_DIR = '/content/drive/MyDrive/Embryo data/Train'
SAMPLES_PER_CLASS = 3  # Number of sample images to display per class

# ================================
# Function to Count Images Per Class
# ================================
def count_images_per_class(base_dir):
    """
    Counts the number of images in each class folder.
    Supports nested directory structure and handles a separate 'Error' folder.
    
    Args:
        base_dir (str): Base directory path containing class folders.
    
    Returns:
        dict: Dictionary with class names as keys and image counts as values.
    """
    image_counts = defaultdict(int)

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        
        if not os.path.isdir(folder_path):
            continue

        # Handle the "Error" folder separately
        if folder.lower() == "error":
            image_counts["Error"] = len(os.listdir(folder_path))
            continue

        # For nested structure: e.g., Day3/GradeA
        for grade in os.listdir(folder_path):
            grade_path = os.path.join(folder_path, grade)
            if os.path.isdir(grade_path):
                key = f"{folder}_{grade}"
                image_counts[key] = len(os.listdir(grade_path))

    return image_counts

# ================================
# Function to Plot Class Distribution
# ================================
def plot_class_distribution(image_counts):
    """
    Plots the number of images per class using a bar chart.
    
    Args:
        image_counts (dict): Dictionary with class names and image counts.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(image_counts.keys(), image_counts.values(), color='teal')
    plt.xticks(rotation=45, ha='right')
    plt.title("Train Set: Image Count per Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()

# ================================
# Function to Show Sample Images
# ================================
def show_samples_from_nested_folders(base_path, samples_per_class=3):
    """
    Displays a few sample images from each class for visual inspection.
    
    Args:
        base_path (str): Base directory path with class folders.
        samples_per_class (int): Number of sample images to show per class.
    """
    plt.figure(figsize=(15, 10))
    idx = 1  # Subplot index

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue

        # Handle the "Error" folder separately
        if folder.lower() == "error":
            images = os.listdir(folder_path)
            sample_imgs = random.sample(images, min(len(images), samples_per_class))
            for img_name in sample_imgs:
                img_path = os.path.join(folder_path, img_name)
                img = Image.open(img_path)
                plt.subplot(10, samples_per_class, idx)
                plt.imshow(img)
                plt.axis('off')
                plt.title("Error")
                idx += 1
            continue

        # Handle nested folders (e.g., Day1/GradeA)
        for grade in os.listdir(folder_path):
            class_path = os.path.join(folder_path, grade)
            if os.path.isdir(class_path):
                images = os.listdir(class_path)
                sample_imgs = random.sample(images, min(len(images), samples_per_class))
                for img_name in sample_imgs:
                    img_path = os.path.join(class_path, img_name)
                    img = Image.open(img_path)
                    plt.subplot(10, samples_per_class, idx)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"{folder}_{grade}")
                    idx += 1

    plt.tight_layout()
    plt.show()

# ================================
# Main Execution
# ================================
if __name__ == "__main__":
    # Step 1: Count images per class
    class_counts = count_images_per_class(BASE_DIR)

    # Step 2: Plot the class distribution
    plot_class_distribution(class_counts)

    # Step 3: Visualize sample images from each class
    show_samples_from_nested_folders(BASE_DIR, samples_per_class=SAMPLES_PER_CLASS)
