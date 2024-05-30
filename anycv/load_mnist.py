import os
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

# Define the directory structure
base_dir = './data/'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

print("Organizing MNIST dataset into the specified folder structure...")
for i in range(10):
    os.makedirs(os.path.join(train_dir, f'class_{i}'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, f'class_{i}'), exist_ok=True)

print("Loading MNIST dataset...")
# Load the MNIST dataset
(x_train, y_train), (x_val, y_val) = mnist.load_data()

# Helper function to save images
def save_images(images, labels, directory):
    for idx, (image, label) in enumerate(zip(images, labels)):
        img = Image.fromarray(image)
        img_path = os.path.join(directory, f'class_{label}', f'{idx}.png')
        img.save(img_path)

# Save training images
save_images(x_train, y_train, train_dir)

# Save validation images
save_images(x_val, y_val, val_dir)

print("MNIST dataset organized into the specified folder structure.")