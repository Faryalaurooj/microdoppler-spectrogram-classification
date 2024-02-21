import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define function to load spectrograms and labels
def load_spectrograms(directory):
    spectrograms = []
    labels = []
    class_map = {'Copper': 0, 'Aluminum': 1, 'Brass': 2}  # Map subfolder names to numerical labels

    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith('.png'):
                    filepath = os.path.join(class_dir, filename)
                    spectrogram = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Load spectrogram as grayscale
                    # Resize spectrogram to desired dimensions
                    spectrogram = cv2.resize(spectrogram, (256, 256))  # Assuming width = 256, height = 256
                    spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension
                    spectrograms.append(spectrogram)
                    labels.append(class_map[class_name])

    return np.array(spectrograms), np.array(labels), class_map  # Return class_map as well

# Define parameters
directory = 'doppler data'  # Path to directory containing subfolders of spectrogram images

# Load spectrograms, labels, and class_map
X, y, class_map = load_spectrograms(directory)

# Create ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # Randomly rotate images by 10 degrees
    width_shift_range=0.1,  # Randomly shift images horizontally by 10%
    height_shift_range=0.1,  # Randomly shift images vertically by 10%
    shear_range=0.2,  # Shear angle in counter-clockwise direction in degrees
    zoom_range=0.2,  # Randomly zoom images by 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    vertical_flip=True,  # Randomly flip images vertically
    fill_mode='nearest'  # Fill mode for points outside the input boundaries
)

# Define different resolutions for augmentation
resolutions = [(64, 64), (96, 96), (128, 128), (256,256)]

# Save the augmented images to subfolders
for class_name in ['Copper', 'Aluminum', 'Brass']:
    class_dir = os.path.join('augmented_data3', class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Filter images belonging to current class
    class_indices = np.where(y == class_map[class_name])[0]
    class_images = X[class_indices]

    # Apply augmentation for each resolution
    for resolution in resolutions:
        # Create directory for current resolution
        resolution_dir = os.path.join(class_dir, f'{resolution[0]}x{resolution[1]}')
        if not os.path.exists(resolution_dir):
            os.makedirs(resolution_dir)

        # Resize images to current resolution
        resized_images = [cv2.resize(image, (resolution[0], resolution[1])) for image in class_images]

        # Reshape input data to have rank 4
        resized_images = np.expand_dims(resized_images, axis=-1)  # Add channel dimension
        resized_images = np.repeat(resized_images, 3, axis=-1)  # Repeat grayscale channel to simulate RGB channels

        # Apply augmentation and save images
        i = 0
        for X_batch, y_batch in datagen.flow(resized_images, np.zeros(len(resized_images)), batch_size=32, save_to_dir=resolution_dir, save_prefix='aug', save_format='png'):
            i += 1
            if i >= 10:  # Save 10 augmented images per original image
                break

print("Data augmentation completed and saved to subfolders in 'augmented_data3'")




