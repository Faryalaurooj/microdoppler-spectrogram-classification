import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import precision_score, recall_score, f1_score

# Define function to load augmented spectrograms and labels
def load_augmented_data(directory):
    augmented_spectrograms = []
    augmented_labels = []
    class_map = {'Copper': 0, 'Aluminum': 1, 'Brass': 2}  # Map subfolder names to numerical labels

    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for resolution_folder in os.listdir(class_dir):
                resolution_dir = os.path.join(class_dir, resolution_folder)
                if os.path.isdir(resolution_dir):
                    for filename in os.listdir(resolution_dir):
                        if filename.startswith('aug'):
                            filepath = os.path.join(resolution_dir, filename)
                            spectrogram = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Load spectrogram as grayscale
                            # Resize spectrogram to desired dimensions
                            spectrogram = cv2.resize(spectrogram, (256, 256))  # Assuming width = 256, height = 256
                            augmented_spectrograms.append(spectrogram)
                            augmented_labels.append(class_map[class_name])

    return np.array(augmented_spectrograms), np.array(augmented_labels)

# Load original spectrograms and labels
def load_original_data(directory):
    original_spectrograms = []
    original_labels = []
    class_map = {'Copper': 0, 'Aluminum': 1, 'Brass': 2}  # Map subfolder names to numerical labels

    for class_name in class_map:
        class_dir = os.path.join(directory, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.png'):
                filepath = os.path.join(class_dir, filename)
                spectrogram = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Load spectrogram as grayscale
                # Resize spectrogram to desired dimensions
                spectrogram = cv2.resize(spectrogram, (256, 256))  # Assuming width = 256, height = 256
                original_spectrograms.append(spectrogram)
                original_labels.append(class_map[class_name])

    return np.array(original_spectrograms), np.array(original_labels)

# Load augmented spectrograms and labels
X_augmented, y_augmented = load_augmented_data('augmented_data3')

# Load original spectrograms and labels
X_original, y_original = load_original_data('doppler data')

# Concatenate augmented and original data
X_combined = np.concatenate([X_augmented, X_original], axis=0)
y_combined = np.concatenate([y_augmented, y_original], axis=0)

# Split combined data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# # Define the model
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(256, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(512, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(3, activation='softmax')  # Assuming 3 classes: copper, aluminium, brass
# ])
# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(1024, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # Assuming 3 classes: copper, aluminium, brass
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define a checkpoint callback
checkpoint_callback = callbacks.ModelCheckpoint(filepath='best_model.h5',
                                                 monitor='val_loss',
                                                 mode='min',
                                                 save_best_only=True,
                                                 verbose=1)

# Reshape the input data
X_train = X_train.reshape(X_train.shape[0], 256, 256, 1)
X_test = X_test.reshape(X_test.shape[0], 256, 256, 1)

# Train the model with the checkpoint callback
model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test), callbacks=[checkpoint_callback])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Predict labels for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate precision, recall, and F1-score for each class
precision = precision_score(y_test, y_pred_classes, average=None)
recall = recall_score(y_test, y_pred_classes, average=None)
f1 = f1_score(y_test, y_pred_classes, average=None)

# Print precision, recall, and F1-score for each class
for i, class_name in enumerate(['Copper', 'Aluminum', 'Brass']):
    print(f'Class: {class_name}')
    print(f'  Precision: {precision[i]}')
    print(f'  Recall: {recall[i]}')
    print(f'  F1-score: {f1[i]}')
