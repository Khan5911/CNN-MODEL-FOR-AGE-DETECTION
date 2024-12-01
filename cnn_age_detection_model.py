

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset (Assuming you have the images and labels in specific folders)
def load_data(image_dir, label_file):
    data = []
    labels = pd.read_csv(label_file)  # Age labels
    images = os.listdir(image_dir)

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64, 64))  # Resize to 64x64 pixels
        data.append(image)

    data = np.array(data)
    labels = np.array(labels['age'])

    return data, labels

# Normalize and preprocess
def preprocess_data(data, labels):
    data = data / 255.0  # Normalize image pixel values
    labels = to_categorical(labels, num_classes=100)  # Assuming age range is 0-100
    return data, labels

# Define the CNN model
def create_cnn_model(input_shape):
    model = Sequential()

    # First Convolutional Layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional Layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer for Age Prediction (Assuming age range 0-100)
    model.add(Dense(100, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training the CNN Model
def train_model(image_dir, label_file):
    X, y = load_data(image_dir, label_file)
    X, y = preprocess_data(X, y)

    input_shape = (64, 64, 3)
    cnn_model = create_cnn_model(input_shape)

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    cnn_model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    return cnn_model

if __name__ == "__main__":
    image_dir = '/path/to/images'  # Replace with the path to your image folder
    label_file = '/path/to/labels.csv'  # Replace with the path to your labels CSV file
    model = train_model(image_dir, label_file)
