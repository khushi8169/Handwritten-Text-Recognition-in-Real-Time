import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Fix encoding issues for console output
sys.stdout.reconfigure(encoding='utf-8')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

charset = [str(i) for i in range(657)]
num_classes = len(charset)

# Dataset paths
ENGLISH_TRAIN_DIR = r'c:\Users\hp\OneDrive\Desktop\miner\datasets\data_english\data'
HINDI_TRAIN_DIR = r'c:\Users\hp\OneDrive\Desktop\miner\datasets\data_hindi\Train'

# Function to load images from directories
def load_images_from_directory(directory, target_size, color_mode='grayscale'):
    """Load images from a directory into numpy arrays."""
    images = []
    labels = []
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            label = int(folder)  # Assuming folder names are numeric labels (e.g., 0, 1, 2...)
            for filename in os.listdir(folder_path):
                filepath = os.path.join(folder_path, filename)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(filepath):
                    img = load_img(filepath, target_size=target_size, color_mode=color_mode)
                    img_array = img_to_array(img) / 255.0
                    images.append(img_array)
                    labels.append(label)
    return np.array(images), np.array(labels)

# Preprocessing for English
def load_english_data():
    target_size = (200, 200)  # Resize to smaller if needed
    train_images, train_labels = load_images_from_directory(ENGLISH_TRAIN_DIR, target_size)
    train_labels = to_categorical(train_labels, num_classes=672)  # Update classes if needed
    return train_images, train_labels

# Preprocessing for Hindi
def load_hindi_data():
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_data = train_datagen.flow_from_directory(
        HINDI_TRAIN_DIR,
        target_size=(32, 128),
        color_mode='grayscale',
        batch_size=16,  # Adjust to avoid memory warnings
        class_mode='categorical'
    )
    return train_data

# CTC Loss Function
def ctc_loss(y_true, y_pred):
    return K.ctc_batch_cost(y_true, y_pred)

# CNN + RNN Model Creation
def create_ctc_model(num_classes, input_shape):
    inputs = Input(shape=input_shape)
    
    # CNN Layers
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Reshape for RNN
    new_shape = (-1, x.shape[1], x.shape[2], 64)  # Change channels to 64
    x = TimeDistributed(Flatten())(x)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# Training Function
def train():
    # Define model save paths
    english_model_path = 'iam_model.keras'
    hindi_model_path = 'devanagari_model.keras'
   
    print("Loading English dataset...")
    try:
        # Load and preprocess English dataset
        train_images, train_labels = load_english_data()
       
        # Build and train the English model
        english_model = create_ctc_model(num_classes=672, input_shape=(200, 200, 1))
        english_model.fit(
            train_images, train_labels,
            epochs=20,
            batch_size=16,
            validation_split=0.2,
            verbose=1
        )
       
        # Save the trained English model
        english_model.save(english_model_path)
        print(f"English model saved as '{english_model_path}'.")
    except Exception as e:
        print(f"Error with English dataset: {e}")

    print("Loading Hindi dataset...")
    try:
        # Load and preprocess Hindi dataset
        hindi_train_data = load_hindi_data()
       
        # Build and train the Hindi model
        hindi_model = create_ctc_model(num_classes=hindi_train_data.num_classes, input_shape=(32, 128, 1))
        hindi_model.fit(
            hindi_train_data,
            epochs=10,
            verbose=1
        )
       
        # Save the trained Hindi model
        hindi_model.save(hindi_model_path)
        print(f"Hindi model saved as '{hindi_model_path}'.")
    except Exception as e:
        print(f"Error with Hindi dataset: {e}")  

if __name__ == "__main__":
    train()

