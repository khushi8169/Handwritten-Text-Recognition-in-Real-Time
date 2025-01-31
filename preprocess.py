import cv2
import numpy as np
import os


def preprocess_image(image_path, for_english):
    """
    Preprocess the image for model input.
    - Resize based on the model's expected input dimensions.
    - Normalize and expand dimensions for channel input.
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read the image at {image_path}. Ensure it exists and is a valid image.")


    # Resize based on model requirements
    if for_english:
        target_size = (200, 200)  # Match the English model's expected size
    else:
        target_size = (32, 128)  # Match the Hindi model's expected size


    # Resize and ensure correct dimension ordering
    img = cv2.resize(img, target_size)


    # Normalize pixel values to [0, 1]
    img = img.astype('float32') / 255.0


    # Add channel dimension for grayscale images
    img = np.expand_dims(img, axis=-1)


    # For Hindi model, ensure shape is (32, 128, 1)
    if not for_english:
        img = img.transpose((1, 0, 2))  # Swap dimensions if needed


    return img


