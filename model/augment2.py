import tensorflow as tf
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
import os
import albumentations as A
import numpy as np
import cv2 as cv2


# AUGMENTATION PIPELINE (CROP and FOG )
transform = A.Compose([
    A.RandomResizedCrop(size=(224,224), ratio=(0.7, 1.3),
                        p = 0.9),
    # A.RandomFog()
])

def augment(image, label):
    image = image.numpy()  # Convert tensor to NumPy array

    image = (image * 255).astype(np.uint8)

    print(f"Image shape before augmentation: {image.shape}")

    # ✅ Ensure the image has 3 channels (convert grayscale to RGB)
    if len(image.shape) == 2:  # Grayscale image (shape: (height, width))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        print("FOUND 2D IMAGE ARRAY")

    if image is None or image.size == 0:
        print(
            "Warning: Encountered an empty image. Skipping augmentation.")
        return tf.convert_to_tensor(
            np.zeros((224, 224, 3), dtype=np.float32)), label

    augmented = transform(image=image)
    image = augmented["image"].astype(np.float32) / 255.0  # Normalize back

    return image, label

# Wrapper function to use with tf.data.Dataset
def tf_augment_fn(image, label):
    image, label = tf.py_function(func=augment,
                                  inp=[image, label],
                                  Tout=[tf.float32, tf.int32])
    image.set_shape((224, 224, 3))  # Fix shape issues
    label.set_shape([])  # Ensure label remains a scalar
    return image, label


                                # # Read an image with OpenCV and convert it to the RGB colorspace
                                # image = cv2.imread("image.jpg")
                                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                #
                                # # Augment an image
                                # transformed = transform(image=image)
                                # transformed_image = transformed["image"]


def remove_corrupt_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    img.verify()  # Check if the image is corrupt
            except Exception as e:
                print(
                    f"Removing corrupt image: {file_path} - Error: {e}")
                os.remove(file_path)  # Remove the corrupt file




saved_model = ("trained_model4.h5")
model = load_model(saved_model)
model.summary()

print("available GPUs:", len(tf.config.list_physical_devices('GPU')))

train_dir = "split-dataset/train"
val_dir = "split-dataset/val"

img_size = (224, 224)
batch_size = 32

# Run the function to remove corrupt images
remove_corrupt_images(train_dir)



train_dataset = image_dataset_from_directory(train_dir,
                                             image_size=img_size,
                                             batch_size=batch_size,
                                             labels='inferred')

train_dataset = train_dataset.map(tf_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
# train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # Optimize performance



val_dataset = image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    labels='inferred'
)

val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# class_names = train_dataset.class_names
#
# print(class_names)
print(train_dataset)


print(tf.__version__)

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(train_dataset)
print(val_dataset)
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=20)

model.save('trained_model_postaug.h5')
