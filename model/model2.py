import tensorflow as tf
import os
import torch
import torch.nn as nn
# from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pathlib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


saved_model = "trained_model2.h5"
model = load_model(saved_model)
model.summary()

print("available GPUs:", len(tf.config.list_physical_devices('GPU')))
# saved_model = "trained_model.h5"
# model = torch.load(saved_model)
# model.train()


# Paths to datasets

# train_dir = r"C:\Users\24036868\Desktop\split-dataset\train"
# val_dir = r"C:\Users\24036868\Desktop\split-dataset\val"
# test_dir = r"C:\Users\24036868\Desktop\split-dataset\test"

train_dir = "data2/train"
val_dir = "data2/val"
test_dir = "data2/test"


img_size = (224, 224)
batch_size = 64
from PIL import Image
import os

def remove_corrupt_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    img.verify()  # Check if the image is corrupt
            except Exception as e:
                print(f"Removing corrupt image: {file_path} - Error: {e}")
                os.remove(file_path)  # Remove the corrupt file

# Run the function to remove corrupt images
remove_corrupt_images(train_dir)
train_dataset = image_dataset_from_directory(train_dir,
                                          image_size=img_size,
                                             batch_size=batch_size, labels='inferred')


val_dataset = image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    labels='inferred'
)


# If you also need a test dataset
test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    labels='inferred'
)


class_names = train_dataset.class_names

print(class_names)
print(train_dataset)

# Take one batch from the dataset
for images, labels in val_dataset.take(10):  # Grab the first batch
    # Access the first image and its label in the batch
    image = images[0]  # First image tensor
    label = labels[0]  # Corresponding label tensor

    # Print the tensor values for the image and its label
    # print("Image Tensor:")
    # print(image.numpy())  # Convert to numpy array for printing
    # print("\nLabel:")
    print(labels)  # Convert label tensor to numpy
    break  # Exit after processing one batch

print(tf.__version__)



model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print(train_dataset)
print(val_dataset)
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=20)

model.save('trained_model3.h5')
           