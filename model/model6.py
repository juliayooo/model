import tensorflow as tf
import os
import torch
import torch.nn as nn
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tensorflow.keras.preprocessing import \
    image_dataset_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
import os



saved_model = "trained_model3.h5"
model = load_model(saved_model)
model.summary()

print("available GPUs:", len(tf.config.list_physical_devices('GPU')))


train_dir = "set-images-cropped-resized/train"
val_dir = "set-images-cropped-resized/val"
test_dir = "set-images-cropped-resized/test"

img_size = (224, 224)
batch_size = 64

# Run the function to remove corrupt images
def remove_corrupt_images(train_dir):
    train_dataset = image_dataset_from_directory(train_dir,
                                                 image_size=img_size,
                                                 batch_size=batch_size,
                                                 labels='inferred')



train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    labels='inferred'
)

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

model.save('trained_model6.h5')
