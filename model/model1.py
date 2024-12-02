import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pathlib

# Paths to datasets

train_dir = "resized_dataset/train"
val_dir = "resized_dataset/val"
test_dir = "resized_dataset/test"
img_size = (224, 224)
batch_size = 32

train_dataset = image_dataset_from_directory(train_dir,
                                          image_size=img_size,
                                             batch_size=batch_size, labels='inferred')

# archive = tf.keras.utils.get_file(origin=train_dir, extract=True)
# check_dir = pathlib.Path(archive)
# image_count = len(list(check_dir.glob('*/*.jpg')))
# print(image_count)
class_names = train_dataset.class_names

print(class_names)
print(train_dataset)

# Take one batch from the dataset
for images, labels in train_dataset.take(1):  # Grab the first batch
    # Access the first image and its label in the batch
    image = images[0]  # First image tensor
    label = labels[0]  # Corresponding label tensor

    # Print the tensor values for the image and its label
    print("Image Tensor:")
    print(image.numpy())  # Convert to numpy array for printing
    print("\nLabel:")
    print(label.numpy())  # Convert label tensor to numpy
    break  # Exit after processing one batch

print(tf.__version__)

