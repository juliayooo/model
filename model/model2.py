import tensorflow as tf
import os
import torch
import torch.nn as nn
# from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_landmarks(landmark_file):
    landmark_data=np.load(landmark_file, allow_pickle=True)
    return dict(landmark_data)


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


def proc_image_w_landmarks(image_path, landmark_dict):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalize

    # Get corresponding landmarks
    image_filename = os.path.basename(image_path)
    if image_filename in landmark_dict:
        landmarks = landmark_dict[image_filename]
    else:
        landmarks = np.zeros(
            (10, 2))  # Default zero landmarks if missing

    return img, landmarks




saved_model = "/Users/juliayoo/Desktop/MODEL-ITERATIONS/trained_model3.h5"
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

train_dir = "split-human-faces-resized/train"
val_dir = "split-human-faces-resized/val"
test_dir = "split-human-faces-resized/test"


img_size = (224, 224)
batch_size = 64

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

model.save('trained_model4.h5')
           