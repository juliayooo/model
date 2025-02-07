import tensorflow as tf
import os
import numpy as np
import cv2
import albumentations as A
from tensorflow.keras.preprocessing import \
    image_dataset_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt

# Define the augmentation pipeline.
# (You can add more transforms if desired.)
transform = A.Compose([
    A.RandomFog(p=1.0),
    A.RandomResizedCrop(size=(224,224), scale=(0.85, 1.0), ratio=(
        0.8, 1.3),
                        p = 0.9),
])


def augment(batch_images, label):
    """
    Expects batch_images to have shape (batch_size, 224, 224, 3)
    and applies the augmentation individually.
    """
    # Convert tensor batch to numpy array and scale to [0, 255]
    batch_images = batch_images.numpy()
    batch_images = batch_images.astype(np.uint8)

    augmented_images = []
    # Process each image in the batch individually.
    for i, img in enumerate(batch_images):
        # Ensure the image is contiguous (helps OpenCV functions)
        img = np.ascontiguousarray(img)

        # If the image is not 3-channel, try to convert it.
        if len(img.shape) == 2 or (
                len(img.shape) == 3 and img.shape[-1] != 3):
            try:
                # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                print(f"Converted grayscale to RGB for image {i}")
            except Exception as e:
                print(f"Error converting image {i} to RGB: {e}")

        # Optionally, you can verify the image shape:
        if img.shape[0] != 224 or img.shape[1] != 224:
            print(
                f"Warning: Image {i} shape is {img.shape} (expected (224,224,3)). Attempting to resize.")
            img = cv2.resize(img, (224, 224))

        # Apply augmentation with a try/except block
        try:
            augmented = transform(image=img)
            augmented_img = augmented["image"]
        except Exception as e:
            print(f"Error augmenting image {i}: {e}")
            augmented_img = img  # Fallback: use the original image

        augmented_images.append(augmented_img)

    # Stack the augmented images back into a batch.
    augmented_images = np.stack(augmented_images)
    # Normalize the images back to [0, 1] and convert to tensor.
    augmented_images = augmented_images.astype(np.float32) / 255.0
    return tf.convert_to_tensor(augmented_images), label


def tf_augment_fn(images, labels):
    # Use tf.py_function to call the numpy-based augmentation.
    aug_images, labels = tf.py_function(func=augment,
                                        inp=[images, labels],
                                        Tout=[tf.float32, tf.int32])
    # Set static shape for TensorFlow.
    aug_images.set_shape((None, 224, 224, 3))
    labels.set_shape((None,))
    return aug_images, labels


def remove_corrupt_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(
                    f"Removing corrupt image: {file_path} - Error: {e}")
                os.remove(file_path)


# Paths and parameters.
train_dir = "full-split-human-faces-resized/train"
val_dir = "full-split-human-faces-resized/val"
img_size = (224, 224)
batch_size = 32

# Remove corrupt images.
remove_corrupt_images(train_dir)

# Load datasets (these are already batched).
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

# Apply the augmentation function to the training dataset.
# IMPORTANT: Do not re-batch if the dataset is already batched.
train_dataset = train_dataset.map(tf_augment_fn,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Load your model.
saved_model = "trained_model4.h5"
model = load_model(saved_model)
model.summary()
print("available GPUs:", len(tf.config.list_physical_devices('GPU')))

model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Inspect one batch to confirm shapes.
for images, labels in train_dataset.take(1):
    print("Augmented batch shape:",
          images.shape)  # Should be (batch_size, 224, 224, 3)

# Retrieve the first batch from your training dataset
for batch_images, batch_labels in train_dataset.take(1):
    # Convert TensorFlow tensors to numpy arrays (if needed)
    images_np = batch_images.numpy()

    # Create a figure to display the first 10 images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(10):
        # Show the image; adjust if necessary (e.g., if your images are normalized [0,1])
        axes[i].imshow(images_np[i])
        axes[i].set_title(f"Aug {i + 1}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()  # This will block execution until the plot window is closed
    break  # Only need to process the first batch



# Train the model.
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=10)

model.save('trained_model_postaug.h5')
