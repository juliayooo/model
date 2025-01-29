import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pathlib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
print("available GPUs:", len(tf.config.list_physical_devices('GPU')))

# Paths to datasets

# train_dir = r"C:\Users\24036868\Desktop\split-dataset\train"
# val_dir = r"C:\Users\24036868\Desktop\split-dataset\val"
# test_dir = r"C:\Users\24036868\Desktop\split-dataset\test"

train_dir = r"resized_dataset\train"
val_dir = r"resized_dataset\\val"
test_dir = r"resized_dataset\\test"


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


#  THIS FUNCTION CREATES THE DATA TENSOR ^^^^

# PRETRAINED MODEL RESNET50
pt_model = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(224, 224, 3))

# FREEZE THE BASE MODEL's LAYERS SO MY DATA DOESN'T CHANGE THEM
pt_model.trainable = False

headModel = pt_model.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(images), activation="softmax")(headModel)

model = Model(inputs=pt_model.input, outputs=headModel)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

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
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=20)

model.save('trained_model.h5')
           