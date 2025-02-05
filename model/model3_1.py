import os
import numpy as np
import tensorflow as tf
import google.cloud
import cv2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras import layers, Model
def landmark_input_layer(input_shape):
    return Input(shape=input_shape, name="landmarks")

def load_landmark_data(landmark_file):
    """Load facial landmark data from an NPZ file."""
    if os.path.exists(landmark_file):
        return dict(np.load(landmark_file, allow_pickle=True))
    return {}  # Return empty dict if file not found



def process_image_with_landmarks(image_path, landmark_dict):
    """Loads an image and fetches its corresponding landmarks."""
    # Load image
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalize

    # Fetch landmark data
    image_filename = os.path.basename(image_path)
    landmarks = landmark_dict.get(image_filename, np.zeros((10, 2)))  # Default to zeros if missing

    return img, landmarks

def create_dataset(image_dir, landmark_file, batch_size=32, expected_landmark_shape=(68, 2)):
    # """Creates a dataset combining images and landmark features."""
    # landmark_dict = load_landmark_data(landmark_file)
    #
    # image_paths = []
    # labels = []
    # landmarks = []
    #
    # for root, _, files in os.walk(image_dir):
    #     for file in files:
    #         if file.lower().endswith((".jpg", ".jpeg", ".png")):
    #             image_path = os.path.join(root, file)
    #             image_paths.append(image_path)
    #
    #             # Assign label based on folder structure
    #             label = 1 if "yes" in root else 0
    #             labels.append(label)
    #
    #             # Process image & landmarks
    #             _, landmark_points = process_image_with_landmarks(image_path, landmark_dict)
    #             landmarks.append(landmark_points)
    #
    # # Convert lists to NumPy arrays
    # images = np.array([process_image_with_landmarks(img, landmark_dict)[0] for img in image_paths])
    # labels = np.array(labels)
    # landmarks = np.array(landmarks)
    #
    # dataset = tf.data.Dataset.from_tensor_slices(({"image": images, "landmarks": landmarks}, labels))
    # dataset = dataset.batch(batch_size).shuffle(1000)
    #
    # return dataset
    """Loads images and their corresponding landmarks, ensuring uniform shape."""
       # Load landmark data
    data = np.load(landmark_file, allow_pickle=False)

    images = []
    landmarks_list = []
    labels = []

    for img_name in data.files:
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue  # Skip missing images

        # Load & resize image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalize
        images.append(img)

        # Load landmarks
        landmarks = data[img_name]
        if isinstance(landmarks, dict):
            landmarks = np.array(list(landmarks.values()))  # Convert dict to array
        else:
            landmarks = np.array(landmarks)

        # Fix landmark shape
        if landmarks.shape != expected_landmark_shape:
            padded_landmarks = np.zeros(expected_landmark_shape)
            min_points = min(expected_landmark_shape[0], landmarks.shape[0])
            padded_landmarks[:min_points] = landmarks[:min_points]  # Copy available points
            landmarks = padded_landmarks

        landmarks_list.append(landmarks)

        # Assign label based on folder name
        label = 1 if "yes" in image_dir.lower() else 0
        labels.append(label)

    # Convert lists to NumPy arrays
    images = np.array(images)
    landmarks_list = np.array(landmarks_list)
    labels = np.array(labels)

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(({"input_1": images,
                                                   "landmarks": landmarks_list}, labels))

    # Shuffle and batch
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

### ---- Load Pre-trained Model and Modify ---- ###
saved_model_path = "/Users/juliayoo/Desktop/MODEL-ITERATIONS/trained_model3.h5"
model = load_model(saved_model_path, custom_objects={"landmark_input_layer": landmark_input_layer})
model.summary()

# Add additional input for landmarks
image_input = model.input  # Original image input
landmark_input = layers.Input(shape=(10, 2), name="landmarks")  # Landmark input

# Process landmark input
x = layers.Flatten(name="landmark_input")(landmark_input)
x = layers.Dense(32, activation="relu", name="landmark_dense_1")(x)

# Merge with image model's last layer
merged = layers.concatenate([model.output, x], name="merged_layers")
merged = layers.Dense(64, activation="relu", name="merged_dense_1")(merged)
output = layers.Dense(1, activation="sigmoid",  name="output")(merged)

# Define new model with both inputs
updated_model = Model(inputs=[image_input, landmark_input], outputs=output)

# Compile new model
updated_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
updated_model.summary()

#
# ### ---- Load Datasets and Fine-Tune Model ---- ###
# train_dataset = create_dataset("split-human-faces-resized/train/",
#                                "/Users/juliayoo/PycharmProjects/Learning0/model/split-human-faces-resized/train/train_face_landmarks.npz")
# val_dataset = create_dataset("split-human-faces-resized/val/",
#                              "/Users/juliayoo/PycharmProjects/Learning0/model/split-human-faces-resized/val/val_face_landmarks.npz")
#
# updated_model.fit(train_dataset, validation_data=val_dataset, epochs=10)

train_dataset = create_dataset("split-human-faces-resized/train/",
                               "/Users/juliayoo/PycharmProjects/Learning0/model/split-human-faces-resized/train/train_face_landmarks.npz")

val_dataset = create_dataset("split-human-faces-resized/val/",
                             "/Users/juliayoo/PycharmProjects/Learning0/model/split-human-faces-resized/val/val_face_landmarks.npz")

# Train model
updated_model.fit(train_dataset, validation_data=val_dataset, epochs=10)



model.save('trained_model4.h5')

