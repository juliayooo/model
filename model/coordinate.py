import os
import numpy as np
from google.cloud import vision

npz_file = "split-human-faces-resized/train/train_face_landmarks.npz"
batch_size = 200

def save_landmarks(new_data, output_file=npz_file):
    """Load existing landmark data, merge with new data, and save it."""

    # Load existing data if file exists
    if os.path.exists(output_file):
        existing_data = dict(np.load(output_file, allow_pickle=True))
    else:
        existing_data = {}

    # Update with new landmark data
    existing_data.update(new_data)

    # Save merged data back to file
    np.savez_compressed(output_file, **existing_data)
    print(f"Updated {output_file} with {len(new_data)} new entries.")


def detect_faces(image_path):
    """Detects faces and returns landmarks for an image."""
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    if response.error.message:
        raise Exception(
            f"{response.error.message}\nFor more info, check: https://cloud.google.com/apis/design/errors"
        )

    landmarks_dict = {}

    for i, face in enumerate(faces):
        # Extract (x, y) coordinates of landmarks
        landmarks = {landmark.type_: (landmark.position.x, landmark.position.y) for landmark in face.landmarks}
        landmarks_dict[f"face_{i}"] = landmarks  # Store per face

    return landmarks_dict


def process_directory(base_dir, output_file=npz_file, batch_size=200):
    """Scans all images in a directory and saves face landmarks as NumPy .npz"""
    image_landmarks = {}  # Store landmarks for all images
    image_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_path = os.path.join(root, file)
                image_paths.append(os.path.join(root, file))
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        for image_path in batch:
                try:
                    landmarks = detect_faces(image_path)
                    if landmarks:  # If faces were found
                        image_landmarks[os.path.basename(image_path)] = landmarks
                    print(f"Processing: {image_path}, {i} of 200")

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    #                 save to existing npz file
    save_landmarks(image_landmarks, output_file)
    image_landmarks.clear()



    # # Save all landmarks to .npz for fast access
    # np.save(output_file, **image_landmarks)
    # print(f"Saved {len(image_landmarks)} images with landmarks to {output_file}")
    #

# # Example Usage
# dataset_dir = "split-human-faces-resized/train/no"
# process_directory(dataset_dir,
#                   output_file=npz_file, batch_size = 200)
#

data = np.load("split-human-faces-resized/train/train_face_landmarks.npz", allow_pickle=True)
print(data["003112.png"])  # Get landmarks for "face_1.png"
print(data["004669.png"])  # Get landmarks for "face_1.png"