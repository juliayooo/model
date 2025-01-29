import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

saved_model = "trained_model3.h5"
model = load_model(saved_model)
model.summary()

print("available GPUs:", len(tf.config.list_physical_devices('GPU')))

# choose directory for data
train_dir = "data2/train"
# create the img dataset
pred_dataset = image_dataset_from_directory(train_dir,
                                             image_size=(224,224),
                                             batch_size=25,
                                           labels='inferred')
# identify class names
class_names = pred_dataset.class_names
#use pyplot to show predictions and real answers

for images, labels in pred_dataset.take(1):  # Take one batch
    predictions = model.predict(images)  # Get model predictions
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10, 10))  # Create a figure with size
    for i in range(25):  # Display first 25 images
        ax = plt.subplot(5, 5, i + 1)  # 5x5 grid
        plt.imshow(images[i].numpy().astype("uint8"))  # Convert tensor to image
        plt.title(f"Pred: {class_names[predicted_labels[i]]} | Real:"
                  f" {class_names[labels[i].numpy()]}")
        plt.axis("off")  # Hide axis
    plt.show()  # Show the figure

    break

