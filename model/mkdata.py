import os
import shutil
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk
import pandas as pd

# Global variables
current_image_index = 0
image_list = []
labels = []

# Paths
output_dir = "sorted-human-faces"
yes_dir = os.path.join(output_dir, "Yes")
no_dir = os.path.join(output_dir, "No")
metadata_file = os.path.join(output_dir, "labels.csv")

# Ensure directories exist
os.makedirs(yes_dir, exist_ok=True)
os.makedirs(no_dir, exist_ok=True)


# Load Images Function
def load_images():
    global image_list, current_image_index
    folder = filedialog.askdirectory(title="Select Image Folder")
    if folder:
        image_list = [os.path.join(folder, f) for f in
                      os.listdir(folder) if
                      f.endswith(('.png', '.jpg', '.jpeg'))]
        current_image_index = 0
        show_image()


# Show Image Function
def show_image():
    global current_image_index
    if current_image_index < len(image_list):
        img_path = image_list[current_image_index]
        img = Image.open(img_path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk
        window.title(
            f"Labeling {current_image_index + 1}/{len(image_list)}")
    else:
        save_metadata()
        window.title("Labeling Completed!")
        label.config(text="All images labeled!")


# Label Image
def label_image(label_value):
    global current_image_index
    if current_image_index < len(image_list):
        img_path = image_list[current_image_index]
        labels.append({"image": img_path, "label": label_value})

        # Move the image to the corresponding folder
        dest_dir = yes_dir if label_value == "Yes" else no_dir
        shutil.copy(img_path, dest_dir)

        # Move to next image
        current_image_index += 1
        show_image()


# Save Metadata
def save_metadata():
    if labels:
        df = pd.DataFrame(labels)
        df.to_csv(metadata_file, index=False)
        print(f"Metadata saved to {metadata_file}")


# GUI Setup
window = Tk()
window.title("Dataset Creation Tool")
label = Label(window)
label.pack()

# Buttons
btn_load = Button(window, text="Load Images", command=load_images)
btn_load.pack()

btn_yes = Button(window, text="Yes",
                 command=lambda: label_image("Yes"))
btn_yes.pack(side="left", padx=10)

btn_no = Button(window, text="No", command=lambda: label_image("No"))
btn_no.pack(side="right", padx=10)

window.mainloop()
