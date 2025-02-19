from PIL import Image
import os

# Set paths
input_dir = "set_to_crop"  # Original dataset directory
# containing
# 'yes' and 'no'
output_dir = "set-images-cropped"  # Directory to save
# resized
# images
target_size = (224, 224)  # Desired image size (width, height)

def crop_images(input_dir, output_dir, target_size):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through the 'yes' and 'no' folders
    for category in ["yes", "no"]:
        input_category_dir = os.path.join(input_dir, category)
        output_category_dir = os.path.join(output_dir, category)

        # Create category folder in output directory
        os.makedirs(output_category_dir, exist_ok=True)

        # Process each image in the category
        print(input_category_dir)
        for filename in os.listdir(input_category_dir):
            print(filename)
            if 'Store' not in filename:
                print(filename)
                input_path = os.path.join(input_category_dir, filename)
                output_path = os.path.join(output_category_dir, filename)

                try:
                    # Open, resize, and save the image
                    with Image.open(input_path) as img:
                        print(output_path)
                        box = (17,37,162,182)

                        img2 = img.crop(box)

                        img2.save(output_path)

                        print(f"Resized and saved: {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

# Call the function
crop_images(input_dir, output_dir, target_size)
