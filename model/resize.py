from PIL import Image
import os

# Set paths
input_dir = "dataset"  # Original dataset directory containing 'yes' and 'no'
output_dir = "resized_dataset"  # Directory to save resized images
target_size = (224, 224)  # Desired image size (width, height)

def resize_images(input_dir, output_dir, target_size):
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
        for filename in os.listdir(input_category_dir):
            input_path = os.path.join(input_category_dir, filename)
            output_path = os.path.join(output_category_dir, filename)

            try:
                # Open, resize, and save the image
                with Image.open(input_path) as img:
                    img = img.resize(target_size)
                    img.save(output_path)
                    print(f"Resized and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

# Call the function
resize_images(input_dir, output_dir, target_size)
