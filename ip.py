# process iphone's photo to jpg fomat and 1056 size

from PIL import Image
import pillow_heif
import os

def convert_and_resize_images(input_folder, output_folder, target_size=(1056, 1056)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(".heic"):
                heif_file = pillow_heif.read_heif(file_path)
                image = Image.frombytes(
                    heif_file.mode, 
                    heif_file.size, 
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    heif_file.stride,
                )
                # Convert HEIC to JPEG
                base_name = os.path.splitext(file)[0]
                output_path = os.path.join(output_folder, f"{base_name}.jpg")
                image = image.convert("RGB")
                image.save(output_path, "JPEG")

                # Resize image
                resized_img = image.resize(target_size, Image.LANCZOS)
                resized_output_path = os.path.join(output_folder, f"{base_name}_resized.jpg")
                resized_img.save(resized_output_path)
            elif file.lower().endswith((".jpg", ".jpeg", ".png")):
                with Image.open(file_path) as img:
                    # Resize image
                    resized_img = img.resize(target_size, Image.LANCZOS)
                    base_name = os.path.splitext(file)[0]
                    resized_output_path = os.path.join(output_folder, f"{base_name}_resized.jpg")
                    resized_img.save(resized_output_path)

input_folder = 'C:/Users/96156/Desktop/background'
output_folder = 'C:/Users/96156/Desktop/Back'
convert_and_resize_images(input_folder, output_folder)
