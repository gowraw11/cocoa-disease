from PIL import Image
import os

# Directory containing non-cocoa leaf images
input_dir = '/Users/deekshithsy/Desktop/cocoa-disease/dataset/validation/cocoa'  # Update this path
# Directory to save resized images
output_dir = '/Users/deekshithsy/Desktop/cocoa-disease/resize/validation/cocoa'  # Update this path

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to resize images
def resize_image(input_path, output_path, size=(256, 256)):
    with Image.open(input_path) as img:
        resized_img = img.resize(size, Image.LANCZOS)  # Use Image.LANCZOS for high-quality resizing
        
        # Convert image to 'RGB' mode if it's in 'P' mode
        if resized_img.mode == 'P':
            resized_img = resized_img.convert('RGB')
        
        resized_img.save(output_path)

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        resize_image(input_path, output_path)

print("All images have been resized and saved to", output_dir)
