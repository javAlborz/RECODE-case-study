
import pandas as pd
from PIL import Image
import os

csv_path = 'styles.csv'
source_dir = 'images'
target_dir = 'images_pp'

# Load the dataset information from the CSV
df = pd.read_csv(csv_path)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Loop through the DataFrame and process each image
for index, row in df.iterrows():
    image_filename = f"{row['id']}.jpg"  
    source_image_path = os.path.join(source_dir, image_filename)
    target_image_path = os.path.join(target_dir, image_filename)

    # Open and resize the image
    try:
        with Image.open(source_image_path) as img:
            img_resized = img.resize((60, 80), Image.Resampling.LANCZOS)
            img_resized.save(target_image_path)
    except IOError as e:
        print(f"Error processing image {image_filename}: {e}")

print("Image preprocessing completed.")
