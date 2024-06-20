import os
import pandas as pd
from PIL import Image
import shutil
output_dir = 'FilterImages'
filter_file = 'annotations.csv'
input_dir = 'MaskImages'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(filter_file)
images = df['image_name'].tolist()

for image in images:
    source_path = os.path.join(input_dir,image)
    destination_path = os.path.join(output_dir, image)
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"Copied: {image}")
    else:
        print(f"Image not found: {image}")
