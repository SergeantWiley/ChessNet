import os
from PIL import Image
def slice_image(image_path, output_dir, parts=5):
    img = Image.open(image_path)
    img_width, img_height = img.size
    part_width = img_width // parts
    for i in range(parts):
        left = i * part_width
        right = (i + 1) * part_width if i < parts - 1 else img_width
        img_part = img.crop((left, 0, right, img_height))
        
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_part_{i + 1}{ext}")
        img_part.save(output_path)
    
def process_directory(input_dir, output_dir, parts=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_dir, filename)
            slice_image(image_path, output_dir, parts)
            print(f"{filename} proccessed")    
input_dir = 'MaskImages'
output_dir = 'SliceImages'
os.makedirs(output_dir, exist_ok=True)
process_directory(input_dir, output_dir)