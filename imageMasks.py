import cv2
import numpy as np
import os

def red_mask_intensity(image_path, output_dir):
    # Read the original image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read the image {image_path}.")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 100, 100])
    
    mask = cv2.inRange(image_rgb, lower_red, upper_red)
                           
    green_channel = image_rgb[:, :, 1]
    blue_channel = image_rgb[:, :, 2]
 
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
    gray_mask_float = gray_mask.astype(np.float32)

    gray_mask_float /= 255.0

    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    output_path = os.path.join(output_dir, filename + '.jpg')
    cv2.imwrite(output_path, (gray_mask_float * 255).astype(np.uint8))
    print(f"Grayscale intensity mask saved to {output_path}")

input_dir = 'RawImages'
output_dir = 'MaskImages'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        red_mask_intensity(image_path, output_dir)
