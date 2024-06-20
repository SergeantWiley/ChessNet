# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def red_mask(image_path):
#     # Read the original image
#     image = cv2.imread(image_path)
    
#     # Check if the image was loaded properly
#     if image is None:
#         print("Error: Could not read the image.")
#         return
    
#     # Convert the image to the RGB color space
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Define the lower and upper boundaries for the red color in the RGB space
#     lower_red = np.array([149, 0, 0])
#     upper_red = np.array([255, 150, 150])
    
#     # Create a mask that identifies the red regions
#     mask = cv2.inRange(image_rgb, lower_red, upper_red)
    
#     # Create an output image where red areas are white and other areas are black
#     output_image = np.zeros_like(image_rgb)
#     output_image[mask > 0] = [255, 255, 255]
    
#     # Display the original and the red-mask images using Matplotlib
#     plt.figure(figsize=(10, 5))
    
#     # Display the original image
#     plt.subplot(1, 2, 1)
#     plt.imshow(image_rgb)
#     plt.title('Original Image')
#     plt.axis('off')
    
#     # Display the red mask image
#     plt.subplot(1, 2, 2)
#     plt.imshow(output_image)
#     plt.title('Red Mask Image')
#     plt.axis('off')
    
#     # Show the plots
#     plt.show()


# # Example usage
# input_image_path = 'FOVImages\image63.png'
# #output_image_path = 'path/to/save/grayscale_image.jpg'
# red_mask(input_image_path)

import cv2
import numpy as np
import matplotlib.pyplot as plt

def red_outline_and_mask(image_path):
    # Read the original image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded properly
    if image is None:
        print("Error: Could not read the image.")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lower_red = np.array([149, 0, 0])
    upper_red = np.array([255, 150, 150])
    
    mask = cv2.inRange(image_rgb, lower_red, upper_red)
    green_channel = image_rgb[:, :, 1]
    blue_channel = image_rgb[:, :, 2]
    exclude_mask = (green_channel > 100) & (blue_channel > 100)
    mask[exclude_mask] = 0
    red_regions = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    gray_red_regions = cv2.cvtColor(red_regions, cv2.COLOR_RGB2GRAY)
    
    edges = cv2.Canny(gray_red_regions, 100, 200)
    
    outline_image = image_rgb.copy()
    outline_image[edges > 0] = [255, 255, 255]
    
    mask_image = np.zeros_like(gray_red_regions)
    mask_image[edges > 0] = 255

    plt.figure(figsize=(15, 5))
    
    # Display the original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Display the outlined image
    plt.subplot(1, 3, 2)
    plt.imshow(outline_image)
    plt.title('Outlined Red Regions')
    plt.axis('off')
    
    # Display the mask image
    plt.subplot(1, 3, 3)
    plt.imshow(mask_image, cmap='gray')
    plt.title('Black and White Mask')
    plt.axis('off')
    
    # Show the plots
    plt.show()

# Example usage
input_image_path = 'FOVImages/image123.png'
red_outline_and_mask(input_image_path)
