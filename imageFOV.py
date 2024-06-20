import cv2
import os

def crop_center(image, crop_width, crop_height):
    # Get the dimensions of the image
    height, width, _ = image.shape

    center_x, center_y = width // 2, height // 2

    x1 = center_x - crop_width // 2
    x2 = center_x + crop_width // 2
    y1 = center_y - crop_height // 2
    y2 = center_y + crop_height // 2

    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def process_images(input_dir, output_dir, crop_size):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Check if the image was loaded properly
            if image is None:
                print(f"Error: Could not read the image {filename}.")
                continue

            cropped_image = crop_center(image, crop_size[0], crop_size[1])

            # Save the cropped image to the output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cropped_image)
            print(f"Cropped image saved to {output_path}")

# Example usage
input_directory = 'RawImages'
output_directory = 'FOVImages'
crop_size = (1920, 400)  # Width x Height of the crop region
process_images(input_directory, output_directory, crop_size)
