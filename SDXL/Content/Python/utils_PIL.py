import numpy as np
from PIL import Image, ImageOps


def adjust_contrast(image, min_color, max_color, contrast_factor):
        img_array = np.array(image)

        # Normalize pixel values
        normalized_img = img_array / 255.0

        # Centering around center_point
        centered_img = normalized_img - .5

        # Apply contrast adjustment within the color range
        contrast_adjusted_img = centered_img.copy()
        contrast_adjusted_img[(img_array >= min_color) & (img_array <= max_color)] *= contrast_factor

        # Re-centering
        adjusted_img = contrast_adjusted_img + .5

        # Denormalize pixel values
        adjusted_img = np.clip(adjusted_img * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(adjusted_img)


def scale_pixel_range(image, input_min, input_max, output_min, output_max):
    img_array = image.convert("L")  # Convert to grayscale

    # Scale pixel values within the specified input range to the output range
    scaled_img_array = (img_array - input_min) * ((output_max - output_min) / (input_max - input_min)) + output_min

    # Clip values to ensure they are within the output range
    scaled_img_array = scaled_img_array.clip(output_min, output_max)

    # Convert back to image
    scaled_image = Image.fromarray(scaled_img_array.astype('uint8'))

    return scaled_image

def invert_image(image):
     image = ImageOps.invert(image)
     return image

