import numpy as np
from PIL import Image, ImageOps
import cv2
from PIL.ImageQt import ImageQt
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import QFileDialog



def adjust_contrast(image, min_color, max_color, contrast_factor):
        """ Adjusts the contrast of an image by applying a contrast factor within a specified color range.
        :param image: The input image.
        :param min_color: The minimum color value to apply the contrast factor to.
        :param max_color: The maximum color value to apply the contrast factor to.
        :param contrast_factor: The factor to adjust the contrast by.
        :return: The adjusted image.
        """
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
    """ Scales the pixel values of an image to a specified output range.
    :param image: The input image.
    :param input_min: The minimum pixel value in the input image.
    :param input_max: The maximum pixel value in the input image.
    :param output_min: The minimum pixel value in the output image.
    :param output_max: The maximum pixel value in the output image.
    :return: The scaled image."""
    img_array = image.convert("L")  # Convert to grayscale

    # Scale pixel values within the specified input range to the output range
    scaled_img_array = (img_array - input_min) * ((output_max - output_min) / (input_max - input_min)) + output_min

    # Clip values to ensure they are within the output range
    scaled_img_array = scaled_img_array.clip(output_min, output_max)

    # Convert back to image
    scaled_image = Image.fromarray(scaled_img_array.astype('uint8'))

    return scaled_image

# Invert the colors of an image
# image: The input image
# Returns the inverted image
def invert_image(image):
     """ Inverts the colors of an image.
        :param image: The input image.
        :return: The inverted image.
        """
     
     image = ImageOps.invert(image)
     return image

#
def calculate_normal_map(height_map):
    """ Calculate the normal map from a height map using the sobel method.
    :param height_map: The height map image.
    :return: The normal map image.
    """
    # Compute gradients in the x and y directions
    sobel_x = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate the z component of the normal (assuming depth is 1 for all normals)
    z = np.ones_like(sobel_x)
    
    # Normalize the gradients to create normal vectors
    normal_map = np.dstack((sobel_x, sobel_y, z))
    norms = np.linalg.norm(normal_map, axis=2)
    normal_map_normalized = normal_map / norms[:,:,None]
    
    # Map components from [-1, 1] to [0, 255]
    normal_map_normalized = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)
    
    return normal_map_normalized

def display_image(image, imageLabel):
    """ Display an image in a QLabel.
    :param image: The image to display.
    :param imageLabel: The QLabel to display the image in.
    :return: The displayed image.
    """
    if isinstance(image, np.ndarray):
        image = numpy_to_PIL(image)
    pixmap = QPixmap.fromImage(ImageQt(image))
    imageLabel.setMaximumWidth(400)
    imageLabel.setPixmap(pixmap)
    imageLabel.setMaximumHeight(400)
    return image


def numpy_to_PIL(numpy_image):
    """ Convert a numpy array to a PIL Image.
    :param numpy_image: The input numpy array.
    :return: The converted PIL Image.
    """

    # Validate the input is at least 2-dimensional
    if numpy_image.ndim < 2:
        raise ValueError("Expected an array with at least 2 dimensions.")

    # Handle grayscale (2D) or color (3D) images
    if numpy_image.ndim == 2:
        # Directly handle 2D grayscale images
        return handle_grayscale(numpy_image)
    elif numpy_image.ndim == 3:
        # Check for channel location based on the size and number of channels
        if numpy_image.shape[0] in [1, 3, 4] and all(dim > 3 for dim in numpy_image.shape[1:]):
            # This is likely C, H, W format
            return handle_color(numpy_image, channel_first=True)
        elif numpy_image.shape[2] in [1, 3, 4] and all(dim > 3 for dim in numpy_image.shape[:2]):
            # This is likely H, W, C format
            return handle_color(numpy_image, channel_first=False)
        else:
            raise ValueError("3D array does not meet expected channel configurations.")
    else:
        raise ValueError("Unsupported array dimensions; expected 2D or 3D array.")

def handle_grayscale(numpy_image):
    """ Handle grayscale images."""
    # Normalize and convert if necessary
    if numpy_image.dtype == np.float32 or numpy_image.dtype == np.float64:
        numpy_image = (numpy_image * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(numpy_image, 'L')

def handle_color(numpy_image, channel_first):
    """ Handle color images."""
    if channel_first:
        # Convert C, H, W to H, W, C
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
    if numpy_image.dtype == np.float32 or numpy_image.dtype == np.float64:
        # Normalize and convert to uint8
        numpy_image = (numpy_image * 255).clip(0, 255).astype(np.uint8)
    
    # Determine the correct color mode
    if numpy_image.shape[2] == 1:
        return Image.fromarray(numpy_image.squeeze(), 'L')  # Grayscale with one channel
    elif numpy_image.shape[2] == 3:
        return Image.fromarray(numpy_image, 'RGB')  # RGB
    elif numpy_image.shape[2] == 4:
        return Image.fromarray(numpy_image, 'RGBA')  # RGBA
    else:
        raise ValueError("Unsupported number of channels.")


def load_image(file_path, imageLabel):
    """ Load an image from a file path and display it in a QLabel."""
    image = Image.open(file_path)
    display_image(image, imageLabel)
    return image

