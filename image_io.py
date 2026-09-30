import os

import numpy as np

# OpenCV only reads OpenEXR files if this variable is set before cv2 is
# imported. Setting it after the import has no effect, and cv2.imread then
# fails with "OpenEXR codec is disabled". For this reason the other modules
# in this project do not import cv2 themselves; they use the functions here.
# Background: https://github.com/opencv/opencv/issues/21326
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2


def load_hdr_image(file_path):
    """
    Load an HDR image, such as an OpenEXR (.exr) file, as a float32 RGB array.

    The values are linear: each value is proportional to the amount of light
    in the scene, and values above 1 are allowed. OpenCV returns the channels
    in BGR order, so they are reordered to RGB.

    Negative values are set to 0. HDR files can contain small negative
    values (Desk.exr has some down to about -0.012) from noise or colours
    outside the RGB gamut. Negative light has no physical meaning, and a
    negative luminance would make the logarithms in the tone mapping
    operators undefined.
    """
    image = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(f"Could not load HDR image: {file_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.maximum(image, 0.0).astype(np.float32)


def load_sdr_image(file_path):
    """
    Load an 8 bit SDR image, such as a PNG file, as a float32 RGB array with
    values from 0 to 1.

    The values stay in the sRGB encoding they were saved with; they are not
    converted to linear light. Any alpha channel is dropped.
    """
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load SDR image: {file_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def encode_srgb(linear_image):
    """
    Convert linear values from 0 to 1 to sRGB encoded values from 0 to 1.

    This is the sRGB transfer function of IEC 61966-2-1:1999:

        V = 12.92 * L                        if L <= 0.0031308
        V = 1.055 * L ** (1 / 2.4) - 0.055   otherwise

    Displays and 8 bit image files expect values in this encoding. It spends
    more of the 256 levels on dark tones, where the eye is more sensitive.
    Linear values that are shown or saved without it look too dark,
    especially in the shadows.
    """
    linear_image = np.clip(linear_image, 0.0, 1.0)
    encoded = np.where(linear_image <= 0.0031308,
                       12.92 * linear_image,
                       1.055 * np.power(linear_image, 1.0 / 2.4) - 0.055)
    return encoded.astype(np.float32)


def quantize_to_8bit(image):
    """
    Round values from 0 to 1 to the nearest of the 256 levels of an 8 bit
    file, and return them as float32 values from 0 to 1.

    The metrics are computed on these values, so they describe exactly the
    images that are saved to disk.
    """
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.float32) / 255.0


def save_image(image, file_path):
    """
    Save an RGB image with values from 0 to 1 as an 8 bit file.

    Values are clipped to [0, 1] and rounded to the nearest of the 256 levels.
    OpenCV writes files in BGR order, so the channels are reordered first.
    """
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    image_8bit = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not cv2.imwrite(file_path, cv2.cvtColor(image_8bit, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save image: {file_path}")


def save_side_by_side(images, file_path, scale=0.5):
    """
    Place RGB images of the same height next to each other, left to right,
    shrink the result by scale, and save it as an 8 bit file.
    """
    row = np.hstack(images)
    height, width = row.shape[:2]
    new_size = (round(width * scale), round(height * scale))
    small_row = cv2.resize(row, new_size, interpolation=cv2.INTER_AREA)
    save_image(small_row, file_path)
