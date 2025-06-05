import os
import cv2
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def load_hdr_image(file_path: str):
    try:
        img = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"Could not load image at {file_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Error loading HDR image: {e}")
        return None


def load_sdr_image(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not load image at {file_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    except Exception as e:
        print(f"Error loading SDR image: {e}")
        return None


def view_hdr_image(hdr_img):
    if hdr_img is not None:
        print(f"Image shape: {hdr_img.shape}, dtype: {hdr_img.dtype}")
        if not np.issubdtype(hdr_img.dtype, np.floating):
            print("Warning: Image is not floating-point.  Ensure it's properly converted.")

        cv2.imshow("HDR Image (Original)", hdr_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to load HDR image.")
