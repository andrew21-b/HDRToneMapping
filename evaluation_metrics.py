import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage import color


def calculate_psnr(image1, image2, data_range=1.0):
    return peak_signal_noise_ratio(image1, image2, data_range=data_range)


def calculate_ssim(image1, image2, data_range=1.0, multichannel=True, win_size=3):
    min_dim = min(image1.shape[:2])
    if win_size is None:
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1

    if win_size > min_dim:
        raise ValueError("Images are too small to calculate SSIM.")
    return structural_similarity(image1, image2, data_range=data_range, multichannel=multichannel, win_size=win_size)


def calculate_delta_e(image1, image2):
    lab1 = color.rgb2lab(image1)
    lab2 = color.rgb2lab(image2)
    delta_e = color.deltaE_ciede2000(lab1, lab2)
    return np.mean(delta_e)
