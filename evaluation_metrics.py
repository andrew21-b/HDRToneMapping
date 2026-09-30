"""
Full reference image quality metrics.

Each function compares a test image with a reference image. Both must be
float RGB arrays of the same size with sRGB encoded values from 0 to 1,
which is how 8 bit image files store colour.
"""
import numpy as np
from skimage import color
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def calculate_psnr(reference, test):
    """
    Peak signal to noise ratio (PSNR) in decibels. Higher is better.

        MSE  = mean of (reference - test) ** 2 over all pixels and channels
        PSNR = 10 * log10(1 / MSE)

    The 1 in the formula is the largest possible value (data_range = 1).
    PSNR only measures pixel by pixel differences, so it does not model how
    people see images.
    """
    return peak_signal_noise_ratio(reference, test, data_range=1.0)


def calculate_ssim(reference, test):
    """
    Mean structural similarity (SSIM) index of Wang et al. (2004). Higher is
    better, and 1 means the images are identical.

    SSIM compares the local brightness, contrast and structure of the two
    images inside a small window that slides over every pixel, then averages
    the result over the image. The settings follow the paper: an 11 x 11
    Gaussian window with a standard deviation of 1.5 pixels, K1 = 0.01,
    K2 = 0.03 (the scikit-image defaults) and population rather than sample
    covariance.

    channel_axis=-1 tells scikit-image that the last axis holds the R, G and
    B channels. SSIM is computed on each channel and the three values are
    averaged. Wang et al. evaluated SSIM on the luminance channel only.
    """
    return structural_similarity(
        reference,
        test,
        data_range=1.0,
        channel_axis=-1,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )


def calculate_delta_e(reference, test):
    """
    Mean CIEDE2000 colour difference (Delta E 2000). Lower is better, and 0
    means the colours are identical.

    Both images are converted from sRGB to CIELAB with a D65 white point.
    CIELAB is designed so that equal distances look like roughly equal colour
    differences. The CIEDE2000 formula (Luo, Cui and Rigg, 2001), implemented
    as described by Sharma, Wu and Dalal (2005), gives the difference for
    each pixel, and the mean over all pixels is returned.
    """
    reference_lab = color.rgb2lab(reference)
    test_lab = color.rgb2lab(test)
    return float(np.mean(color.deltaE_ciede2000(reference_lab, test_lab)))
