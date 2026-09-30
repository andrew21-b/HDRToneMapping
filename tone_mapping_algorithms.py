"""
Three global tone mapping operators.

A tone mapping operator turns an HDR image, whose values can be far above 1,
into display values from 0 to 1. A global operator applies the same curve to
every pixel, so the output of a pixel depends only on its own luminance and
on statistics of the whole image, not on its neighbours.

Every operator here takes a linear RGB float image (values of 0 or more) and
works in the same three steps:

1. Compute the luminance (brightness) of each pixel from its R, G and B.
2. Compress the luminance with the curve of the operator.
3. Multiply R, G and B of each pixel by (new luminance / old luminance).

The output is still linear. It must be encoded with the sRGB transfer
function (image_io.encode_srgb) before it is shown, saved or compared with
an 8 bit image.
"""
import numpy as np

# Weights that convert linear RGB with Rec. 709 primaries and a D65 white
# point (the same primaries and white point as sRGB) to luminance Y.
# Source: Recommendation ITU-R BT.709-6.
REC709_LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)


def compute_luminance(image):
    """Return the luminance Y of each pixel of a linear RGB image."""
    red_weight, green_weight, blue_weight = REC709_LUMINANCE_WEIGHTS
    return (red_weight * image[:, :, 0]
            + green_weight * image[:, :, 1]
            + blue_weight * image[:, :, 2])


def log_average_luminance(luminance, delta=1e-6):
    """
    Return the log average luminance (the geometric mean) of an image.

    Reinhard et al. (2002), Equation 1:

        L_avg = exp(mean(log(delta + L_w)))

    L_w is the luminance of each pixel and delta is a small number that
    avoids log(0) for black pixels. (The printed Equation 1 places the 1/N
    of the mean outside the exp. The geometric mean needs it inside, as here.)

    It estimates the overall brightness of the scene. Unlike the ordinary
    mean, it is not dominated by a few very bright pixels such as a window
    or a lamp.
    """
    return float(np.exp(np.mean(np.log(delta + luminance))))


def apply_display_luminance(image, world_luminance, display_luminance):
    """
    Give each pixel its new (display) luminance while keeping its colour.

    R, G and B are all multiplied by display_luminance / world_luminance.
    Multiplying all three channels by the same number keeps the ratios
    between them, so hue and saturation do not change. It is the same as
    scaling X, Y and Z in the CIE XYZ colour space, so the chromaticity
    (x, y) of each pixel is unchanged.

    Pixels with zero luminance are black and stay black, which avoids a
    division by zero. A very saturated pixel can have one channel above 1
    even when its luminance is below 1; such values are clipped to 1.
    """
    ratio = np.divide(display_luminance, world_luminance,
                      out=np.zeros_like(world_luminance),
                      where=world_luminance > 0)
    return np.clip(image * ratio[:, :, np.newaxis], 0.0, 1.0)


def reinhard_tone_mapping(image, key=0.18):
    """
    Global photographic tone reproduction operator of Reinhard et al. (2002).

    Equation 2 scales the luminance so that the log average luminance is
    mapped to the key value a:

        L = (a / L_avg) * L_w

    Equation 3 then compresses the scaled luminance into the range 0 to 1:

        L_d = L / (1 + L)

    Dark values (L much smaller than 1) are almost unchanged, while bright
    values are divided by about L, so they approach 1 but never reach it.

    key: the key value a. The paper uses 0.18 (middle grey) for a normal
    scene and suggests values from 0.045 (darker) to 0.72 (brighter).

    Only this simple global operator is implemented. The paper also gives
    Equation 4, which lets the brightest values burn out to white, and a
    local version with automatic dodging and burning; neither is used here.
    """
    world_luminance = compute_luminance(image)
    scaled_luminance = key / log_average_luminance(world_luminance) * world_luminance
    display_luminance = scaled_luminance / (1.0 + scaled_luminance)
    return apply_display_luminance(image, world_luminance, display_luminance)


def drago_tone_mapping(image, bias=0.85, max_display_luminance=100.0):
    """
    Adaptive logarithmic mapping of Drago et al. (2003).

    Section 3.3: the luminance of each pixel is divided by the world
    adaptation luminance L_wa, which is the log average luminance. (The
    paper also allows an extra exposure factor, which is 1 here.) L_wmax is
    the largest luminance after this division.

    Equation 4 then compresses each value with a logarithm whose base
    changes smoothly from 2 for the darkest pixels (more contrast) to 10 for
    the brightest pixels (more compression):

        L_d = (L_dmax * 0.01 / log10(L_wmax + 1))
              * log(L_w + 1) / log(2 + 8 * (L_w / L_wmax) ** (log(b) / log(0.5)))

    The term (L_w / L_wmax) ** (log(b) / log(0.5)) is the bias function of
    Perlin and Hoffert (Equation 3 of the paper). With L_dmax = 100 the
    brightest pixel is mapped to exactly 1 (white).

    bias: the bias parameter b. The paper proposes 0.85 as the default and
    finds values from 0.7 to 0.9 the most useful. Smaller values give a
    brighter image with more detail in dark areas. Below about 0.7 the curve
    rises above 1 for some bright values, which are then clipped (Figure 5
    of the paper).

    max_display_luminance: L_dmax, the maximum luminance of the display in
    cd/m2. The paper uses 100.

    Section 3.3 also divides L_wa by (1 + b - 0.85) ** 5, so the overall
    brightness stays about the same when b changes. At the default b = 0.85
    this factor is 1.

    The gamma correction proposed in Section 4 of the paper is not applied
    here. All three operators are encoded with the same sRGB curve in
    main.py, so they are compared on equal terms.
    """
    world_luminance = compute_luminance(image)
    adaptation_luminance = log_average_luminance(world_luminance) / (1.0 + bias - 0.85) ** 5
    scaled_luminance = world_luminance / adaptation_luminance
    max_scaled_luminance = scaled_luminance.max()
    bias_exponent = np.log(bias) / np.log(0.5)
    bias_term = (scaled_luminance / max_scaled_luminance) ** bias_exponent
    display_luminance = (max_display_luminance * 0.01 / np.log10(max_scaled_luminance + 1.0)
                         * np.log(scaled_luminance + 1.0) / np.log(2.0 + 8.0 * bias_term))
    return apply_display_luminance(image, world_luminance, display_luminance)


def logarithmic_mapping(image):
    """
    Basic logarithmic mapping, Equation 1 of Drago et al. (2003):

        L_d = log(L_w + 1) / log(L_max + 1)

    L_max is the largest luminance in the image, so the brightest pixel is
    mapped to 1 (white) and every other value is compressed smoothly below
    it. Drago et al. present this curve following Stockham (1972), who
    recommended a logarithmic relation between light and perceived
    brightness for image processing.

    Drago et al. note that it compresses luminance too much, so the image
    loses its impression of high contrast; their adaptive mapping
    (drago_tone_mapping) was designed to fix this. It is included here as a
    simple baseline. It is applied to the luminance values as stored in the
    file, so unlike the other two operators its result depends on the scale
    of those values.
    """
    world_luminance = compute_luminance(image)
    display_luminance = np.log(world_luminance + 1.0) / np.log(world_luminance.max() + 1.0)
    return apply_display_luminance(image, world_luminance, display_luminance)
