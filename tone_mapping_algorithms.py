import numpy as np

def calculate_average_luminance(luminance):
    mask = luminance <= 0
    safe_luminance = np.where(mask, 1e-6, luminance)

    log_luminance = np.log(safe_luminance)
    average_log_luminance = np.mean(log_luminance)
    average_luminance = np.exp(average_log_luminance)

    return average_luminance


def reinhard_tone_mapping(image, tone_mapping_parameter=0.18):
    luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    average_luminance = calculate_average_luminance(luminance)
    scaled_luminance = tone_mapping_parameter * luminance / average_luminance
    tone_mapped_luminance = scaled_luminance / (1 + scaled_luminance)
    tone_mapped_image = np.zeros_like(image)
    for i in range(3):
        tone_mapped_image[:, :, i] = image[:, :, i] * (tone_mapped_luminance / luminance)
    return np.clip(tone_mapped_image, 0.0, 1.0)


def drago_tone_mapping(image, gamma=2.2, saturation=1.0, detail=0.85):
    luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    average_luminance = calculate_average_luminance(luminance)
    scaled_luminance = np.log(luminance + 1) / np.log(average_luminance + 1)
    tone_mapped_luminance = (np.log(scaled_luminance * detail + 1) / np.log(detail + 1))
    tone_mapped_image = np.zeros_like(image)
    for i in range(3):
        tone_mapped_image[:, :, i] = np.power(np.clip(image[:, :, i] * (tone_mapped_luminance / luminance), 0.0, 1.0),
                                              1.0 / gamma) * saturation
    return tone_mapped_image


def adaptive_logarithmic_mapping(image):
    luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    average_luminance = calculate_average_luminance(luminance)
    scaled_luminance = np.log(luminance + 1) / np.log(average_luminance + 1)
    tone_mapped_image = np.zeros_like(image)
    for i in range(3):
        tone_mapped_image[:, :, i] = np.clip(image[:, :, i] * (scaled_luminance / luminance), 0.0, 1.0)
    return tone_mapped_image
