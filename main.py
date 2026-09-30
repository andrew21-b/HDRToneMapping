"""
Tone map an HDR image with three operators and compare each result with an
SDR reference image.

For each operator this script:
1. tone maps the linear HDR image to linear display values from 0 to 1,
2. encodes them with the sRGB transfer function, the same encoding as the
   reference PNG,
3. rounds them to 8 bits, exactly as they are saved,
4. prints PSNR, SSIM and Delta E 2000 against the reference,
5. saves the image to the results folder.

It also saves results/comparison.jpg, which shows the reference and the three
results side by side.
"""
from evaluation_metrics import calculate_delta_e, calculate_psnr, calculate_ssim
from image_io import (encode_srgb, load_hdr_image, load_sdr_image, quantize_to_8bit,
                      save_image, save_side_by_side)
from tone_mapping_algorithms import drago_tone_mapping, logarithmic_mapping, reinhard_tone_mapping

HDR_PATH = "images/Desk.exr"
REFERENCE_PATH = "images/Desk_sdr.png"
RESULTS_FOLDER = "results"

# (name printed in the table, output file name, tone mapping function)
OPERATORS = [
    ("Reinhard", "reinhard.png", reinhard_tone_mapping),
    ("Drago", "drago.png", drago_tone_mapping),
    ("Logarithmic", "logarithmic.png", logarithmic_mapping),
]


def main():
    hdr_image = load_hdr_image(HDR_PATH)
    reference_image = load_sdr_image(REFERENCE_PATH)
    comparison_images = [reference_image]

    print(f"{'Operator':<12} {'PSNR (dB)':>10} {'SSIM':>7} {'Delta E 2000':>13}")
    for name, file_name, tone_map in OPERATORS:
        linear_result = tone_map(hdr_image)
        result = quantize_to_8bit(encode_srgb(linear_result))

        psnr = calculate_psnr(reference_image, result)
        ssim = calculate_ssim(reference_image, result)
        delta_e = calculate_delta_e(reference_image, result)
        print(f"{name:<12} {psnr:>10.2f} {ssim:>7.3f} {delta_e:>13.2f}")

        save_image(result, f"{RESULTS_FOLDER}/{file_name}")
        comparison_images.append(result)

    save_side_by_side(comparison_images, f"{RESULTS_FOLDER}/comparison.jpg")


if __name__ == "__main__":
    main()
