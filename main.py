from load_image import load_sdr_image, load_hdr_image
from evaluation_metrics import calculate_ssim, calculate_psnr, calculate_delta_e
from tone_mapping_algorithms import reinhard_tone_mapping, drago_tone_mapping, adaptive_logarithmic_mapping

sdr_img = load_sdr_image("images/Desk_sdr.png")
hdr_img = load_hdr_image("images/Desk.exr")

reinhard_img = reinhard_tone_mapping(hdr_img)
drago_img = drago_tone_mapping(hdr_img)
alm_img = adaptive_logarithmic_mapping(hdr_img)

psnr_reinhard = calculate_psnr(sdr_img, reinhard_img)
ssim_reinhard = calculate_ssim(sdr_img, reinhard_img)
delta_e_reinhard = calculate_delta_e(sdr_img, reinhard_img)

psnr_drago = calculate_psnr(sdr_img, drago_img)
ssim_drago = calculate_ssim(sdr_img, drago_img)
delta_e_drago = calculate_delta_e(sdr_img, drago_img)

psnr_alm = calculate_psnr(sdr_img, alm_img)
ssim_alm = calculate_ssim(sdr_img, alm_img)
delta_e_alm = calculate_delta_e(sdr_img, alm_img)

print(f"Reinhard Tone Mapping - PSNR: {psnr_reinhard:.2f}, SSIM: {ssim_reinhard:.2f}, Delta E: {delta_e_reinhard:.2f}")
print(f"Drago Tone Mapping - PSNR: {psnr_drago:.2f}, SSIM: {ssim_drago:.2f}, Delta E: {delta_e_drago:.2f}")
print(f"Adaptive Logarithmic Mapping - PSNR: {psnr_alm:.2f}, SSIM: {ssim_alm:.2f}, Delta E: {delta_e_alm:.2f}")
