# HDR Tone Mapping

Python implementations of three tone mapping algorithms for converting a high dynamic range (HDR) image to standard dynamic range (SDR), evaluated against an SDR reference using PSNR, SSIM and CIEDE2000 Delta E.

## Algorithms

All three are implemented from scratch in NumPy in `tone_mapping_algorithms.py`. Each one computes luminance using the Rec. 709 weights and scales it by the log average luminance of the image.

* **Reinhard**: global photographic operator using a key value of 0.18 and the `L / (1 + L)` compression curve.
* **Drago**: logarithmic operator with a detail parameter of 0.85, followed by gamma correction (2.2).
* **Adaptive Logarithmic**: log scaled luminance normalised by the log average luminance.

## Evaluation metrics

Defined in `evaluation_metrics.py` using scikit image:

* **PSNR** (higher is better): pixel level fidelity to the reference.
* **SSIM** (higher is better): similarity of local structure, contrast and luminance.
* **Delta E CIEDE2000** (lower is better): mean perceptual colour difference, calculated in CIELAB space.

## Setup

```
pip install -r requirments.txt
python main.py
```

The metrics are printed to the console and the tone mapped images are saved to `results/`.

## Results

Test image: `Desk.exr` (644 x 874), compared against `Desk_sdr.png`.

| Algorithm | PSNR (dB) | SSIM | Delta E 2000 |
|---|---|---|---|
| Reinhard | **14.99** | 0.55 | **12.83** |
| Drago | 8.08 | 0.39 | 30.23 |
| Adaptive Logarithmic | 10.68 | **0.59** | 18.69 |

![Reference, Reinhard, Drago, Adaptive Logarithmic](results/comparison.jpg)

From left to right: SDR reference, Reinhard, Drago, Adaptive Logarithmic.

## Findings

* **Reinhard** was the closest match to the reference overall, with the highest PSNR and the lowest colour error. It kept the most detail and colour saturation in the bright stained glass window. However, the output is darker than the reference in the shadows because no gamma correction is applied after tone mapping.
* **Drago** scored lowest on every metric. The logarithmic curve combined with gamma correction lifts the image too far, so the window highlights are blown out and colours are washed out, which gives a mean Delta E more than twice that of Reinhard.
* **Adaptive Logarithmic** had the highest SSIM, meaning it best preserved local structure and the overall brightness distribution, especially in the desk and shadow areas. It also clips the window highlights, which is why its PSNR and colour accuracy fall behind Reinhard.
* The results show a trade off between keeping highlight detail (Reinhard) and brightening shadows to match the reference (logarithmic operators). No single metric captures both, which is why all three are reported.

## Possible improvements

* Apply gamma correction to the Reinhard output before comparison.
* Tune the Drago detail and gamma parameters to reduce highlight clipping.
* Add a local operator (for example Durand bilateral filtering) and test on more HDR images.

## Credits

`images/Desk.exr` is from the [OpenEXR sample images](https://github.com/AcademySoftwareFoundation/openexr-images) by the Academy Software Foundation, distributed under the BSD 3 Clause licence.
