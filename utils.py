import numpy as np

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)

def psnr2(raw_image: np.ndarray, dst_image: np.ndarray, crop_border: int) -> float:
    """Python implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function

    Args:
        raw_image (np.ndarray): image data to be compared, BGR format, data range [0, 255]
        dst_image (np.ndarray): reference image data, BGR format, data range [0, 255]
        crop_border (int): crop border a few pixels

    Returns:
        psnr_metrics (np.float64): PSNR metrics

    """
    # Check if two images are similar in scale and type
    # _check_image(raw_image, dst_image)

    # crop border pixels
    if crop_border > 0:
        raw_image = raw_image[crop_border:-crop_border, crop_border:-crop_border, ...]
        dst_image = dst_image[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Convert data type to numpy.float64 bit
    raw_image = raw_image.astype(np.float64)
    dst_image = dst_image.astype(np.float64)

    psnr_metrics = 10 * np.log10((255.0 ** 2) / np.mean((raw_image - dst_image) ** 2) + 1e-8)

    return psnr_metrics
