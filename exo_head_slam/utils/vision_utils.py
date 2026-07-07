import numpy as np


def apply_semantic_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blacks out dynamic objects in the image based on the mask.

    Args:
        image: Original RGB image.
        mask: Binary mask (1 for dynamic objects, 0 for static).

    Returns:
        Masked image.
    """
    masked_image = image.copy()
    masked_image[mask > 0] = 0
    return masked_image