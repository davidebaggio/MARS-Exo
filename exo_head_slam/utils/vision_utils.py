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


def blend_depth_holes(
    raw_depth: np.ndarray,
    predicted_depth: np.ndarray,
    raw_valid: np.ndarray,
    confidence_mask: np.ndarray,
    gate_by_confidence: bool,
) -> np.ndarray:
    """Fill invalid camera-depth pixels without replacing valid measurements."""
    use_prediction = ~raw_valid
    if gate_by_confidence:
        use_prediction &= confidence_mask

    combined = np.where(use_prediction, predicted_depth, raw_depth)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    combined[combined < 0.0] = 0.0
    return combined


def unproject_depth_np(
    depth: np.ndarray,
    camera_from_world: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Unproject camera depth into VGGT world coordinates."""
    height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    camera_points = np.stack(
        [
            (x - intrinsic[0, 2]) / intrinsic[0, 0] * depth,
            (y - intrinsic[1, 2]) / intrinsic[1, 1] * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = camera_from_world[:3, :3]
    translation = camera_from_world[:3, 3]
    return np.einsum("ij,hwj->hwi", rotation.T, camera_points - translation)


def transform_points(points: np.ndarray, output_from_input: np.ndarray) -> np.ndarray:
    """Apply a homogeneous rigid transform to an array of 3D points."""
    rotation = output_from_input[:3, :3]
    translation = output_from_input[:3, 3]
    return np.einsum("ij,nj->ni", rotation, points) + translation
