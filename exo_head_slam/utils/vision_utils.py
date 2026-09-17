import numpy as np
import cv2


def sanitize_depth(
    depth: np.ndarray,
    encoding: str,
    unit_scale: float = 0.001,
    min_depth: float = 0.1,
    max_depth: float = 6.0,
) -> np.ndarray:
    """Convert a ROS depth image to finite 32-bit metres."""
    result = depth.astype(np.float32, copy=True)
    if encoding.lower() == '16uc1':
        result *= unit_scale
    invalid = ~np.isfinite(result) | (result < min_depth) | (result > max_depth)
    result[invalid] = 0.0
    return result


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


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Expand uncertain segmentation borders by a configurable radius."""
    if pixels <= 0 or not np.any(mask):
        return mask
    size = 2 * pixels + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8))


def blend_depth_holes(
    raw_depth: np.ndarray,
    predicted_depth: np.ndarray,
    raw_valid: np.ndarray,
    confidence_mask: np.ndarray,
    gate_by_confidence: bool,
    fillable_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fill invalid camera-depth pixels without replacing valid measurements."""
    use_prediction = ~raw_valid
    if gate_by_confidence:
        use_prediction &= confidence_mask
    if fillable_mask is not None:
        use_prediction &= fillable_mask

    combined = np.where(use_prediction, predicted_depth, raw_depth)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    combined[combined < 0.0] = 0.0
    return combined


def depth_edge_mask(depth: np.ndarray, rtol: float = 0.03) -> np.ndarray:
    """Mark both sides of large relative depth discontinuities."""
    valid = np.isfinite(depth) & (depth > 0.0)
    edges = np.zeros(depth.shape, dtype=bool)
    for axis in (0, 1):
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis] = slice(None, -1)
        second[axis] = slice(1, None)
        a = depth[tuple(first)]
        b = depth[tuple(second)]
        pair_valid = valid[tuple(first)] & valid[tuple(second)]
        discontinuity = pair_valid & (np.abs(a - b) > rtol * np.minimum(a, b))
        edges[tuple(first)] |= discontinuity
        edges[tuple(second)] |= discontinuity
    return edges


def robust_metric_scale(
    raw_depths: list[np.ndarray],
    predicted_depths: list[np.ndarray],
    confidence_masks: list[np.ndarray],
    min_support: int = 1000,
    max_camera_disagreement: float = 0.15,
    edge_rtol: float = 0.03,
) -> tuple[float, list[float | None], list[int]]:
    """Estimate one metric scale using robust log-depth ratios."""
    log_ratios = []
    camera_scales = []
    supports = []
    for raw, predicted, confident in zip(raw_depths, predicted_depths, confidence_masks):
        raw_valid = (
            np.isfinite(raw) & (raw > 0.1) & (raw < 10.0)
        ).astype(np.uint8)
        raw_valid = cv2.erode(raw_valid, np.ones((3, 3), np.uint8)).astype(bool)
        valid = (
            raw_valid & np.isfinite(predicted) & confident & (predicted > 0.0)
            & ~depth_edge_mask(raw, edge_rtol)
            & ~depth_edge_mask(predicted, edge_rtol)
        )
        ratios = np.log(raw[valid] / predicted[valid])
        supports.append(int(ratios.size))
        if ratios.size:
            log_ratios.append(ratios)
            camera_scales.append(float(np.exp(np.median(ratios))))
        else:
            camera_scales.append(None)

    if sum(supports) < min_support or not log_ratios:
        raise ValueError(f'insufficient scale support: {sum(supports)} < {min_support}')
    available = [value for value in camera_scales if value is not None]
    if len(available) == 2:
        disagreement = abs(available[0] - available[1]) / max(available)
        if disagreement > max_camera_disagreement:
            raise ValueError(
                f'camera scale disagreement: {disagreement:.3f} > '
                f'{max_camera_disagreement:.3f}'
            )
    return float(np.exp(np.median(np.concatenate(log_ratios)))), camera_scales, supports


def bidirectional_overlap(
    depths: tuple[np.ndarray, np.ndarray],
    camera_from_world: tuple[np.ndarray, np.ndarray],
    intrinsics: tuple[np.ndarray, np.ndarray],
    masks: tuple[np.ndarray, np.ndarray],
    stride: int = 8,
) -> float:
    """Return mean fraction of valid points projecting into the other view."""
    scores = []
    for source, target in ((0, 1), (1, 0)):
        depth = depths[source]
        valid = masks[source] & np.isfinite(depth) & (depth > 0.0)
        sampled = np.zeros_like(valid)
        sampled[::stride, ::stride] = valid[::stride, ::stride]
        points = unproject_depth_np(
            depth, camera_from_world[source], intrinsics[source]
        )[sampled]
        if not len(points):
            scores.append(0.0)
            continue
        target_camera = camera_from_world[target]
        camera_points = (
            points @ target_camera[:3, :3].T + target_camera[:3, 3]
        )
        z = camera_points[:, 2]
        intrinsic = intrinsics[target]
        with np.errstate(divide='ignore', invalid='ignore'):
            x = intrinsic[0, 0] * camera_points[:, 0] / z + intrinsic[0, 2]
            y = intrinsic[1, 1] * camera_points[:, 1] / z + intrinsic[1, 2]
        height, width = depths[target].shape
        inside = (
            (z > 0.0) & (x >= 0.0) & (x < width)
            & (y >= 0.0) & (y < height)
        )
        target_match = np.zeros_like(inside)
        if inside.any():
            indices = np.flatnonzero(inside)
            px = np.floor(x[inside]).astype(int)
            py = np.floor(y[inside]).astype(int)
            target_depth = depths[target][py, px]
            target_valid = masks[target][py, px] & np.isfinite(target_depth)
            depth_match = np.abs(target_depth - z[inside]) <= (
                0.15 * np.maximum(target_depth, z[inside])
            )
            target_match[indices] = target_valid & depth_match
        scores.append(float(np.mean(target_match)))
    return float(np.mean(scores))


def camera_points_from_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Unproject depth to its camera frame."""
    identity = np.eye(4, dtype=np.float64)[:3]
    return unproject_depth_np(depth, identity, intrinsic)


def voxel_keep_first(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-deduplicate while preserving first-source priority."""
    if not len(points):
        return points, colors
    voxels = np.floor(points / voxel_size).astype(np.int64)
    _, first = np.unique(voxels, axis=0, return_index=True)
    first.sort()
    return points[first], colors[first]


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
