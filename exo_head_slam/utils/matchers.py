import numpy as np
import cv2
from typing import Optional, Tuple

class ORBMatcher:
    name = 'orb'

    """ORB-based feature matching placeholder."""
    def __init__(self, n_features: int = 1000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Matches features between two images. Returns (N, 2) arrays of (u, v)."""
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
            
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        return pts1, pts2


class LightGlueMatcher:
    name = 'lightglue'

    def __init__(self, device: str = 'cpu', max_keypoints: int = 2048):
        self.device = device
        self.max_keypoints = max_keypoints
        self.available = False
        self.error: Optional[Exception] = None
        self.torch = None
        self.extractor = None
        self.matcher = None

        try:
            import torch
            from lightglue import LightGlue, SuperPoint

            self.torch = torch
            self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
            self.matcher = LightGlue(features='superpoint').eval().to(device)
            self.available = True
        except Exception as exc:
            self.error = exc

    def _to_tensor(self, image: np.ndarray):
        gray = image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray.astype(np.float32) / 255.0)
        tensor = self.torch.from_numpy(gray)[None, None, :, :]
        return tensor.to(self.device)

    def _to_numpy(self, value):
        if value is None:
            return None
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _extract_points(self, features):
        keypoints = self._to_numpy(features.get('keypoints'))
        if keypoints is None:
            return np.empty((0, 2), dtype=np.float32)
        return np.squeeze(keypoints, axis=0) if keypoints.ndim == 3 else keypoints

    def match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.available:
            raise RuntimeError(f'LightGlue unavailable: {self.error}')

        image0 = self._to_tensor(img1)
        image1 = self._to_tensor(img2)

        with self.torch.inference_mode():
            features0 = self.extractor.extract(image0)
            features1 = self.extractor.extract(image1)
            matches = self.matcher({'image0': features0, 'image1': features1})

        keypoints0 = self._extract_points(features0)
        keypoints1 = self._extract_points(features1)

        if 'matches0' in matches:
            matches0 = self._to_numpy(matches['matches0'])
            if matches0 is None:
                return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
            matches0 = np.squeeze(matches0).astype(np.int64)
            valid = matches0 > -1
            if not np.any(valid):
                return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
            pts0 = keypoints0[valid]
            pts1 = keypoints1[matches0[valid]]
            return pts0.astype(np.float32), pts1.astype(np.float32)

        if 'matches' in matches:
            paired_matches = self._to_numpy(matches['matches'])
            if paired_matches is None or paired_matches.size == 0:
                return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
            paired_matches = np.asarray(paired_matches)
            if paired_matches.ndim == 2 and paired_matches.shape[1] == 2:
                pts0 = keypoints0[paired_matches[:, 0].astype(np.int64)]
                pts1 = keypoints1[paired_matches[:, 1].astype(np.int64)]
                return pts0.astype(np.float32), pts1.astype(np.float32)

        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
