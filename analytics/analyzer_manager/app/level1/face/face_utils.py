from enum import Enum
import cv2
import numpy as np

from general.core import AttrConfidence


class FaceConfidence(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

def get_min_confidence(conf1, conf2):
    return min(conf1, conf2, key=lambda x: conf_order_map[x])

face_conf_to_attr_conf = {
    FaceConfidence.NONE: AttrConfidence.LOW,
    FaceConfidence.LOW: AttrConfidence.LOW,
    FaceConfidence.MEDIUM: AttrConfidence.MEDIUM,
    FaceConfidence.HIGH: AttrConfidence.HIGH,
}

attr_conf_to_face_conf = {
    AttrConfidence.LOW: FaceConfidence.LOW,
    AttrConfidence.MEDIUM: FaceConfidence.MEDIUM,
    AttrConfidence.HIGH : FaceConfidence.HIGH
}

conf_order_map = {
    FaceConfidence.NONE: 0,
    FaceConfidence.LOW: 1,
    FaceConfidence.MEDIUM: 2,
    FaceConfidence.HIGH: 3,
}
conf2cosine_th = {
    FaceConfidence.NONE: 0.7,
    FaceConfidence.LOW: 0.6,
    FaceConfidence.MEDIUM: 0.55,
    FaceConfidence.HIGH: 0.5,
}
conf2factor = {
    FaceConfidence.NONE: 0.6,
    FaceConfidence.LOW: 0.75,
    FaceConfidence.MEDIUM: 0.9,
    FaceConfidence.HIGH: 1,
}
conf_incrementor = {
    FaceConfidence.NONE: FaceConfidence.LOW,
    FaceConfidence.LOW: FaceConfidence.MEDIUM,
    FaceConfidence.MEDIUM: FaceConfidence.HIGH,
    FaceConfidence.HIGH: FaceConfidence.HIGH,
}
conf2score = {
    FaceConfidence.NONE: 0.2,
    FaceConfidence.LOW: 0.4,
    FaceConfidence.MEDIUM: 0.6,
    FaceConfidence.HIGH: 0.8,
}


def gen_face_mask(image, landmarks, bbox, dilation_scale=0.3):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    sz = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    polygon = landmarks[[0, 1, 4, 3], :]
    cv2.fillPoly(mask, [polygon.astype(int)], 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(sz[0] * dilation_scale), int(sz[1] * dilation_scale)))
    dilated_mask = cv2.dilate(mask, kernel)
    return dilated_mask


def face_quality_estimator(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian operator
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    return np.nanvar(lap), np.std(image)
#    return np.nanvar(lap[mask == 1]), np.std(image[mask == 1])


def torchvision_like_resize(image, target_size=(112, 112), sigma_scale=0.4):
    """
    Resize image similar to torchvision.transforms.Resize with bilinear interpolation and antialiasing.
    Args:
        image: Input image (numpy array, RGB)
        target_size: (width, height) tuple
        sigma_scale: Multiplier for sigma calculation (sigma = sigma_scale / scale)
    
    The actual sigma applied is: sigma = sigma_scale / scale_factor
    This means larger downscaling automatically gets more blur.
    """
    h, w = image.shape[:2]
    scale_x = target_size[0] / float(w)
    scale_y = target_size[1] / float(h)
    result = image.copy()
    if scale_x < 1 or scale_y < 1:
        sigma_x = sigma_scale / scale_x if scale_x < 1 else 0
        sigma_y = sigma_scale / scale_y if scale_y < 1 else 0
        if sigma_x > 0 or sigma_y > 0:
            ksize_x =  max(1, int(np.ceil(sigma_x * 6)) | 1 if sigma_x > 0 else 1)
            ksize_y = max(1, int(np.ceil(sigma_y * 6)) | 1 if sigma_y > 0 else 1)
            result = cv2.GaussianBlur(result, (ksize_x, ksize_y), sigmaX=sigma_x, sigmaY=sigma_y)
    return cv2.resize(result, target_size, interpolation=cv2.INTER_LINEAR)
