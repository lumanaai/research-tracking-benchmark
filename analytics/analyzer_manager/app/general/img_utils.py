from math import floor
from typing import List
from shapely.geometry import box
from shapely.ops import unary_union

from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import cv2


def crop_image(image, pos, margins=None, bgr_map=True):
    [x1, y1, x2, y2] = fit_bbox_with_margins(pos, image.shape[1::-1], margins)
    if bgr_map:
        cropped = image[y1:y2, x1:x2].copy()
    else:
        cropped = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return cropped

def fit_bbox_with_margins(pos, image_size, margins=None):
    if margins is None:
        margins = [0.05, 0.05]

    bbox = pos * np.array([*image_size, *image_size])
    y_gb = int((bbox[3] - bbox[1]) * margins[1])
    x_gb = int((bbox[2] - bbox[0]) * margins[0])

    y1 = int(max((bbox[1] - y_gb), 0))
    y2 = int(min((bbox[3] + y_gb), image_size[1] - 1))
    x1 = int(max((bbox[0] - x_gb), 0))
    x2 = int(min((bbox[2] + x_gb), image_size[0] - 1))
    return [x1, y1, x2, y2]


def crop_image_by_bbox(image, bbox, margins=None):
    if margins is None:
        margins = [0.05, 0.05]

    y_gb = int((bbox[3] - bbox[1]) * margins[1])
    x_gb = int((bbox[2] - bbox[0]) * margins[0])

    y1 = int(max((bbox[1] - y_gb), 0))
    y2 = int(min((bbox[3] + y_gb), image.shape[0] - 1))
    x1 = int(max((bbox[0] - x_gb), 0))
    x2 = int(min((bbox[2] + x_gb), image.shape[1] - 1))
    return image[y1:y2, x1:x2].copy()



def crop_image_by_bbox_and_ar(image, bbox, aspect_ratio, margins=None, const_margin=False, bgr_map=True, return_pos=False):
    if margins is None:
        margins = [0.05, 0.05]
    pos = fix_bbox_to_ar(bbox, aspect_ratio, image.shape[1::-1], margins, const_margin)
    reminders = np.zeros(2, dtype=int)
    if pos[0] < 0:
        reminders[0] = -pos[0]
        pos[0] = 0
    if pos[1] < 0:
        reminders[1] = -pos[1]
        pos[1] = 0
    if bgr_map:
        crop = image[pos[1] : pos[3], pos[0] : pos[2]].copy()
    else:
        crop = cv2.cvtColor(image[pos[1] : pos[3], pos[0] : pos[2]], cv2.COLOR_BGR2RGB)
    if np.any(reminders > 0):
        crop = cv2.copyMakeBorder(crop, reminders[1], 0, reminders[0], 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if return_pos:
        return crop, pos
    return crop


def fix_bbox_to_ar(bbox, aspect_ratio, image_size, margins=None, const_margin=False, min_size=None):
    if margins is None:
        margins = [0.0, 0.0]

    x1, y1, x2, y2 = bbox
    bb_width = x2 - x1
    bb_height = y2 - y1
    bb_center_x = x1 + bb_width / 2
    bb_center_y = y1 + bb_height / 2

    y_gb = bb_height * margins[1]
    x_gb = bb_width * margins[0]

    if const_margin:
        x_gb, y_gb = 20, 20

    bb_height += 2 * y_gb
    bb_width += 2 * x_gb

    # Compute the size of the new bounding box based on target aspect ratio
    if bb_width / bb_height > aspect_ratio:
        # Width is larger than desired. Adjust height.
        new_height = bb_width / aspect_ratio
        new_width = bb_width
    else:
        # Height is larger than desired. Adjust width.
        new_width = bb_height * aspect_ratio
        new_height = bb_height

    # Enforce minimum dimensions and re-fix AR if needed
    if min_size is not None:
        min_w, min_h = min_size
        new_width = max(new_width, min_w)
        new_height = max(new_height, min_h)
        if new_width / new_height > aspect_ratio:
            new_height = new_width / aspect_ratio
        else:
            new_width = new_height * aspect_ratio

    # Check for image boundaries and adjust if necessary
    left = bb_center_x - new_width / 2
    top = bb_center_y - new_height / 2
    right = bb_center_x + new_width / 2
    bottom = bb_center_y + new_height / 2

    # to keep the aspect ratio, we need to address the cases where
    # the face is close to the boarder

    if left < 0:
        right -= left
        left = 0
    if right > image_size[0]:
        left -= right - image_size[0]
        right = image_size[0]
    if top < 0:
        bottom -= top
        top = 0
    if bottom > image_size[1]:
        top -= bottom - image_size[1]
        bottom = image_size[1]

    return np.array([left, top, right, bottom]).astype(int)


def crop_with_minimal(image, bbox, minimal_size, ar:float = None):
    """
    Crops an image using a bounding box in TLBR format.
    If one of the axes is smaller than the minimal size,
    it crops to the minimal size but keeps the original bbox centered.

    :param image: Input image
    :param bbox: Bounding box in TLBR format (top, left, bottom, right)
    :param minimal_size: Tuple of (min_width, min_height)
    :param ar: Aspect ratio to keep
    :return: Cropped image
    """
    im_size = image.shape[1::-1]
    left, top, right, bottom = (bbox * np.array([*im_size, *im_size])).astype(int)
    original_width = right - left
    original_height = bottom - top
    center_x, center_y = left + original_width // 2, top + original_height // 2

    # Adjusting the width and height to the minimum size if necessary
    # Ensuring the bounding box does not exceed image boundaries

    if original_width < minimal_size[0]:
        left = max(center_x - minimal_size[0] // 2, 0)
        right = min(left + minimal_size[0], image.shape[1])
    if original_height < minimal_size[1]:
        top = max(center_y - minimal_size[1] // 2, 0)
        bottom = min(top + minimal_size[1], image.shape[0])
    if ar is None:
        # Cropping and returning the image
        cropped_image = image[top:bottom, left:right].copy()
    else:
        cropped_image = crop_image_by_bbox_and_ar(image, [left, top, right, bottom], ar, margins=[0,0])

    return cropped_image


def crop_image_by_perimeter(image, perimeter, zoom_factor=0.1):
    y_gb = int((perimeter["bottomRight"]["y"] - perimeter["topLeft"]["y"]) * zoom_factor * image.shape[0])
    x_gb = int((perimeter["bottomRight"]["x"] - perimeter["topLeft"]["x"]) * zoom_factor * image.shape[1])

    y1 = max(int(perimeter["topLeft"]["y"] * image.shape[0] - y_gb), 0)
    y2 = min(int(perimeter["bottomRight"]["y"] * image.shape[0] + y_gb), image.shape[0] - 1)
    x1 = max(int(perimeter["topLeft"]["x"] * image.shape[1] - x_gb), 0)
    x2 = min(int(perimeter["bottomRight"]["x"] * image.shape[1] + x_gb), image.shape[1] - 1)
    return image[y1:y2, x1:x2]


def scale_image_by_width(im, width):
    scale_factor = width / im.shape[1]  # percent of original size
    height = floor(im.shape[0] * scale_factor / 2) * 2
    return cv2.resize(im, (width, height))


# extracting YUV420 to Y (cv2 format)
def yuv420_to_luma_extract(data, width, height):
    byteArray = bytearray(data)
    s = width * height
    luma = byteArray[0:s]
    luma = np.reshape(luma, (height, width))
    luma = luma.astype(np.uint8)
    return luma

def is_cv2_cuda_enabled() -> bool:
    return cv2.cuda.getCudaEnabledDeviceCount() > 0

def motion_score(image_crop, low_pctl=10, low_th=30, crop_sz=0.8, min_grad_len=500) -> float:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)

    # blur speckle noise
    image = cv2.medianBlur(gray, 5)

    height, width = image.shape[:2]
    crop_start = (1 - crop_sz) / 2
    crop_end = 1 - crop_start

    # Crop the center of the image
    start_row, start_col = int(height * crop_start), int(width * crop_start)
    end_row, end_col = int(height * crop_end), int(width * crop_end)
    image = image[start_row:end_row, start_col:end_col]
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grads = np.abs(sobelx)
    grads_arr = grads[grads > low_th]
    if len(grads_arr) > min_grad_len:
        low_grad = np.percentile(grads_arr, low_pctl)
        score = np.count_nonzero(grads_arr > low_grad * 2) / len(grads_arr)
    else:
        score = 0
    return score


def is_image_monochrome(img, threshold=30, percentage=0.8, subsample: int = 4) -> bool:
    """
    Check if an image is close to monochrome by comparing its channels.
    :param img: the image to check.
    :param threshold: Maximum allowed difference between channels for a pixel to be considered similar.
    :param percentage: Percentage of pixels that should be similar for the image to be considered monochrome.
    :param subsample: subsampling factor.
    :return: True if the image is close to monochrome, otherwise False.
    """
    r, g, b = img[::subsample, ::subsample, 0], img[::subsample, ::subsample, 1], img[::subsample, ::subsample, 2]
    similar_pixels = np.logical_and(np.abs(r - g) < threshold, np.abs(g - b) < threshold, np.abs(r - b) < threshold)
    return bool(np.mean(similar_pixels) >= percentage)


def measure_contrast(image: np.array):
    """
    Measure the contrast of an RGB image.

    Args:
        image (np.ndarray): Input image as an RGB NumPy array.

    Returns:
        float: The contrast of the image.
    """
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.std(grayscale)


def downscale_to_max_dim(image, max_dim):
    """
    Downscale an image to fit within a maximum dimension while preserving the aspect ratio.

    :param image: Input image as a NumPy array.
    :param max_dim: Maximum dimension (either width or height) to downscale the image to.
    :return: Resized image as a NumPy array.
    """

    # Get the original dimensions
    height, width = image.shape[:2]

    if height <= max_dim and width <= max_dim:
        return image

    # Determine the scaling factor to fit the max dimension
    if height > width:
        scaling_factor = max_dim / float(height)
    else:
        scaling_factor = max_dim / float(width)

    # Calculate the new dimensions
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image

def draw_target(img: np.array, bbox: np.array, color=(0, 255, 255), thickness=2, max_size=4):
    """
    Draw a cross on an image.
    :param img: The image to draw on.
    :param bbox: The bounding box of the cross.
    :param color: The color of the cross.
    :param thickness: The thickness of the cross.
    :param max_size: The maximum size of the cross.
    :return: The image with the cross drawn on it.
    """
    x1, y1, x2, y2 = bbox
    center = (int((x1 + x2) / 2), y2)
    size = min(max_size, int((x2-x1)/3))
    #cv2.line(img, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), color, thickness)
    #cv2.line(img, (center[0] - size, center[1] + size), (center[0] + size, center[1] - size), color, thickness)
    cv2.line(img, (center[0] , center[1] + size), (center[0], center[1] - size), color, thickness)
    #cv2.circle(img, center, size, color, thickness=-1)
    return img

def apply_overlay(image, active_grid: np.array, color=[0, 255, 0], alpha = 0.3):

    # Create a copy of the original image to apply the overlay
    overlay = np.zeros((active_grid.shape[0], active_grid.shape[1], 3), dtype=np.uint8)
    overlay[active_grid == 1] = color
    resized_overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Blend the overlay with the original image
    return cv2.addWeighted(image, 1 - alpha, resized_overlay, alpha, 0)


def concatenate_images(image_list: List[np.array], is_vertical: bool = True) -> np.array:
    """
       Concatenate a list of images horizontally\ vertically using OpenCV.

       Args:
           image_list (list): List of image arrays (OpenCV format).
          is_vertical (bool): Concatenate vertically if True, horizontally if False.

       Returns:
           np.ndarray: The concatenated image.
       """
    # Ensure all images have the same height
    axis = 1 if is_vertical else 0
    dims = [img.shape[axis] for img in image_list]
    max_dim = max(dims)

    # Resize images to have the same height
    if is_vertical:
        resized_images = [
            cv2.resize(img, (max_dim, int(img.shape[0] * max_dim / img.shape[1]))) for img in image_list
        ]
        return np.vstack(resized_images)
    else:
        resized_images = [
            cv2.resize(img, (int(img.shape[1] * max_dim / img.shape[0]), max_dim)) for img in image_list
        ]
        return np.hstack(resized_images)


def sharpness_score_batch(detector_images_bgr):
    """
    Compute sharpness scores for a batch of BGR NumPy images.
    
    Args:
        detector_images_bgr (List[np.ndarray]): List of BGR images.
        
    Returns:
        List[float]: Sharpness score for each image.
    """
    # from PIL import Image, ImageFilter, ImageStat
    scores = []
    for bgr_image in detector_images_bgr:
        # Optionally resize
        h, w = bgr_image.shape[:2]
        ratio = max(w, h) / w
        if ratio > 1:
            bgr_image = cv2.resize(bgr_image, (max(int(w // ratio), 1), max(int(h // ratio), 1)))
        # Convert to grayscale
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        # Apply Laplacian filter
        edges = cv2.filter2D(gray, ddepth=-1, kernel=np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32))
        # Sharpness
        sharpness = np.std(edges)
        scores.append(sharpness)
    return np.array(scores, dtype=np.float32)


def brightness_score_batch(detector_images_bgr):
    """
    Compute brightness scores for a batch of BGR NumPy images.
    
    Args:
        detector_images_bgr (List[np.ndarray]): List of BGR images.
        
    Returns:
        List[float]: Brightness score for each image.
    """
    scores = []
    for bgr_image in detector_images_bgr:
        # Convert BGR to RGB by swapping channels
        rgb_image = bgr_image[..., ::-1].astype(np.float32)
        # Compute mean per channel
        mean_rgb = np.mean(rgb_image, axis=(0, 1))  # shape (3,)
        red, green, blue = mean_rgb
        # Compute brightness
        cur_bright = (
            np.sqrt(
                0.241 * (red ** 2) +
                0.691 * (green ** 2) +
                0.068 * (blue ** 2)
            ) / 255
        )
        scores.append(cur_bright * 100)
    return np.array(scores, dtype=np.float32)


def union_bboxes(bboxes_xyxyn, detections_lst):
    """
    Given list of bboxes [x_min,y_min,x_max,y_max]
    and indices, return non-overlapping rectangular bboxes
    that cover their union.
    """
    polys = [box(*bboxes_xyxyn[i]) for i in detections_lst]
    merged = unary_union(polys)
    out_bboxes = []
    if merged.geom_type == "Polygon":
        out_bboxes.append(list(merged.bounds))
    elif merged.geom_type == "MultiPolygon":
        for p in merged.geoms:
            out_bboxes.append(list(p.bounds))
    return out_bboxes


class ResizeAndPadToTarget(ImageOnlyTransform):
    def __init__(self, target_height=256, target_width=256, interpolation=cv2.INTER_LINEAR, always_apply=True, p=1.0):
        super(ResizeAndPadToTarget, self).__init__(always_apply, p)
        self.target_height = target_height
        self.target_width = target_width
        self.interp_mode = interpolation

    def apply(self, img, **params):
        # Calculate the new size preserving the aspect ratio
        height, width = img.shape[:2]
        if height == self.target_height and width == self.target_width:
            return img

        scale = min(self.target_height / height, self.target_width / width)
        new_height, new_width = int(height * scale), int(width * scale)

        # Resize the image
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=self.interp_mode)

        # Calculate padding
        top_pad = (self.target_height - new_height) // 2
        bottom_pad = self.target_height - new_height - top_pad
        left_pad = (self.target_width - new_width) // 2
        right_pad = self.target_width - new_width - left_pad

        # Pad the resized image
        padded_img = cv2.copyMakeBorder(
            resized_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        return padded_img

    def get_transform_init_args_names(self):
        return "target_height", "target_width", "interpolation"


class LowerRightCrop(ImageOnlyTransform):
    def __init__(self, target_height=256, target_width=256, always_apply=True, p=1.0):
        super(LowerRightCrop, self).__init__(always_apply, p)
        self.target_height = target_height
        self.target_width = target_width

    def apply(self, img, **params):
        y_max, x_max = img.shape[:2]
        ymin = y_max - self.target_height
        xmin = x_max - self.target_width
        cropped = img[ymin:y_max, xmin:x_max]
        return cropped

    def get_transform_init_args_names(self):
        return "target_height", "target_width"


