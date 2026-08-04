import cv2
import numpy as np

from general import proj
from general.image_encoder import ImageEncoder
from general.img_utils import crop_image


def select_roi_and_crop(image):
    # Load the image
    if image is None:
        print("Error: Could not load image.")
        return None, None

    # Function to capture ROI
    def draw_rectangle(event, x, y, flags, param):
        nonlocal start_point, end_point, cropping, roi_selected
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)
            cropping = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping:
                end_point = (x, y)
                temp_image = image.copy()
                cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 2)
                cv2.imshow("Image", temp_image)
        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            cropping = False
            roi_selected = True
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("Image", image)

    # Initialize variables
    start_point = (0, 0)
    end_point = (0, 0)
    cropping = False
    roi_selected = False

    # Show the image
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", draw_rectangle)

    # Wait for ROI selection
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        if key == 13 and roi_selected:  # Enter to confirm
            break

    cv2.destroyAllWindows()

    # Crop and return the ROI
    if roi_selected:
        x1, y1 = start_point
        x2, y2 = end_point
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        roi = image[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)
    else:
        print("No ROI selected.")
        return None, None



if __name__ == "__main__":
    encoder = ImageEncoder({}, is_local=True)
    video_path = "/mnt/d/alerts_testing/missing_obj.mp4"
    # image = cv2.imread(proj.resource_path("tests", "images", "bus.jpg"))
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    cropped_roi, coordinates = select_roi_and_crop(image)
    if cropped_roi is not None:
        print("Coordinates of ROI:", coordinates)
        cv2.imshow("Cropped ROI", cropped_roi)
        cv2.waitKey(1000)
        normalized_coordinates = [coordinates[0] / image.shape[1], coordinates[1] / image.shape[0],
                                  coordinates[2] / image.shape[1], coordinates[3] / image.shape[0]]
        crop = encoder.apply_resize(crop_image(image, normalized_coordinates, margins=[0, 0], bgr_map=False))
        desc = encoder.forward_on_crop_list([crop])
        print(f"descriptor: {desc.astype(np.float32).tobytes().hex()}")
        print(f"coordinates: {normalized_coordinates}")

    cv2.destroyAllWindows()