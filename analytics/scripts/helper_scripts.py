import os
import shutil

import cv2
import numpy as np


def test_dedup_folder():
    from general.image_encoder import ImageEncoder
    import glob
    batch_size = 100
    match_th = 0.8
    image_encoder = ImageEncoder({}, is_local=True)
    search_folder = "/mnt/c/Temp/wc/*.jpg"
    out_folder = "/mnt/c/Temp/wc/dedup"
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder, exist_ok=True)
    files = glob.glob(search_folder)

    # work on 100 crops at a time
    descriptors = np.empty((0,image_encoder.feature_length))
    for i in range(0, len(files), batch_size):
        iter_files = files[i:i + batch_size]
        crop_list = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in iter_files]
        encodings = image_encoder.forward_on_crop_list(crop_list)
        descriptors = np.vstack((descriptors,encodings))

    matchings = descriptors @ descriptors.T

    # dedup
    remaining_idx = set(list(range(len(descriptors))))
    while remaining_idx:
        idx = remaining_idx.pop()
        to_remove = set(np.argwhere(matchings[idx] > match_th).flatten())
        remaining_idx -= to_remove
        shutil.copy2(files[idx], os.path.join(out_folder, os.path.basename(files[idx])))
