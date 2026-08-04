import random

import numpy as np


class c_auto_saver:
    """
    Class for saving images automatically for propper fitting.
    Keeps active id lists of limited size of the id number and class type.
    This list is updated every time the step function is called.
    If an ID became inactive (not found for a few frames) and no images was saved for a few frames the image will be saved.
    """

    def __init__(self, fps, max_age):
        self.max_id_list_length = 10  # max number of active ids
        self.min_seconds_since_last_save = 4  # min number of savings in each second, this corresponds to saving time and not necessarily to image time
        self.xyxy_queue_size = 3  # compare to last # detection xyxys

        self.fps = fps
        self.max_frames_not_found = max_age
        self.max_ID = -1
        self.ids_num = []
        self.ids = []
        self.frames_since_last_save = 0
        self.min_frames_since_last_save = self.min_seconds_since_last_save * self.fps
        self.xyxy_queue = xyxy_queue(max_size=self.xyxy_queue_size)

    def step(self, track_results, image):  # for each batch
        if len(track_results):
            current_ids = track_results[:, 4]  # current tracked ids
            xyxys = track_results[:, :4].astype(np.int32)
            new_ids_list = self.get_new_ids(current_ids)  # find which ids are new
            self.initialize_new_ids(new_ids_list)  # initialize id class for them
        else:
            xyxys = []
            current_ids = []

        return_images = self.ids_step(current_ids, image, xyxys, current_ids)  # update current ids

        if len(return_images) > 0:  # if more than one image available choose randomly
            return_image = random.choice(return_images)
        else:
            return_image = None

        return return_image

    def get_new_ids(self, current_ids):  # which of the ids is a new id
        current_max_id = current_ids.max()
        new_ids_mask = current_ids > self.max_ID
        new_ids = current_ids[new_ids_mask]  # new ids list
        if current_max_id > self.max_ID:  # new id present
            self.max_ID = current_max_id
        current_max_id = current_ids.max()
        return new_ids

    def initialize_new_ids(self, new_ids):  # for each new id number create new id class
        for new_id in new_ids:
            if len(self.ids_num) <= self.max_id_list_length:  # if image buffer not full
                self.ids_num.append(new_id)  # push new id number to list
                self.ids.append(IdInformation(self.fps, self.max_frames_not_found, new_id))  # push new id class to list

    def ids_step(self, current_ids, img, xyxys, ids):
        return_images = []
        not_found_ids = np.ones(len(self.ids_num), dtype=bool)  # active ids which are not found this time
        self.frames_since_last_save = self.frames_since_last_save + 1  # general count since last save

        for i, id_n in enumerate(current_ids):  # found ids
            try:  # Will work if the id is in the list (did not reach max_id_list_length)
                index = self.ids_num.index(id_n)
            except Exception as e:
                continue
            not_found_ids[index] = False  # found
            self.ids[index].next_step(img, xyxys, ids)  # update current id with current image

        active_ids = np.logical_not(not_found_ids)  # only active ones, initalize with found ones
        for i, id_ in enumerate(self.ids):  # active ids
            if not_found_ids[i]:
                is_active, do_save, img_and_xyxy = id_.not_found()  # mark not found in frame
                if is_active:  # still active (was found a few frames ago)
                    active_ids[i] = True
                elif self.frames_since_last_save > self.min_frames_since_last_save:  # no image saved for a few frames
                    if do_save and self.xyxy_queue.step(img_and_xyxy):
                        return_images.append(img_and_xyxy.image)  # became inactive and can be saved
                        self.frames_since_last_save = 0

        self.ids_num = [i for indx, i in enumerate(self.ids_num) if active_ids[indx] == True]  # update non active ids
        self.ids = [i for indx, i in enumerate(self.ids) if active_ids[indx] == True]

        return return_images


class IdInformation:
    """
    Class to maintain id information and return an image of the id during its lifetime.
    Maintains a queue of images of the id during its activeness.
    First image always enters the queue and remains in the first index of it.
    Every random amount of seconds the current image enters the queue:
        a. If queue is not full it is pushed inside.
        b. If queue full it randomly replaces on of the images in the queue.
    If the id is not found for a few frames it becomes inactive.
    When an ID becomes inactive a random image is chosen for the queue for saving.
    """

    def __init__(self, fps, max_frames_not_found, id_num):
        self.fps = fps
        self.max_frames_not_found = max_frames_not_found
        self.id_num = id_num

        self.q_max_size = 3  # image queue size for each id
        # self.min_appearance_frames = int(0.5 * self.fps) # minimum number of seconds for saving
        self.current_id_image_queue = []
        self.current_id_xyxy_queue = []
        self.current_id_loss_count = 0
        self.current_id_appearance_count = 0

        self.step = 0
        self.next_save = 0
        self.not_found_count = 0  # active

        self.initial_image = True
        self.change_first = False

    def next_step(self, img, xyxy, ids):
        id_ind = np.where(ids == self.id_num)
        if self.step == 0:  # random save time arrived (or first image)
            self.push(img, xyxy[id_ind])  # enter current image
            self.next_save = random.randint(1, int(self.fps * 3))  # random next saving time
            if self.initial_image:  # initial image
                self.initial_image = False
                self.change_first = True
        self.step = (self.step + 1) % self.next_save
        self.current_id_appearance_count = self.current_id_appearance_count + 1
        self.not_found_count = 0  # active

    def push(self, img, xyxy):
        if self.change_first:  # if only initialization image present
            self.current_id_image_queue[0] = id_data(img, xyxy)  # initialize queue, this image remains here
            self.change_first = False
        elif len(self.current_id_image_queue) < self.q_max_size:  # first images
            self.current_id_image_queue.append(id_data(img, xyxy))  # if queue not full push to it

        else:
            random_pick = random.randint(
                1, self.q_max_size - 1
            )  # quque full, keep first image to not always lose a starting one
            self.current_id_image_queue[random_pick] = id_data(img, xyxy)
        return

    def not_found(self):  # image not found in current frame
        self.not_found_count = self.not_found_count + 1
        if self.not_found_count == self.max_frames_not_found:  # become inactive
            if len(self.current_id_image_queue) > 1:  # not only first frame present
                random_pick = random.randint(
                    0, len(self.current_id_image_queue) - 1
                )  # choose random image of the queue for saving
                return_img_and_xyxy = self.current_id_image_queue[random_pick]
                # if self.current_id_appearance_count >= self.min_appearance_frames:
                #     return False, return_img    #   inactive and save
                return False, True, return_img_and_xyxy  # inactive, save, img
            return False, False, None  # inactive, nosave, None
        return True, False, None  # active, nosave, None


class id_data:
    def __init__(self, image, xyxy):
        self.image = image
        self.xyxy = xyxy[0]


class xyxy_queue:
    """
    A class to maintain the last detection xyxys of ids that their images were saved.
    The last few xyxys are saved in a queue.
    The new xyxy will be checked to see if similar (high iou) to last xyxys.
    """

    def __init__(self, max_size=1, iou_thresh: float = 0.7):
        self.iou_thresh = iou_thresh  # if iou is higher do not use new image
        self.queue_list = []
        self.max_size = max_size
        self.cur_size = 0
        self.oldest_index = 0

    def step(self, img_and_xyxy):
        save = self.check_distinciton(img_and_xyxy.xyxy)
        if save:
            self.push(img_and_xyxy.xyxy)
        return save

    def check_distinciton(self, xyxy):
        max_iou = -1
        for q_xyxy in self.queue_list:
            iou = self.compute_iou(xyxy, q_xyxy)
            max_iou = max(iou, max_iou)

        if max_iou > self.iou_thresh:
            return False

        return True

    def compute_iou(self, xyxy1, xyxy2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        """
        # determine the coordinates of the intersection rectangle
        x_left = max(xyxy1[0], xyxy2[0])
        y_top = max(xyxy1[1], xyxy2[1])
        x_right = min(xyxy1[2], xyxy2[2])
        y_bottom = min(xyxy1[3], xyxy2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        xyxy1_area = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
        xyxy2_area = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth areas - the interesection area
        iou = intersection_area / float(xyxy1_area + xyxy2_area - intersection_area)

        # for numerical stability
        if iou < 0:
            iou = 0.0
        if iou > 1:
            iou = 1.0

        return iou

    def push(self, xyxy):
        if self.cur_size < self.max_size:
            self.cur_size = self.cur_size + 1
            self.queue_list.append(xyxy)
        else:
            self.queue_list[self.oldest_index] = xyxy
            self.oldest_index = (self.oldest_index + 1) % self.max_size
