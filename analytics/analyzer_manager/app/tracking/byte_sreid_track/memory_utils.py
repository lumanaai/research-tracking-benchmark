import numpy as np
from typing import Dict, List
from level1.vehicle.vehicle_parsing import  ResizeAndPadToTarget
import albumentations

class ReIDMemoryHandler:
    def __init__(self, num_max_ids, num_max_crops, crop_size: List[int], dtype=np.uint8, resize_func=None):
        self._num_max_ids = num_max_ids # maximal number of ids
        self._num_max_crops = num_max_crops # maximal number of crops per id
        self._data = np.empty([num_max_ids, num_max_crops, *crop_size, 3], dtype=dtype) # max ids, max crops for id, crop size, color_channels
        self._free = set(list(range(num_max_ids)))
        self._crop_index = np.zeros(num_max_ids, dtype=int)
        self.id_to_idx = {}
        self.resize = resize_func

    def _push_new_crop(self, id: int, crop: np.ndarray):
        if id not in self.id_to_idx:
            idx = self._acquire()
            if idx == -1:   # no free buffers
                return False
            self.id_to_idx[id] = idx
        else:
            idx = self.id_to_idx[id]
        crop_idx = self._crop_index[idx] % self._num_max_crops
        if self.resize is not None:
            self._data[idx, crop_idx] = self.resize(image=crop)['image']
        else:
            self._data[idx, crop_idx] = crop
        
        self._crop_index[idx] += 1
        return True

    def _acquire(self) -> int:
        if len(self._free) == 0:
            return -1  # no free buffers
        idx = self._free.pop()
        return idx

    def _release(self, id: int):
        idx = self.id_to_idx[id]
        self._free.add(idx)
        self._crop_index[idx] = 0
        del self.id_to_idx[id]

    def _get_crops_release_id(self, id: int) -> List[np.array]:
        crop_list = []
        idx = self.id_to_idx[id]
        loop_range = min(self._crop_index[idx], self._num_max_crops)
        for i in range(loop_range):
            crop_list.append(self._data[idx, i])
        self._release(id)
        return crop_list
    
    def _is_id_in_memory(self, id: int) -> bool:
        return id in self.id_to_idx
    


class ReIDMemoryBank():
    def __init__(self, memory_banks_args: Dict, reid_models: Dict):
        self._id_to_class_id = {}

        resize_dict = {}
        crop_size_dict = {}
        for i in memory_banks_args:
            resize_dict[i] = None
            crop_size_dict[i] = reid_models[i].args.im_size
            for transform in reid_models[i].transform.transforms:
                if isinstance(transform, ResizeAndPadToTarget) or isinstance(transform, albumentations.augmentations.geometric.resize.Resize):
                    resize_dict[i] = transform
                    break

        self._memory_banks = { key: ReIDMemoryHandler(**memory_banks_args[key], crop_size=crop_size_dict[key], resize_func=resize_dict[key]) for key in memory_banks_args}


    def push_new_crop(self, id: int, crop: np.ndarray, class_id: int):
        success = self._memory_banks[class_id]._push_new_crop(id, crop)
        if success: #   memory not full
            self._id_to_class_id[id] = class_id
        return success
    
    def release(self, id: int):
        class_id = self._id_to_class_id[id]
        self._memory_banks[class_id]._release(id)
        del self._id_to_class_id[id]

    def get_crops_release_id(self, id: int) -> List[np.array]:
        class_id = self._id_to_class_id[id]
        del self._id_to_class_id[id]
        return self._memory_banks[class_id]._get_crops_release_id(id)
    
    def is_id_in_memory(self, id: int) -> bool:
        return id in self._id_to_class_id
    

        

