from typing import Dict
import numpy as np

ort_type_to_numpy: Dict = {
    'tensor(float)': np.float32,
    'tensor(half)': np.float16,
    'tensor(float16)': np.float16,
    'tensor(float32)': np.float32,
    'tensor(int8)': np.uint8,
    'tensor(int32)': np.int32,
}
