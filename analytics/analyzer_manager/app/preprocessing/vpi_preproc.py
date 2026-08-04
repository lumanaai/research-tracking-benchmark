from typing import List

import numpy as np

from detection.detector import BaseDetector
from general.core import AnalyticImage
from preprocessing import PreprocessorBase
import vpi


class PreprocessorVpi(PreprocessorBase):
    use_cuda_extract = True