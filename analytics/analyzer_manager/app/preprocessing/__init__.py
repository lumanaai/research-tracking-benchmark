from typing import List, Dict

from detection.detector import BaseDetector
from general.analyzer_general import logger
from .preprocessor import PreprocessorBase, PreprocessorCv2Cpu
from general.proj import is_jetson_platform, is_cv2_cuda_available

class PreprocessorFactory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PreprocessorFactory, cls).__new__(cls)
        return cls._instance

    def create(
        self, preprocessor_name: str, args: Dict, full_resolution: List[int], detector: BaseDetector
    ) -> PreprocessorBase:

        logger.info(f"Building {preprocessor_name} preprocessor with the arguments : {args}")
        if preprocessor_name == "cv2":
            if is_jetson_platform() or is_cv2_cuda_available():
                from .cv2_cuda_preproc import PreprocessorCv2CudaCpp # PreprocessorCv2Cuda,
                return PreprocessorCv2CudaCpp(args, full_resolution, detector)
                #return PreprocessorCv2Cuda(args, full_resolution, detector)
            return PreprocessorCv2Cpu(args, full_resolution, detector)
        elif preprocessor_name == "lumana":
            from .cuda_preproc import PreprocessorLumCuda
            return PreprocessorLumCuda(args, full_resolution, detector)
        elif preprocessor_name == "vpi":
            from .vpi_preproc import PreprocessorVpi
            # return PreprocessorVpi(args, full_resolution, detector)
            raise NotImplemented
        else:
            raise NotImplemented