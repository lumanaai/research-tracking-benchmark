import ctypes
import threading
from ctypes import POINTER, c_uint8, c_int, c_ulonglong, py_object, c_void_p, c_float, c_bool
import numpy as np
import cupy as cp
from .proj import cuda_path


def c_ptr(array, c_type):
    return array.ctypes.data_as(POINTER(c_type))


class ImageExtractor:
    def __init__(self, batch_size, resolution):
        self.batch_size = batch_size
        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.gpu_mem = cp.zeros(batch_size * resolution[1] * resolution[0] * 3, dtype=np.uint8)
        ImageExtractorIF()
        self.image_extractor_address = ImageExtractorIF.generate_extractor(
            resolution[0], resolution[1], batch_size, int(self.gpu_mem.data)
        )
        self.set_nv12(False)

    def extract_images_i420(self, image_batch, out: np.array = None):
        ImageExtractorIF().run_image_extract(
            self.image_extractor_address, image_batch, self.image_width, self.image_height, out
        )

    def extract_images_nv12(self, image_batch, out: np.array = None):
        ImageExtractorIF().run_image_extract_nv12(
            self.image_extractor_address, image_batch, self.image_width, self.image_height, out
        )

    def update_inputs_sizes(self, resolution=None, batch_size=None):
        if resolution is not None:
            self.image_width = resolution[0]
            self.image_height = resolution[1]
        if batch_size is not None:
            self.batch_size = batch_size
        del self.gpu_mem
        self.gpu_mem = cp.zeros(self.batch_size * resolution[1] * resolution[0] * 3, dtype=np.uint8)
        ImageExtractorIF().destroy(self.image_extractor_address)
        self.image_extractor_address = ImageExtractorIF.generate_extractor(
            resolution[0], resolution[1], self.batch_size, int(self.gpu_mem.data)
        )

    def __del__(self):
        ImageExtractorIF().destroy(self.image_extractor_address)

    def set_nv12(self, is_nv12: bool):
        self.is_nv12 = is_nv12
        if is_nv12:
            self.extract_images = self.extract_images_nv12
        else:
            self.extract_images = self.extract_images_i420



class ImageExtractorIF(object):
    _instance = None
    cuda_lib: ctypes.CDLL
    init_loc = threading.Lock()

    def __new__(cls, use_static_detector: bool = False):
        with cls.init_loc:
            if cls._instance is None:
                cls._instance = super(ImageExtractorIF, cls).__new__(cls)
                cuda_lib = ctypes.CDLL(cuda_path("image_extractor.so"))
                cuda_lib.createImageExtractor.restype = c_ulonglong
                cuda_lib.createImageExtractor.argstype = (POINTER(c_uint8), c_int, c_int)
                cuda_lib.imageExtractBGR.argstype = (c_ulonglong, POINTER(c_uint8), POINTER(c_uint8), c_int)
                cuda_lib.imageExtractBGRFromNv12.argstype = (c_ulonglong, POINTER(c_uint8), POINTER(c_uint8), c_int)
                cls.cuda_lib = cuda_lib
        return cls._instance

    @classmethod
    def generate_extractor(cls, image_width, image_height, batch_size, gpu_mem_ptr):
        assert image_width > 0, f"Invalid width {image_width}"
        assert image_height > 0, f"Invalid height {image_height}"
        assert batch_size > 0, f"Invalid batch size {batch_size}"

        image_extractor_address = cls.cuda_lib.createImageExtractor(
            c_void_p(gpu_mem_ptr), c_int(image_width), c_int(image_height)
        )
        return image_extractor_address

    @staticmethod
    def run_image_extract(image_extractor_address, image_batch, image_width, image_height, out: np.array = None):
        for i, image in enumerate(image_batch):
            if out is None:
                out = np.ndarray(shape=(image_height, image_width, 3), dtype=np.uint8)
            ImageExtractorIF.cuda_lib.imageExtractBGR(
                c_ulonglong(image_extractor_address),
                c_ptr(out, c_uint8),
                ctypes.cast(image.frame, POINTER(c_uint8)),
                c_int(i),
            )
            image.frame = out
        ImageExtractorIF.cuda_lib.synchronizeCudaStream(c_ulonglong(image_extractor_address))


    @staticmethod
    def run_image_extract_nv12(image_extractor_address, image_batch, image_width, image_height, out: np.array = None):
        for i, image in enumerate(image_batch):
            if out is None:
                out = np.ndarray(shape=(image_height, image_width, 3), dtype=np.uint8)
            ImageExtractorIF.cuda_lib.imageExtractBGRFromNv12(
                c_ulonglong(image_extractor_address),
                c_ptr(out, c_uint8),
                ctypes.cast(image.frame, POINTER(c_uint8)),
                c_int(i),
            )
            image.frame = out
        ImageExtractorIF.cuda_lib.synchronizeCudaStream(c_ulonglong(image_extractor_address))

    @staticmethod
    def initialize():
        ImageExtractorIF()

    @staticmethod
    def destroy(image_extractor_address):
        ImageExtractorIF.cuda_lib.destroyImageExtractor(c_ulonglong(image_extractor_address))

    @staticmethod
    def cuda_image_extract(image_batch, image_width, image_height, batch_size):
        gpu_mem = cp.zeros(batch_size * image_height * image_width * 3, dtype=np.uint8)
        image_extractor_address = ImageExtractorIF.generate_extractor(
            image_width, image_height, batch_size, int(gpu_mem)
        )
        ImageExtractorIF.run_image_extract(image_batch, image_width, image_height)
        ImageExtractorIF.destroy(image_extractor_address)
        return gpu_mem


class NvJpegEncoder(object):
    _instance = None
    cuda_lib: ctypes.CDLL
    init_loc = threading.Lock()

    def __new__(cls, use_static_detector: bool = False):
        with cls.init_loc:
            if cls._instance is None:
                cls._instance = super(NvJpegEncoder, cls).__new__(cls)
                cuda_lib = ctypes.CDLL(cuda_path("nvJpeg_encoder.so"))
                cuda_lib.createEncoder.restype = c_ulonglong
                cuda_lib.nvJpegEncode.argtypes = (c_ulonglong, POINTER(c_uint8), c_int, c_int, c_int)
                cuda_lib.nvJpegEncode.restype = py_object
                cuda_lib.destroyEncoder.argtypes = (c_ulonglong,)
                cls.cuda_lib = cuda_lib
        return cls._instance

    @staticmethod
    def create_nvjpeg_encoder():
        return NvJpegEncoder.cuda_lib.createEncoder()

    @staticmethod
    def encode(encoder_address, image, quality=75):
        quality = min(100, quality)
        jpeg_bytes = NvJpegEncoder.cuda_lib.nvJpegEncode(
            c_ulonglong(encoder_address),
            image.ctypes.data_as(POINTER(c_uint8)),
            c_int(image.shape[1]),
            c_int(image.shape[0]),
            c_int(quality),
        )
        return jpeg_bytes

    @staticmethod
    def destroy(encoder_address):
        NvJpegEncoder.cuda_lib.destroyEncoder(c_ulonglong(encoder_address))

    @staticmethod
    def initialize():
        NvJpegEncoder()


class ImageParser(object):
    _instance = None
    cuda_lib: ctypes.CDLL
    init_loc = threading.Lock()

    def __new__(cls, use_static_detector: bool = False):
        with cls.init_loc:
            if cls._instance is None:
                cls._instance = super(ImageParser, cls).__new__(cls)
                cuda_lib = ctypes.CDLL(cuda_path("parse_images_cuda.so"))
                cuda_lib.createCudaParser.restype = c_ulonglong
                cuda_lib.createCudaParser.argstype = (
                    c_void_p,
                    c_bool,
                    c_bool,
                    c_int,
                    c_int,
                    c_int,
                    c_int,
                    c_int,
                    c_int,
                    c_bool,
                )
                cuda_lib.parseImages.argstype = (
                    c_ulonglong,
                    POINTER(c_float),
                    POINTER(c_uint8),
                    POINTER(c_uint8),
                )
                cuda_lib.destroyCudaParser.argstype = (c_ulonglong,)
                cls.cuda_lib = cuda_lib
        return cls._instance

    @staticmethod
    def initialize():
        ImageParser()

    @staticmethod
    def create_parser(
        resize_kernel,
        antialias: bool,
        normalize: bool,
        kernel_size: int,
        batch_size: int,
        in_height: int,
        in_width: int,
        height: int,
        width: int,
        is_half: bool,
    ):
        cuda_parser_address = ImageParser.cuda_lib.createCudaParser(
            c_ptr(resize_kernel, c_float),
            c_bool(antialias),
            c_bool(normalize),
            c_int(kernel_size),
            c_int(batch_size),
            c_int(in_height),
            c_int(in_width),
            c_int(height),
            c_int(width),
            c_bool(is_half),
        )
        return cuda_parser_address

    @staticmethod
    def parse_images(cuda_parser_address, images_tensor_ptr, images_np, images_gpu_ptr):
        ImageParser.cuda_lib.parseImages(
            c_ulonglong(cuda_parser_address),
            c_void_p(images_tensor_ptr),
            c_ptr(images_np, c_uint8),
            c_void_p(images_gpu_ptr),
        )

    @staticmethod
    def destroy(cuda_parser_address):
        ImageParser.cuda_lib.destroyCudaParser(c_ulonglong(cuda_parser_address))


class ImageParserCV2(object):
    def __init__(self, full_resolution, detector_resolution, antialias: bool, normalize: bool):
        self.antialias = antialias
        self.normalize = normalize
        self.in_height = full_resolution[1]
        self.in_width = full_resolution[0]
        self.height = detector_resolution[0]
        self.width =  detector_resolution[1]
        uint8_size = np.dtype(np.uint8).itemsize
        float_size = np.dtype(np.float32).itemsize

        self.work_mem = cp.zeros(3 * self.height * self.width * (uint8_size + float_size), dtype=np.uint8)

        self.cuda_lib = ctypes.CDLL(cuda_path("parse_images_cv2.so"))
        self.cuda_lib.process_image_from_gpu_memory.argstype = (
            POINTER(c_uint8),
            c_int,
            c_int,
            c_int,
            c_int,
            c_bool,
            c_bool,
            POINTER(c_uint8),
            POINTER(c_uint8),
            POINTER(c_float),
        )

    def parse_images(self,gpu_mem, cpu_bgr_mem, cpu_processed):
        self.cuda_lib.process_image_from_gpu_memory(
            c_void_p(int(gpu_mem)),
            self.in_width,
            self.in_height,
            self.width,
            self.height,
            self.normalize,
            self.antialias,
            c_void_p(int(self.work_mem.data)),
            c_ptr(cpu_bgr_mem, c_uint8),
            c_ptr(cpu_processed, c_float),
        )
