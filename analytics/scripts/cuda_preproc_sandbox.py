import time
import numpy as np
import cupy as cp
from ctypes import POINTER, c_uint8, c_int, c_ulonglong, py_object, c_void_p, c_float, c_bool
import ctypes
from general.proj import cuda_path
import cv2
import general.proj as proj
def run_transform_image():


    # definitions and arguments
    batch_size = 4
    imgsz = [384, 640]
    resize_factor = 0.5
    antialias = True
    normalize = True
    is_half = True


    # memory allocations
    dtype = np.float16 if is_half else np.float32
    image_mem = np.zeros((batch_size, *imgsz, 3), dtype=np.uint8)
    internal_mem =  cp.zeros((batch_size, 3, *imgsz), dtype=dtype)

    # image to load
    image = cv2.imread(proj.resource_path("tests", "images", "tampering", "regular_0.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # cdll load
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

    # helper functions
    def c_ptr(array, c_type):
        return array.ctypes.data_as(POINTER(c_type))

    def init_resize_kernel(resize_factor):
        sigma = 1 / (3 * resize_factor)
        kernel_size = 4 * sigma
        if kernel_size % 2 != 1:  # if not uneven round to next uneven
            kernel_size = (np.floor(kernel_size / 2) + np.floor(kernel_size % 2)) * 2 + 1
        kernel_size = int(kernel_size)
        return cv2.getGaussianKernel(kernel_size, sigma).astype(np.float32)


    resize_kernel = init_resize_kernel(resize_factor)
    kernel_size = np.shape(resize_kernel)[0]
    in_height = image.shape[0]
    in_width = image.shape[1]
    cuda_parser_address = cuda_lib.createCudaParser(
        c_ptr(resize_kernel, c_float),
        c_bool(antialias),
        c_bool(normalize),
        c_int(kernel_size),
        c_int(batch_size),
        c_int(in_height),
        c_int(in_width),
        c_int(imgsz[0]),
        c_int(imgsz[1]),
        c_bool(is_half),
    )
    image_np = np.array([image.copy(), image.copy(),image.copy(), image.copy()])
    image_gpu = cp.asarray(image_np)

    tries = 10
    warmup = 5
    for i in range(warmup):
        cuda_lib.parseImages(
            c_ulonglong(cuda_parser_address),
            c_void_p(int(internal_mem.data)),
            c_ptr(image_mem, c_uint8),
            c_void_p(int(image_gpu.data)),
        )
    image_1 = image_mem[1].copy()

    t0 = time.time()
    for i in range(tries):
        cuda_lib.parseImages(
            c_ulonglong(cuda_parser_address),
            c_void_p(int(internal_mem.data)),
            c_ptr(image_mem, c_uint8),
            c_void_p(int(image_gpu.data)),
        )
    print(f"proprietery CUDA Time: {time.time() - t0}")

    for i in range(warmup):
        image_2 = cv2.resize(image, (imgsz[1], imgsz[0]), interpolation=cv2.INTER_AREA)
    t0 = time.time()
    for i in range(tries):
        for b in range(batch_size):
            image_2 = cv2.resize(image, (imgsz[1], imgsz[0]), interpolation=cv2.INTER_AREA)

    print(f"OpenCV Time: {time.time() - t0}")

    gpu_mat_resized  = cv2.cuda_GpuMat()
    gpu_mat_resized.upload(np.zeros((imgsz[0], imgsz[1], 3), dtype=np.uint8))
    gpu_mat_float = cv2.cuda.GpuMat()
    gpu_mat_float.upload(np.zeros((imgsz[0], imgsz[1], 3), dtype=np.float16))
    gpu_mat =  cv2.cuda.GpuMat()
    gpu_mat.upload(image)

    for i in range(warmup):

        cv2.resize(gpu_mat, (imgsz[1], imgsz[0]), interpolation=cv2.INTER_AREA, dst=gpu_mat_resized)
        gpu_mat_resized.convertTo(gpu_mat_float, cv2.CV_16F, 1 / 255.0)
        image_3 = gpu_mat_resized.download()
    t0 = time.time()
    for i in range(tries):
        for b in range(batch_size):
            image_2 = cv2.resize(image, (imgsz[1], imgsz[0]), interpolation=cv2.INTER_AREA)

    print(f"OpenCV CUDA Time: {time.time() - t0}")


    #compare accuracy
    diff_image = image_1.astype(float) - image_2.astype(float)
    diff_n = np.sqrt(np.mean(diff_image**2))
    print(f"RMS diff 1-2: {diff_n}")

    diff_image = image_1.astype(float) - image_3.astype(float)
    diff_n = np.sqrt(np.mean(diff_image**2))
    print(f"RMS diff 1-3: {diff_n}")
