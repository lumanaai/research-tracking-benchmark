import cv2


def check_opencv_cuda():
    print("OpenCV version:", cv2.__version__)

    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count == 0:
            print("CUDA is not enabled or no CUDA-capable device detected.")
        else:
            print(f"{count} CUDA-capable device(s) detected.")
            for i in range(count):
                device_info = cv2.cuda
                print(
                    f"Device {i}: Compute Capability: {device_info.majorVersion()}.{device_info.minorVersion()}")
    except AttributeError:
        print("OpenCV is not compiled with CUDA support.")


if __name__ == "__main__":
    check_opencv_cuda()
