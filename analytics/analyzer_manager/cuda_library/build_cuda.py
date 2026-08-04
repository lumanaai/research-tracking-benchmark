import subprocess
import os

from pathlib import Path

files_to_compile = [
    "parse_images_cuda",
    "image_extractor",
    "motion_estimator",
    "motion_estimation_extractor",
]

def is_cuda_available():
    try:
        import cv2

        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except Exception:
        return False

def print_subprocess_results(ret_code: subprocess.CompletedProcess):
    if ret_code.stderr:
        print(ret_code.stderr.decode("UTF-8"))
    if ret_code.stdout:
        print(ret_code.stdout.decode("UTF-8"))


def build_all(nvcc_path: str = "nvcc"):
    cuda_command = [nvcc_path, "-Xcompiler", "-fPIC", "--shared", "-o"]
    file_path = Path(__file__).parent
    for _file in files_to_compile:
        so_file = str(file_path.joinpath(_file + ".so"))
        cu_file = str(file_path.joinpath(_file + ".cu"))
        print(f"compiling {_file}")
        sp_out = subprocess.run(cuda_command + [so_file, "-O3", cu_file])
        print_subprocess_results(sp_out)
    if os.environ.get("HOST", "").lower() == "orin" or os.environ.get("CV2_GPU_ENABLED", "false").lower() == "true" :
    
        _file = "parse_images_cv2"
        file_path = Path(__file__).parent
        cmd = [ "g++",
                "-shared",
                "-o",
                str(file_path.joinpath(_file + ".so")),
                "-fPIC",
                str(file_path.joinpath(_file + ".cpp")),
                "-I/usr/local/include/opencv4",
                "-L/usr/local/lib",
                "-lopencv_cudawarping",
                "-lopencv_core",
                "-lopencv_cudaarithm",
                "-lopencv_cudaimgproc",
                "-lopencv_cudev",
                "-lopencv_imgproc",
            ]

        # Run the command and capture the output using os.popen()
        print(f"Compiling {_file}")
        sp_out = subprocess.run(cmd)
        print_subprocess_results(sp_out)


if __name__ == "__main__":
    build_all()
