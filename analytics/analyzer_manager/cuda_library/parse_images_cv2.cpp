#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

using namespace cv;

extern "C" void process_image_from_gpu_memory(unsigned char* gpu_ptr, int in_width, int in_height, int out_width, int out_height, bool normalize, bool antialias,
                                     unsigned char* work_mem, unsigned char* cpu_bgr_mem, float* cpu_processed) {
    // Step 1: Create a GpuMat from the GPU memory pointer
    cuda::GpuMat gpuImg(in_height, in_width, CV_8UC3, gpu_ptr);

    int bgr_offset = out_height * out_width * 3 * sizeof(unsigned char);
    int rgb_offset = 0;
    int float_offset  = bgr_offset;

    // Step 2: Resize the image to a new size (let's say 640x480 for this example)
    cuda::GpuMat gpuResized(out_height, out_width, CV_8UC3, (unsigned char*)(work_mem + bgr_offset));
    if (antialias){
        cuda::resize(gpuImg, gpuResized, Size(out_width, out_height), 0, 0, INTER_AREA);
    } else {
        cuda::resize(gpuImg, gpuResized, Size(out_width, out_height), 0, 0, INTER_LINEAR);
    }
    Mat cpu_bgrMat(out_height, out_width, CV_8UC3, cpu_bgr_mem);
    gpuResized.download(cpu_bgrMat);

    cuda::GpuMat gpuRGB(out_height, out_width, CV_8UC3, work_mem + rgb_offset);
    cuda::cvtColor(gpuResized, gpuRGB, COLOR_BGR2RGB);

    // Step 3: Convert the resized image to float and normalize
    cuda::GpuMat gpuFloat(out_height, out_width, CV_32FC3, work_mem + float_offset);

    float factor = 1.0;
    if (normalize) {
        factor = 1.0 / 255.0;
    }
    gpuRGB.convertTo(gpuFloat, CV_32F, factor); // Normalize to range [0,1
    Mat cpu_processedMat(out_height, out_width, CV_32FC3, cpu_processed);

    gpuFloat.download(cpu_processedMat);
}
