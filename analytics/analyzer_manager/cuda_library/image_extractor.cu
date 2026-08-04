#include "image_extractor.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void YUV420ToBGRKernel(unsigned char* bgr, const unsigned char* yuv420, const int height, const int width, const int size)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    const int idx = y * width + x;
    const int halfWidth = width / 2;
    const float nY = yuv420[idx];
    const float nU = yuv420[(y / 2) * halfWidth + x / 2 + size] - 128.0f;
    const float nV = yuv420[(y / 2) * halfWidth + x / 2 + size + size / 4] - 128.0f;
    bgr[idx * 3] = saturate_cast(nY + 2.032f * nU);
    bgr[idx * 3 + 1] = saturate_cast(nY - 0.394f * nU - 0.581f * nV);
    bgr[idx * 3 + 2] = saturate_cast(nY + 1.140f * nV);
}


__global__ void NV12ToBGRKernel(unsigned char* bgr, const unsigned char* nv12, const int height, const int width, const int size) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int y_index = y * width + x;
    const int uv_start = width * height;
    const int uv_index = (y / 2) * width + (x & ~1);  // Even index for UV pair

    const float Y = nv12[y_index];
    const float U = nv12[uv_start + uv_index] - 128.0f;
    const float V = nv12[uv_start + uv_index + 1] - 128.0f;

    const int bgr_index = y_index * 3;
    bgr[bgr_index + 0] = saturate_cast(Y + 2.032f * U);
    bgr[bgr_index + 1] = saturate_cast(Y - 0.394f * U - 0.581f * V);
    bgr[bgr_index + 2] = saturate_cast(Y + 1.140f * V);
}



uint64_t createImageExtractor(uint8_t* d_bgrBatch, const int width, const int height)
{
    ImageExtractor* imageExtractor = new ImageExtractor(d_bgrBatch, width, height);
    return uint64_t(imageExtractor);
}

void destroyImageExtractor(uint64_t imageExtractorAddress)
{
	ImageExtractor* imageExtractor = reinterpret_cast<ImageExtractor*>(imageExtractorAddress);
	delete imageExtractor;
}

void imageExtractBGR(uint64_t imageExtractorAddress, uint8_t* bgr, const uint8_t* yuv420, const int frameIdx)
{
    ImageExtractor* imageExtractor = reinterpret_cast<ImageExtractor*>(imageExtractorAddress);
    uint8_t* d_bgrFrame = imageExtractor->d_bgr + imageExtractor->size * 3 * frameIdx;
    CUDA_CHECK(cudaMemcpyAsync(rawPtr(imageExtractor->d_yuv420), yuv420, imageExtractor->size * 1.5 * sizeof(uint8_t), cudaMemcpyHostToDevice, imageExtractor->stream));
    YUV420ToBGRKernel<<<imageExtractor->gridSize2D, imageExtractor->blockSize2D, 0, imageExtractor->stream>>>(d_bgrFrame, rawPtr(imageExtractor->d_yuv420), imageExtractor->height, imageExtractor->width, imageExtractor->size);
    CUDA_CHECK(cudaMemcpyAsync(bgr, d_bgrFrame, imageExtractor->size * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost, imageExtractor->stream));
}


void imageExtractBGRFromNv12(uint64_t imageExtractorAddress, uint8_t* bgr, const uint8_t* nv12, const int frameIdx)
{
    ImageExtractor* imageExtractor = reinterpret_cast<ImageExtractor*>(imageExtractorAddress);
    uint8_t* d_bgrFrame = imageExtractor->d_bgr + imageExtractor->size * 3 * frameIdx;
    CUDA_CHECK(cudaMemcpyAsync(rawPtr(imageExtractor->d_yuv420), nv12, imageExtractor->size * 1.5 * sizeof(uint8_t), cudaMemcpyHostToDevice, imageExtractor->stream));
    NV12ToBGRKernel<<<imageExtractor->gridSize2D, imageExtractor->blockSize2D, 0, imageExtractor->stream>>>(d_bgrFrame, rawPtr(imageExtractor->d_yuv420), imageExtractor->height, imageExtractor->width, imageExtractor->size);
    CUDA_CHECK(cudaMemcpyAsync(bgr, d_bgrFrame, imageExtractor->size * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost, imageExtractor->stream));
}

void synchronizeCudaStream(uint64_t imageExtractorAddress)
{
	ImageExtractor* imageExtractor = reinterpret_cast<ImageExtractor*>(imageExtractorAddress);
	CUDA_CHECK(cudaStreamSynchronize(imageExtractor->stream));
}