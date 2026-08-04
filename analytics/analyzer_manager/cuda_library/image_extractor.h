#pragma once
#include "common.h"
#include <thrust/device_vector.h>

struct ImageExtractor
{
    ImageExtractor(uint8_t* d_bgrBatch, const int _width, const int _height) : d_bgr(d_bgrBatch), width(_width), height(_height)
    {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        getGirdSize2D(gridSize2D, blockSize2D, width, height, 1);
        size = width * height;
        d_yuv420.resize(size * 1.5);
    }

    ~ImageExtractor()
    {
        cudaStreamDestroy(stream);
    }
    
    int size{};
    int width{};
    int height{};
    uint8_t* d_bgr{};
    thrust::device_vector<uint8_t> d_yuv420{};
    cudaStream_t stream{};
    dim3 gridSize2D;
    dim3 blockSize2D = {16, 8, 1};
};

EXPORT uint64_t createImageExtractor(uint8_t* d_bgrBatch, const int width, const int height);
EXPORT void destroyImageExtractor(uint64_t imageExtractorAddress);
EXPORT void imageExtractBGR(uint64_t imageExtractorAddress, uint8_t* bgr, const uint8_t* yuv420, const int frameIdx);
EXPORT void imageExtractBGRFromNv12(uint64_t imageExtractorAddress, uint8_t* bgr, const uint8_t* nv12, const int frameIdx);
EXPORT void synchronizeCudaStream(uint64_t imageExtractorAddress);
