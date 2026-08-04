#pragma once
#include "common.h"

static int id = 0;

struct NVJPGEncoder
{
    NVJPGEncoder()
    {
        encoderName = "nvjpg" + std::to_string(id++);
        encoder = NvJPEGEncoder::createJPEGEncoder(encoderName.c_str());
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~NVJPGEncoder()
    {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    unsigned long encode(std::vector<unsigned char>& jpegBytes, const uint8_t* img, const int width, const int height, const int quality);
    unsigned long nvJpegEncode(std::vector<unsigned char>& jpegBytes, const uint8_t* image, const int width, const int height, const int quality);
 
    cudaStream_t stream{};
    NvJPEGEncoder* encoder{};
    std::string encoderName{};
};

EXPORT uint64_t createEncoder();
EXPORT PyObject* nvJpegEncode(uint64_t encoderAddress, const uint8_t* image, const int width, const int height, const int quality);
EXPORT void destroyEncoder(uint64_t encoderAddress);