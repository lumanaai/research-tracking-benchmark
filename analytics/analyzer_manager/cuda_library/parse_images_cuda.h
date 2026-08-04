#pragma once
#include <vector>
#include "common.h"

struct CudaImageParser
{
    CudaImageParser(const int _kernelSize, const bool _antialias, const bool _normalize, const int _batchSize, const int _inHeight, const int _inWidth, const int _outHeight, const int _outWidth, const bool _isHalf) :
        kernelSize(_kernelSize), antialias(_antialias), normalize(_normalize), 
        batchSize(_batchSize), inHeight(_inHeight), inWidth(_inWidth), 
        outHeight(_outHeight), outWidth(_outWidth), isHalf(_isHalf)
    {
        inSize = inWidth * inHeight;
        outSize = outWidth * outHeight;
        xScale = (float)inWidth / outWidth;
	    yScale = (float)inHeight / outHeight;
	    d_buff.resize(inSize * 3);
        gridSize = (inSize + blockSize - 1) / blockSize;
	    gridSizeSmall = (outSize + blockSize - 1) / blockSize;
        getGirdSize2D(gridSize2D, blockSize2D, inWidth, inHeight, 1);
        getGirdSize2D(gridSize2DResize, blockSize2D, outWidth, outHeight, 1);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~CudaImageParser()
    {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    template<class T> void parseImages(T* d_outputs, uint8_t* outputs, uint8_t* d_bgrBatch);
    
    cudaStream_t stream;
    thrust::device_vector<uint8_t> d_buff;
    int inWidth{};
    int inHeight{};
    int outWidth{};
    int outHeight{};
    float xScale{};
    float yScale{};
    int inSize{};
    int outSize{};
    int kernelSize{};
    int batchSize{};
    bool antialias{};
    bool normalize{};
    const unsigned int blockSize = 512;
    const dim3 blockSize2D = { 16U, 8U, 1U };
    unsigned int gridSize{};
    unsigned int gridSizeSmall{};
    bool isHalf;
    dim3 gridSize2D{};
    dim3 gridSize2DResize{};
};

EXPORT uint64_t createCudaParser(const float* gaussianFilter, const bool antialias, const bool normalize, const int kernelSize, const int batchSize, const int inHeight, const int inWidth, const int outHeight, const int outWidth, const bool isHalf);
EXPORT void parseImages(uint64_t cudaParserAddress, void* d_outputsNormalized, uint8_t* outputs, uint8_t* d_bgrBatch);
EXPORT void destroyCudaParser(uint64_t cudaParserAddress);