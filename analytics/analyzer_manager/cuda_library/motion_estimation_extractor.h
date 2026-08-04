#pragma once
#include "common.h"
#include <vector>

#define THREADS_BLOCK_X 32
#define THREADS_BLOCK_Y 32
#define PIX_PER_THREAD   2

inline void generateMorphologicalFilter(int width, int height, int type, uint8_t* filter)
{
    int xCenter = width / 2;
    int yCenter = height / 2;
    int radii = min(xCenter, yCenter);

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            filter[i + width * j] = (uint8_t)((abs(j - yCenter) + abs(i - xCenter)) <= radii);
        }
    }
}

struct MotionEstimator
{
    MotionEstimator(const int _width, const int _height, 
        const int _outWidth, const int _outHeight,
        const int _morphFilterWidth, const int _morphFilterHeight,
        const uint8_t _maxLevel, const float _normFactor, const int _batchSize) : 
        width(_width), height(_height), outWidth(_outWidth), outHeight(_outHeight),
        morphFilterWidth(_morphFilterWidth), morphFilterHeight(_morphFilterHeight), 
        maxLevel(_maxLevel), normFactor(_normFactor), batchSize(_batchSize)
    {
        inSize = width * height;  
        outSize = outWidth * outHeight;
        decimationFactorHeight = (height + outHeight / 2) / outHeight;
        decimationFactorWidth = (width + outWidth / 2) / outWidth;
        d_sumLuma.resize(outSize);
        d_sumDiff.resize(outSize);
        dialted.resize(outSize);
        diffTh.resize(batchSize);
        for(int i = 0; i < batchSize; ++i)
        {
            diffTh[i].resize(outSize);
        }
        d_diffTh.resize(outSize);
        d_frame.resize(inSize);
        d_prevFrame.resize(inSize);
        filter.resize(morphFilterWidth * morphFilterHeight);
        girdSize = {(unsigned) ceil((float)width / (THREADS_BLOCK_X * PIX_PER_THREAD)), (unsigned) ceil((float)height / THREADS_BLOCK_Y), 1};
        gridSize1D = static_cast<unsigned int>(ceil((float)inSize / blockSize1D));
        gridSize1DIcon = static_cast<unsigned int>(ceil((float)outSize / blockSize1DIcon));
        generateMorphologicalFilter(morphFilterWidth, morphFilterHeight, 0, filter.data());
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~MotionEstimator()
    {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void calculateMotionEstimation(uint8_t* meBatch, const uint8_t* d_bgrBatch, const bool*);

    int width{};
    int height{};
    int outWidth{};
    int outHeight{};
    int inSize{};
    int outSize{};
    int batchSize{};
    float normFactor{};
    float maxLevel{};
    bool isFirst = true;
    int morphFilterWidth{};
    int morphFilterHeight{};
    int decimationFactorWidth{};
    int decimationFactorHeight{};
    std::vector<std::vector<uint8_t>> diffTh;
    std::vector<uint8_t> filter;
    std::vector<uint8_t> dialted;
    dim3 girdSize;
    const dim3 blockSize = {THREADS_BLOCK_X, THREADS_BLOCK_Y, 1};
    unsigned int gridSize1D{};
    unsigned int blockSize1D = 512;
    unsigned int gridSize1DIcon{};
    unsigned int blockSize1DIcon = 32;
    cudaStream_t stream{};
    thrust::device_vector<int> d_sumDiff;
    thrust::device_vector<int> d_sumLuma;
    thrust::device_vector<uint8_t> d_diffTh;
    thrust::device_vector<uint8_t> d_frame;
    thrust::device_vector<uint8_t> d_prevFrame;
};

EXPORT uint64_t createMotionEstimator(const int width, const int height, const int outWidth, const int outHeight, const int morphFilterWidth, const int morphFilterHeight, const uint8_t maxLevel, const float normFactor, const int batchSize);
EXPORT void motionEstimation(uint64_t motionEstimatorAddress, uint8_t* me, const uint8_t* d_bgr, const bool* skipFilter);
EXPORT void destroyMotionEstimator(uint64_t motionEstimatorAddress);