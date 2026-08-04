#pragma once
#include "common.h"

struct MotionEstimator
{
    MotionEstimator(uint8_t* firstFrame, 
        const int _width, const int _height, 
        const int _l1Width, const int _l1Height,
        const int _outWidth, const int _outHeight) : 
        width(_width), height(_height), outWidth(_outWidth), outHeight(_outHeight), l1Width(_l1Width), l1Height(_l1Height)
    {
        inSize = width * height;  
        l1Size = l1Width * l1Height;
        outSize = outWidth * outHeight;
        xScaleL1 = (float)width / l1Width;
        yScaleL1 = (float)height / l1Height;
        xScale = (float)l1Width / outWidth;
        yScale = (float)l1Height / outHeight;
        d_me.resize(outSize);
        d_frame.resize(inSize);
        d_prevFrame.resize(inSize);
        d_resizedDiff.resize(l1Size);
        getGirdSize2D(gridSizeResize, blockSize, outWidth, outHeight, 1);
        getGirdSize2D(gridSizeResizeL1, blockSize, l1Width, l1Height, 1);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMemcpyAsync(rawPtr(d_frame), firstFrame, inSize * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    }
    ~MotionEstimator()
    {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void calculateMotionEstimation(uint8_t* me, uint8_t* imageResized, const uint8_t* frame);

    int width{};
    int height{};
    int l1Width{}; 
    int l1Height{};
    int outWidth{};
    int outHeight{};
    int inSize{};
    int l1Size{};
    int outSize{};
    float xScale{};
    float yScale{};
    float xScaleL1{};
    float yScaleL1{};
    dim3 gridSizeResize{};
    dim3 gridSizeResizeL1{};
    const dim3 blockSize = {16, 8, 1};
    cudaStream_t stream{};
    thrust::device_vector<uint8_t> d_me;
    thrust::device_vector<uint8_t> d_frame;
    thrust::device_vector<uint8_t> d_prevFrame;
    thrust::device_vector<uint8_t> d_resizedDiff;
};

EXPORT uint64_t createMotionEstimator(uint8_t* firstFrame, const int l1Width, const int l1Height, const int width, const int height, const int outWidth, const int outHeight);
EXPORT void motionEstimationCuda(uint64_t motionEstimatorAddress, uint8_t* me, uint8_t* imageResized, const uint8_t* frame);
EXPORT void synchronizeCudaStream(uint64_t motionEstimatorAddress);
EXPORT void destroyMotionEstimator(uint64_t motionEstimatorAddress);