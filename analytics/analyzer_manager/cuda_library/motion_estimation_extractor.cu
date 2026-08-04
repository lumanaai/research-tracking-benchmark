#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "motion_estimation_extractor.h"

__global__ void sumOfDiffsSharedKernel(int* sumDiffs, int* sumImg, const uint8_t* frame, const uint8_t* prevFrame, const int width, const int height, const int decimationFactorWidth, const int decimationFactorHeight, const int outWidth, const int outHeight) 
{
    __shared__ int s_diffs[4];
    __shared__ int s_pixs[4];
    __shared__ int s_idx[4];

    int x = (blockIdx.x * blockDim.x + threadIdx.x) * PIX_PER_THREAD;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int source_index = y * width + x;
    int targetIndex = floorf(y / decimationFactorHeight) * outWidth + floorf(x / decimationFactorWidth);

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        s_idx[0] = targetIndex;
        s_pixs[0] = 0;
        s_diffs[0] = 0;
    }
    if (threadIdx.x == THREADS_BLOCK_X - 1 && threadIdx.y == 0)
    {
        s_idx[1] = targetIndex;
        s_pixs[1] = 0;
        s_diffs[1] = 0;
    }
    if (threadIdx.x == 0 && threadIdx.y == THREADS_BLOCK_Y - 1)
    {
        s_idx[2] = targetIndex;
        s_pixs[2] = 0;
        s_diffs[2] = 0;
    }
    if (threadIdx.x == THREADS_BLOCK_X - 1 && threadIdx.y == THREADS_BLOCK_Y - 1)
    {
        s_idx[3] = targetIndex;
        s_pixs[3] = 0;
        s_diffs[3] = 0;
    }
    __syncthreads();

    if (y < height && x < width)
    {
        int pixValue = 0;
        int sumDiff = 0;
        for (int i = 0; i < PIX_PER_THREAD; i++)
        {
            pixValue += frame[source_index + i];
            sumDiff += abs((int)frame[source_index + i] - (int)prevFrame[source_index + i]);
        }

        for (int i = 0; i < 4; i++)
        {
            if (s_idx[i] == targetIndex)
            {
                atomicAdd(&s_diffs[i], sumDiff);
                atomicAdd(&s_pixs[i], pixValue);
                break;
            }
        }

    }
    __syncthreads();

    if (threadIdx.x < 4) 
	{
        atomicAdd(&sumImg[s_idx[threadIdx.x]], s_pixs[threadIdx.x]);
        atomicAdd(&sumDiffs[s_idx[threadIdx.x]], s_diffs[threadIdx.x]);
    }
}

int applyMorphFilter(uint8_t* image, int width, int height, uint8_t* filter, int filterWidth, int filterHeight, int type, uint8_t* result)
{
    // apply the filter
    int filter_sum = 0;
    for (int x = 0; x < filterWidth; x++)
    {
		for (int y = 0; y < filterHeight; y++)
        {
            filter_sum += filter[y * filterWidth + x];
        }
	}

    for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
        {
            int max_value = 0;
            int min_value = 256; //assuming max value of 255 in image

            // multiply every value of the filter with corresponding image pixel
            for (int filterY = 0; filterY < filterHeight; filterY++)
            {
                for (int filterX = 0; filterX < filterWidth; filterX++)
                {
                    int imageX = min(max(x - filterWidth / 2 + filterX, 0), width - 1);
                    int imageY = min(max(y - filterHeight / 2 + filterY, 0), height - 1);
                    int filter_value = filter[filterY * filterWidth + filterX];
                    if (filter_value > 0)
                    {
                        max_value = max(max_value, (int)image[imageY * width + imageX]);
                        min_value = min(min_value, (int)image[imageY * width + imageX]);
                    }
                }
            }

            //morphologically apply operation
            if (type == 0) // dilation
            {
                result[y * width + x] = max_value;
            }
            else //erosion
            {
                result[y * width + x] = min_value % 256;
            }
        }
	}
    return 1;
}

__global__ void bgr2grayKernel(uint8_t* gray, const uint8_t* bgr, const int size)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    gray[idx] = saturate_cast(0.114f * bgr[idx * 3] + 0.587f * bgr[idx * 3 + 1] + 0.299f * bgr[idx * 3 + 2]);
}

__global__ void normalizeAndApplyLevelesKernel(uint8_t* diffTh, const int* sumDiff, const int* sumLuma, const int size, const float normFactor, const float maxLevel)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    float normed = ((float)sumDiff[idx] / (float)sumLuma[idx]) * normFactor;
    diffTh[idx] = static_cast<uint8_t>(fminf(normed, maxLevel));
}


uint64_t createMotionEstimator(const int width, const int height, const int outWidth, const int outHeight, const int morphFilterWidth, const int morphFilterHeight, const uint8_t maxLevel, const float normFactor, const int batchSize)
{
    MotionEstimator* motionEstimator = new MotionEstimator(width, height, outWidth, outHeight, morphFilterWidth, morphFilterHeight, maxLevel, normFactor, batchSize);
    return uint64_t(motionEstimator);
}

void MotionEstimator::calculateMotionEstimation(uint8_t* meBatch, const uint8_t* d_bgrBatch, const bool* skipFilter)
{
    int startIndex = 0;
    uint8_t* me = meBatch;
    const uint8_t* d_bgr = d_bgrBatch;

    if (isFirst)
    {
        bgr2grayKernel<<<gridSize1D, blockSize1D, 0, stream>>>(rawPtr(d_frame), d_bgr, inSize);
        d_bgr += inSize * 3;
        me += outSize;
        isFirst = false;
        startIndex = 1;
    }

    for(int i = startIndex; i < batchSize; ++i)
    {
        if (skipFilter[i]){
            thrust::swap(d_frame, d_prevFrame);
            bgr2grayKernel<<<gridSize1D, blockSize1D, 0, stream>>>(rawPtr(d_frame), d_bgr, inSize);
            CUDA_CHECK(cudaMemsetAsync(rawPtr(d_sumLuma), 0, outSize * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(rawPtr(d_sumDiff), 0, outSize * sizeof(int), stream));
            sumOfDiffsSharedKernel<<<girdSize, blockSize, 0, stream>>>(rawPtr(d_sumDiff), rawPtr(d_sumLuma), rawPtr(d_frame), rawPtr(d_prevFrame), width, height, decimationFactorWidth, decimationFactorHeight, outWidth, outHeight);
            normalizeAndApplyLevelesKernel<<<gridSize1DIcon, blockSize1DIcon, 0, stream>>>(rawPtr(d_diffTh), rawPtr(d_sumDiff), rawPtr(d_sumLuma), outSize, normFactor, maxLevel);
            CUDA_CHECK(cudaMemcpyAsync(diffTh[i].data(), rawPtr(d_diffTh), outSize * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
        }
        d_bgr += inSize * 3;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for(int i = startIndex; i < batchSize; ++i)
    {
        if (skipFilter[i]){
            applyMorphFilter(diffTh[i].data(), outHeight, outWidth, filter.data(), morphFilterWidth, morphFilterHeight, 0, dialted.data());
            applyMorphFilter(dialted.data(), outHeight, outWidth, filter.data(), morphFilterWidth, morphFilterHeight, 1, me);
            }
            me += outSize;
    }
}

void motionEstimation(uint64_t motionEstimatorAddress, uint8_t* me, const uint8_t* d_bgr, const bool* skipFilter)
{
    MotionEstimator* motionEstimator = reinterpret_cast<MotionEstimator*>(motionEstimatorAddress);
    motionEstimator->calculateMotionEstimation(me, d_bgr, skipFilter);
}
void destroyMotionEstimator(uint64_t motionEstimatorAddress)
{
	MotionEstimator* motionEstimator = reinterpret_cast<MotionEstimator*>(motionEstimatorAddress);
	delete motionEstimator;
}