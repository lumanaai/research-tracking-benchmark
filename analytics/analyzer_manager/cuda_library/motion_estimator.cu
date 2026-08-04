#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "motion_estimator.h"

__global__ void resizeBilinearKernel(uint8_t* output, const uint8_t* input, const int inHeight, const int inWidth, const int outHeight, const int outWidth, const float sY, const float sX)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= outWidth || y >= outHeight)
	{
		return;
	}
	float yf = fmaxf(y * sY, sY);
	float xf = fmaxf(x * sX, sX);
	int x0 = fminf(xf, inWidth - 2);
	int y0 = fminf(yf, inHeight - 2);
	float deltaY = yf - y0;
	float deltaX = xf - x0;

	const float a = (1 - deltaY) * (1 - deltaX);
	const float b = (deltaY) * (1 - deltaX);
	const float c = (1 - deltaY) * (deltaX);
	const float d = (deltaY) * (deltaX);

	const int idxOut = y * outWidth + x;
	const int idxA = y0 * inWidth + x0;
	const int idxB = (y0 + 1) * inWidth + x0;
	const int idxC = y0 * inWidth + x0 + 1;
	const int idxD = (y0 + 1) * inWidth + x0 + 1;

    output[idxOut] = saturate_cast(input[idxA] * a + input[idxB] * b + input[idxC] * c + input[idxD] * d);
 }

__global__ void subtractAndResizeKernel(uint8_t* resizedDiff, 
	const uint8_t* frame, const uint8_t* prevFrame, 
	const int inHeight, const int inWidth, 
	const int outHeight, const int outWidth, 
	const float sY, const float sX)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= outWidth || y >= outHeight)
	{
		return;
	}
	float yf = fmaxf(y * sY, sY);
	float xf = fmaxf(x * sX, sX);
	int x0 = fminf(xf, inWidth - 2);
	int y0 = fminf(yf, inHeight - 2);
	float deltaY = yf - y0;
	float deltaX = xf - x0;

	const float a = (1 - deltaY) * (1 - deltaX);
	const float b = (deltaY) * (1 - deltaX);
	const float c = (1 - deltaY) * (deltaX);
	const float d = (deltaY) * (deltaX);

	const int idxOut = y * outWidth + x;
	const int idxA = y0 * inWidth + x0;
	const int idxB = (y0 + 1) * inWidth + x0;
	const int idxC = y0 * inWidth + x0 + 1;
	const int idxD = (y0 + 1) * inWidth + x0 + 1;

    resizedDiff[idxOut] = saturate_cast(max(0, frame[idxA] - prevFrame[idxA]) * a + 
                                        max(0, frame[idxB] - prevFrame[idxB]) * b + 
                                        max(0, frame[idxC] - prevFrame[idxC]) * c + 
                                        max(0, frame[idxD] - prevFrame[idxD]) * d);
}

uint64_t createMotionEstimator(uint8_t* firstFrame, const int l1Width, const int l1Height, const int width, const int height, const int outWidth, const int outHeight)
{
	MotionEstimator* motionEstimator = new MotionEstimator(firstFrame, width, height, l1Width, l1Height, outWidth, outHeight);
	return uint64_t(motionEstimator);
}

void MotionEstimator::calculateMotionEstimation(uint8_t* me, uint8_t* imageResized, const uint8_t* frame)
{
	// save previous frame and copy current frame
    thrust::swap(d_frame, d_prevFrame);
    CUDA_CHECK(cudaMemcpyAsync(rawPtr(d_frame), frame, inSize * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    // subtract and resize
    subtractAndResizeKernel<<<gridSizeResizeL1, blockSize, 0, stream>>>(rawPtr(d_resizedDiff), rawPtr(d_frame), rawPtr(d_prevFrame), height, width, l1Height, l1Width, yScaleL1, xScaleL1);
    // resize to motion estimation dims (icon dims)
    resizeBilinearKernel<<<gridSizeResize, blockSize, 0, stream>>>(rawPtr(d_me), rawPtr(d_resizedDiff), l1Height, l1Width, outHeight, outWidth, yScale, xScale);
	// copy output to host
    CUDA_CHECK(cudaMemcpyAsync(me, rawPtr(d_me), outSize * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
	// resize image for normalization
	resizeBilinearKernel<<<gridSizeResize, blockSize, 0, stream>>>(rawPtr(d_me), rawPtr(d_frame), height, width, outHeight, outWidth, (float)height / outHeight, (float)width / outWidth);
    CUDA_CHECK(cudaMemcpyAsync(imageResized, rawPtr(d_me), outSize * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

void motionEstimationCuda(uint64_t motionEstimatorAddress, uint8_t* me, uint8_t* imageResized, const uint8_t* frame)
{
    MotionEstimator* motionEstimator = reinterpret_cast<MotionEstimator*>(motionEstimatorAddress);
    motionEstimator->calculateMotionEstimation(me, imageResized, frame);
}

void synchronizeCudaStream(uint64_t motionEstimatorAddress)
{
	MotionEstimator* motionEstimator = reinterpret_cast<MotionEstimator*>(motionEstimatorAddress);
	CUDA_CHECK(cudaStreamSynchronize(motionEstimator->stream));
}

void destroyMotionEstimator(uint64_t motionEstimatorAddress)
{
	MotionEstimator* motionEstimator = reinterpret_cast<MotionEstimator*>(motionEstimatorAddress);
	delete motionEstimator;
}