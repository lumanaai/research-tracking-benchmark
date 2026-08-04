#include "parse_images_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/execution_policy.h>

__constant__ __device__ float d_gaussianFilter[100];

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

#pragma unroll
	for (int z = 0; z < 3; ++z)
	{
		output[idxOut * 3 + z] = saturate_cast(input[idxA * 3 + z] * a + input[idxB * 3 + z] * b + input[idxC * 3 + z] * c + input[idxD * 3 + z] * d);
	}
}

__global__ void gaussianColsKernel(uint8_t* output, const uint8_t* input, const int kernelSize, const int height, const int width)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
	{
		return;
	}
	const int radius = (kernelSize - 1) / 2;
	const float* mask = &d_gaussianFilter[radius];
	float sumB = 0.0f;
	float sumG = 0.0f;
	float sumR = 0.0f;
	if (x < radius || x > width - radius - 1 || y < radius || y > height - radius - 1)
	{
#pragma unroll
		for (int maskX = -radius; maskX <= radius; ++maskX)
		{
            const int imageX = fminf(fmaxf(x + maskX, 0), width - 1); // BORDER_REPLICATE 
			const int idx = y * width + imageX;
			sumB += input[idx * 3 + 0] * mask[maskX];
			sumG += input[idx * 3 + 1] * mask[maskX];
			sumR += input[idx * 3 + 2] * mask[maskX];
		}
	}
	else
	{
#pragma unroll
		for (int maskX = -radius; maskX <= radius; ++maskX)
		{
			const int idx = y * width + (x + maskX);
			sumB += input[idx * 3 + 0] * mask[maskX];
			sumG += input[idx * 3 + 1] * mask[maskX];
			sumR += input[idx * 3 + 2] * mask[maskX];
		}
	}
	const int idx = y * width + x;
	output[idx * 3 + 0] = saturate_cast(sumB);
	output[idx * 3 + 1] = saturate_cast(sumG);
	output[idx * 3 + 2] = saturate_cast(sumR);
}

__global__ void gaussianRowsKernel(uint8_t* output, const uint8_t* input, const int kernelSize, const int height, const int width)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
	{
		return;
	}
	const int radius = (kernelSize - 1) / 2;
	const float* mask = &d_gaussianFilter[radius];
	float sumB = 0.0f;
	float sumG = 0.0f;
	float sumR = 0.0f;
	if (x < radius || x > width - radius - 1 || y < radius || y > height - radius - 1)
	{
#pragma unroll
		for (int maskY = -radius; maskY <= radius; ++maskY)
		{
			int imageY = fminf(fmaxf(y + maskY, 0), height - 1); // BORDER_REPLICATE 
			const int idx = imageY * width + x;
			sumB += input[idx * 3 + 0] * mask[maskY];
			sumG += input[idx * 3 + 1] * mask[maskY];
			sumR += input[idx * 3 + 2] * mask[maskY];
		}
	}
	else
	{
#pragma unroll
		for (int maskY = -radius; maskY <= radius; ++maskY)
		{
			const int idx = (y + maskY) * width + x;
			sumB += input[idx * 3 + 0] * mask[maskY];
			sumG += input[idx * 3 + 1] * mask[maskY];
			sumR += input[idx * 3 + 2] * mask[maskY];
		}
	}
	const int idx = y * width + x;
	output[idx * 3 + 0] = saturate_cast(sumB);
	output[idx * 3 + 1] = saturate_cast(sumG);
	output[idx * 3 + 2] = saturate_cast(sumR);
}

template <class T> __global__  void normalizeImageKernel(T* normalized, const uint8_t* img, const int size)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= size)
	{
		return;
	}
	// BGR image to RGB normalized image
	normalized[x] = img[x * 3 + 2] * 0.003921568627451f; // R / 255.0
	normalized[x + size] = img[x * 3 + 1] * 0.003921568627451f; // G / 255.0
	normalized[x + 2 * size] = img[x * 3] * 0.003921568627451f; // B / 255.0
}

template <class T> __global__ void interleaved2planarKernel(T* planar, const uint8_t* interleaved, const int size)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= size)
	{
		return;
	}
	// BGR interleaved to RGB planar 
	planar[x] = interleaved[x * 3 + 2]; // R 
	planar[x + size] = interleaved[x * 3 + 1]; // G
	planar[x + 2 * size] = interleaved[x * 3]; // B
}

uint64_t createCudaParser(const float* gaussianFilter, const bool antialias, const bool normalize, const int kernelSize, const int batchSize, const int inHeight, const int inWidth, const int outHeight, const int outWidth, const bool isHalf)
{
	CudaImageParser* cudaParser = new CudaImageParser(kernelSize, antialias, normalize, batchSize, inHeight, inWidth, outHeight, outWidth, isHalf);
    if(antialias)
	{
		CUDA_CHECK(cudaMemcpyToSymbol(d_gaussianFilter, gaussianFilter, kernelSize * sizeof(float), 0, cudaMemcpyHostToDevice));   
	}
	return uint64_t(cudaParser);
}

void destroyCudaParser(uint64_t cudaParserAddress)
{
	CudaImageParser* cudaParser = reinterpret_cast<CudaImageParser*>(cudaParserAddress);
	delete cudaParser;
}

template<class T> void CudaImageParser::parseImages(T* d_outputs, uint8_t* outputs, uint8_t* d_bgrBatch)
{
	for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
	{
		uint8_t* d_bgr = d_bgrBatch + batchIndex * inSize * 3;
		uint8_t* output = outputs + batchIndex * outSize * 3;
		T* d_output = d_outputs + batchIndex * outSize * 3;
		
		if(antialias)
		{
			gaussianColsKernel<<<gridSize2D, blockSize2D, 0, stream>>>(rawPtr(d_buff), d_bgr, kernelSize, inHeight, inWidth);
			gaussianRowsKernel<<<gridSize2D, blockSize2D, 0, stream>>>(d_bgr, rawPtr(d_buff), kernelSize, inHeight, inWidth);
		}
		
		resizeBilinearKernel<<<gridSize2DResize, blockSize2D, 0, stream>>>(rawPtr(d_buff), d_bgr, inHeight, inWidth, outHeight, outWidth, yScale, xScale);
		
		if(normalize)
		{
			normalizeImageKernel<<<gridSizeSmall, blockSize, 0, stream>>>(d_output, rawPtr(d_buff), outSize);
		}
		else
		{
			interleaved2planarKernel<<<gridSizeSmall, blockSize, 0, stream>>>(d_output, rawPtr(d_buff), outSize);
		}
		CUDA_CHECK(cudaMemcpyAsync(output, rawPtr(d_buff), outSize * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));  
	}
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

void parseImages(uint64_t cudaParserAddress, void* d_outputs, uint8_t* outputs, uint8_t* d_bgrBatch)
{
	CudaImageParser* cudaParser = reinterpret_cast<CudaImageParser*>(cudaParserAddress);
	if (cudaParser->isHalf){
	    cudaParser->parseImages((half*) d_outputs, outputs, d_bgrBatch);
	}
	else{
	  	cudaParser->parseImages((float*) d_outputs, outputs, d_bgrBatch);
	}

}