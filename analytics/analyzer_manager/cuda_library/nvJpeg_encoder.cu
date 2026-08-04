#include <NvJpegEncoder.h>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <Python.h>
#include <sstream>
#include "nvJpeg_encoder.h"

__constant__ float matYuv2Rgb[3][3];
__constant__ float matRgb2Yuv[3][3];

void setMatRgb2Yuv(cudaStream_t stream) 
{
    const int black = 16;
    const int white = 235;
    const int max = 255;
    const float wr = 0.2126f; 
    const float wb = 0.0722f;
    float mat[3][3] = { wr,                         1.0f - wb - wr,                           wb,
                        -0.5f * wr / (1.0f - wb),  -0.5f * (1 - wb - wr) / (1.0f - wb),       0.5f,
                        0.5f,                      -0.5f * (1.0f - wb - wr) / (1.0f - wr),   -0.5f * wb / (1.0f - wr)};

    for (int i = 0; i < 3; i++) 
    {
        for (int j = 0; j < 3; j++) 
        {
            mat[i][j] = (float)(1.0 * (white - black) / max * mat[i][j]);
        }
    }
    cudaMemcpyToSymbolAsync(matRgb2Yuv, mat, sizeof(mat), 0, cudaMemcpyHostToDevice, stream);
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit rgbToY(RgbUnit r, RgbUnit g, RgbUnit b) 
{
    const YuvUnit low = 1 << (sizeof(YuvUnit) * 8 - 4);
    return matRgb2Yuv[0][0] * r + matRgb2Yuv[0][1] * g + matRgb2Yuv[0][2] * b + low;
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit rgbToU(RgbUnit r, RgbUnit g, RgbUnit b) 
{
    const YuvUnit mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    return matRgb2Yuv[1][0] * r + matRgb2Yuv[1][1] * g + matRgb2Yuv[1][2] * b + mid;
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit rgbToV(RgbUnit r, RgbUnit g, RgbUnit b) 
{
    const YuvUnit mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    return matRgb2Yuv[2][0] * r + matRgb2Yuv[2][1] * g + matRgb2Yuv[2][2] * b + mid;
}

template<class YuvUnitx2>
__global__ static void bgrToYuvKernel(uint8_t *pRgb, uint8_t *pYuv, int nYuvPitch, int nWidth, int nHeight) 
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) 
    {
        return;
    }

    uint8_t *pSrc = pRgb + x * 3 + y * nWidth * 3;
    
    uint8_t *int2a = pSrc;
    uint8_t *int2b = pSrc + nWidth * 3;

    uint8_t b = (int2a[0] + int2a[3] + int2b[0] + int2b[3]) / 4,
        g = (int2a[1] + int2a[4] + int2b[1] + int2b[4]) / 4,
        r = (int2a[2] + int2a[5] + int2b[2] + int2b[5]) / 4;

    uint8_t *pDst = pYuv + x + y * nWidth;
    
    pDst[0] = rgbToY<uint8_t, uint8_t>(int2a[0+2], int2a[0+1], int2a[0+0]);
    pDst[1] = rgbToY<uint8_t, uint8_t>(int2a[1*3+2], int2a[1*3+1], int2a[1*3+0]);
    pDst[nWidth] = rgbToY<uint8_t, uint8_t>(int2b[0+2], int2b[0+1], int2b[0+0]);
    pDst[nWidth + 1] = rgbToY<uint8_t, uint8_t>(int2b[1*3+2], int2b[1*3+1], int2b[1*3+0]);
    *(pYuv + nWidth * nHeight + (size_t)(nWidth/2)*((size_t)(y/2)) + x/2) = rgbToU<uint8_t, uint8_t>(r, g, b);
    *(pYuv + nWidth * nHeight + (size_t)(nWidth/2) * (size_t)(nHeight/2) + (size_t)(nWidth/2)*((size_t)(y/2)) + x/2) = rgbToV<uint8_t, uint8_t>(r, g, b);
}

void BGRToYUV420(uint8_t *dpBgra, uint8_t *dpYUV420, int nWidth, int nHeight, cudaStream_t stream) 
{
    setMatRgb2Yuv(stream);
    bgrToYuvKernel<ushort2><<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>(dpBgra, dpYUV420, 4*nWidth, nWidth, nHeight);
}

unsigned long NVJPGEncoder::encode(std::vector<unsigned char>& jpegBytes, const uint8_t* img, const int width, const int height, const int quality)
{
    // allocate device memory
    size_t bgrFrameSize = width * height * 3;
    size_t yuvFrameSize = width * height * 1.5;
    thrust::device_vector<uint8_t> d_yuvFrame(yuvFrameSize);
    thrust::device_vector<uint8_t> d_bgrFrame(bgrFrameSize);
    
    // copy BGR frame to device
    CUDA_CHECK(cudaMemcpyAsync(rawPtr(d_bgrFrame), img, bgrFrameSize, cudaMemcpyHostToDevice, stream));
    
    // convert BGR to YUV
    BGRToYUV420(rawPtr(d_bgrFrame), rawPtr(d_yuvFrame), width, height, stream);

    // copy YUV to host
    std::vector<char> yuvFrame(yuvFrameSize);
    CUDA_CHECK(cudaMemcpyAsync(yuvFrame.data(), rawPtr(d_yuvFrame), yuvFrameSize, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // copy data to NvBuffer
    char* imgData = yuvFrame.data();
    NvBuffer buffer(V4L2_PIX_FMT_YUV420M, width, height, 0);
    buffer.allocateMemory();   
    for (uint32_t i = 0; i < buffer.n_planes; ++i)
    {
        NvBuffer::NvBufferPlane& plane = buffer.planes[i];
        char* data = (char*) plane.data;
        plane.bytesused = plane.fmt.stride * plane.fmt.height;
        memcpy(data, imgData, plane.bytesused);
        imgData += plane.bytesused;
    }

    // encode and return buffer size in bytes
    unsigned long outBuffSize = width * height * 3 / 2;
    jpegBytes.resize(outBuffSize);
    unsigned char* ptr = jpegBytes.data();
    const int status = encoder->encodeFromBuffer(buffer, JCS_YCbCr, &ptr, outBuffSize, quality);
    buffer.deallocateMemory();
    if (status != 0)
    {
        LOG_ERROR(status, "NvJpeg Encoder Error");
    }
    return outBuffSize;
}

uint64_t createEncoder()
{
	NVJPGEncoder* encoder = new NVJPGEncoder();
	return uint64_t(encoder);
}

PyObject* nvJpegEncode(uint64_t encoderAddress, const uint8_t* image, const int width, const int height, const int quality)
{
    NVJPGEncoder* encoder = reinterpret_cast<NVJPGEncoder*>(encoderAddress);
    std::vector<unsigned char> jpegBytes;
    unsigned long outSize = encoder->encode(jpegBytes, image, width, height, quality);
    PyObject* obj = PyBytes_FromStringAndSize((const char*)jpegBytes.data(), outSize);
    return obj; 
}

void destroyEncoder(uint64_t encoderAddress)
{
	NVJPGEncoder* encoder = reinterpret_cast<NVJPGEncoder*>(encoderAddress);
	delete encoder;
}