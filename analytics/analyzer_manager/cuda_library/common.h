#pragma once
#include <thrust/device_vector.h>
#include <iostream>
#include <sstream>
#include <chrono>

#define EXPORT extern "C"
#define LOG_ERROR(err, msg) logError((err), msg,  __FUNCTION__, __LINE__);
#define CUDA_CHECK(status) cudaCheck((status), __FILE__, __LINE__);

void logError(const int err, const char* msg, const char* function, const int line)
{
    std::stringstream ss;
    ss << function << ":" << line << ": " << msg << "(" << err << ")";
    std::cout << ss.str() << std::endl;
    throw std::runtime_error(ss.str());
}

void cudaCheck(const cudaError_t& status, const char* filename, const int line)
{
	if(status != cudaSuccess)
	{
		std::stringstream msg;
		msg << "CUDA error in " << filename << ":" << line << ": " << cudaGetErrorString(status) << " (" << status << ")";
		std::cout << msg.str() << std::endl;
		throw std::runtime_error(msg.str());
	}
}

template<typename T>
inline T* rawPtr(thrust::device_vector<T>& d_vec)
{
	return reinterpret_cast<T*>(thrust::raw_pointer_cast(d_vec.data()));
}

__device__ uint8_t saturate_cast(const float& val)
{
	return static_cast<uint8_t>(fmaxf(0.0f, fminf(val, 255.0f)));
}

inline void getGirdSize2D(dim3& gridSize2D, const dim3& blockSize2D, const int x, const int y, const int z)
{
	gridSize2D = { (x + blockSize2D.x - 1) / blockSize2D.x, (y + blockSize2D.y - 1) / blockSize2D.y, (z + blockSize2D.z - 1) / blockSize2D.z};
}

class Timer
{
public:
	void begin()
	{
		m_start = std::chrono::high_resolution_clock::now();
	}
	double end()
	{
		m_end = std::chrono::high_resolution_clock::now();
		m_runtime = std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start).count() / 1000.0;
		return m_runtime;
	}

private:
	double m_runtime{};
	std::chrono::time_point<std::chrono::high_resolution_clock> m_start{};
	std::chrono::time_point<std::chrono::high_resolution_clock> m_end{};
};