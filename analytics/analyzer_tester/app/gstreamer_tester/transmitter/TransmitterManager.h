#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <iostream>
#include <string>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else 
#define EXPORT extern "C"
#endif

EXPORT void freeSharedMemorySegment(const char* name)
{
    if(shm_unlink(name))
    {
        std::cout << "shm_unlink failed: " << strerror(errno) << std::endl;
        throw std::runtime_error(strerror(errno));
    }
}

template<typename T>
class TransmitterManager
{
public:

    TransmitterManager(const char* name, const int size)
    {
        m_size = size * sizeof(T);
        m_shmFD = shm_open(name, O_CREAT | O_RDWR, 0666);
        ftruncate(m_shmFD, m_size);

        // map pointer to shared memory
        m_ptr = (T*)mmap(0, m_size, PROT_WRITE, MAP_SHARED, m_shmFD, 0);
        if (m_ptr == MAP_FAILED) 
        {
            std::cout << "mmap failed: " << strerror(errno) << std::endl;
            throw std::runtime_error(strerror(errno));
        }
    }
    
    ~TransmitterManager()
    {
        // unmap pointer to shared memory
        if (munmap(m_ptr, m_size) == -1) 
        {
            std::cout << "unmap failed: " << strerror(errno) << std::endl;
            throw std::runtime_error(strerror(errno));
        }

        // close shared memory segment
        if (close(m_shmFD)) 
        {
            std::cout << "close failed: " << strerror(errno) << std::endl;
            throw std::runtime_error(strerror(errno));
        }
    }

    T* getPointer()
    {
        return m_ptr;
    }

private:
    int m_size{}; // shared memory size in bytes
    int m_shmFD{}; // shared memory file descriptor
    T* m_ptr{}; // pointer to shared memory segment
};


EXPORT void transmitFrames(const char* videoPath, const int rate, const uint64_t timestamp, const int numFrames);
EXPORT void testTransmitter(const int framerate);