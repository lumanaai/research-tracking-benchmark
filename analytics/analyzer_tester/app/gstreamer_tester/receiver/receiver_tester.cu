#include "opencv2/opencv.hpp"
#include "ReceiverManager.h"

void receiveFrames(const int width, const int height, const int rate, const uint64_t timestamp, const int numFrames)
{
    const int bufferSize = width * height * 3;
    const uint64_t delay = (1.0 / rate) * 1000;
    for(int i = 0; i < numFrames; ++i)
    {
        try
        {
            // open shared memory segment
            const uint64_t i_timestamp = timestamp + delay * i;
            const std::string shmName = "img-" + std::to_string(i_timestamp) + ".bgr";
            ReceiverManager<uint8_t> receiver(shmName.c_str(), bufferSize);
            uint8_t* ptr = receiver.getPointer();
    
            // create cv Mat from existing buffer
            cv::Mat frame(height, width, CV_8UC3, (void*)ptr);
            cv::namedWindow("test", cv::WINDOW_NORMAL);
            cv::imshow("test", frame);
            cv::waitKey(30);

            // release shared memory 
            freeSharedMemorySegment(shmName.c_str());
        }
        catch(const std::runtime_error& e)
        {
            break;
        }
    }
}

void testReceiver(const int width, const int height)
{
    // shared memory buffer size
    const int channels = 3;
    const int bufferSize = width * height * channels;
    // for testing, limit number of frames received
    const int numFrames = 200;

    for(int i = 0; i < numFrames; ++i)
    {
        // open shared memory segment
        const std::string shmName = "frame" + std::to_string(i) + ".buffer";
        ReceiverManager<uint8_t> receiver(shmName.c_str(), bufferSize);
        uint8_t* ptr = receiver.getPointer();
 
        // create cv Mat from existing buffer
        cv::Mat frame(height, width, CV_8UC3, (void*)ptr);
        cv::namedWindow("test", cv::WINDOW_NORMAL);
        cv::imshow("test", frame);
        cv::waitKey(30);

        // release shared memory 
        freeSharedMemorySegment(shmName.c_str());
    }
}
