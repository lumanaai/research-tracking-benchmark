#include "opencv2/opencv.hpp"
#include "TransmitterManager.h"

void transmitFrames(const char* videoPath, const int rate, const uint64_t timestamp, const int numFrames)
{
    const std::string path(videoPath);
    std::cout << "video path: " << path << std::endl;
    const std::string streamSrc = "filesrc location= " + path;
    std::string pipeline = streamSrc + " ! qtdemux";
    pipeline += " ! queue ! decodebin ! videorate max-rate=" + std::to_string(rate);
    pipeline += " ! nvvidconv ! video/x-raw,format=BGRx";
    pipeline += " ! queue ! videoconvert";
    pipeline += " ! queue ! video/x-raw, format=BGR ! appsink";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    const int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const int bufferSize = width * height * 3;
    const uint64_t delay = (1.0 / rate) * 1000;

    for(int i = 0; i < numFrames; ++i)
    {
        try
        {
            const uint64_t i_timestamp = timestamp + delay * i;
            // open shared memory segment
            const std::string shmName = "img-" + std::to_string(i_timestamp) + ".bgr";
            TransmitterManager<uint8_t> transmitter(shmName.c_str(), bufferSize);
            uint8_t* ptr = transmitter.getPointer();

            // read frames into shared memory
            cv::Mat frame(height, width, CV_8UC3, (void*)ptr);
            bool ret = cap.read(frame);
            if(!ret)
            {
                break;
            }
        }
        catch(const std::runtime_error& e)
        {
            break;
        }
    }
}

void testTransmitter()
{
    const int framerate = 30;
    // streaming source, can be replaced with camera source
    const std::string videoPath = "/home/nvidia/analytics/data/garage.mp4";
    const std::string streamSrc = "filesrc location= " + videoPath;
    std::string pipeline = streamSrc + " ! qtdemux";
    pipeline += " ! queue ! decodebin ! videorate max-rate=" + std::to_string(framerate);
    pipeline += " ! nvvidconv ! video/x-raw,format=BGRx";
    pipeline += " ! queue ! videoconvert";
    pipeline += " ! queue ! video/x-raw, format=BGR ! appsink";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    const int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const int channels = 3;
    // for testing we write only 200 frames
    const int numFrames = 200;

    const int bufferSize = width * height * channels;

    std::cout << "Streaming source: " << videoPath << std::endl;
    std::cout << "Video resolution: " << width << "x" << height << std::endl;

    for(int i = 0; i < numFrames; ++i)
    {
        // open shared memory segment
        const std::string shmName = "frame" + std::to_string(i) + ".buffer";
        TransmitterManager<uint8_t> transmitter(shmName.c_str(), bufferSize);
        uint8_t* ptr = transmitter.getPointer();

        // create cv Mat from existing buffer
        cv::Mat frame(height, width, CV_8UC3, (void*)ptr);
        bool ret = cap.read(frame);
        if(!ret)
        {
            break;
        }
    }
}