import cv2
import time
import ctypes
import numpy as np
import argparse
from ctypes import c_char_p, c_uint64, c_int

cuda_lib = ctypes.CDLL("transmitter_tester.so")
cuda_lib.transmitFrames.argstype =(c_char_p, c_int, c_uint64, c_int)

def transmit_frames(stream_src, rate, timestamp, num_frames):
    cuda_lib.transmitFrames(ctypes.cast(stream_src, c_char_p), c_int(rate), c_uint64(timestamp), c_int(num_frames))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Transmit video frames over shared memory')
    parser.add_argument('--src', type =str, default="/home/nvidia/analytics/data/garage.mp4", help='video source path')
    parser.add_argument('--rate', type = int, default=30, help='streaming framerate')
    parser.add_argument('--frames', type = int, default=200, help='number of frames to transmit')
    args = parser.parse_args()

    timestamp = np.uint64(round(time.time() * 1000))
    print(f"timestamp = {timestamp}")
    transmit_frames(args.src.encode('utf-8'), args.rate, timestamp, args.frames)