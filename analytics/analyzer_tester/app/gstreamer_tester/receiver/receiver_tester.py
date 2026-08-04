import ctypes
import argparse
import numpy as np
from ctypes import c_uint64, c_int

from pkg_resources import require
cuda_lib = ctypes.CDLL("receiver_tester.so")
cuda_lib.receiveFrames.argstype =(c_int, c_int, c_int, c_uint64, c_int)

def receive_frames(width, height, rate, timestamp, num_frames):
    cuda_lib.receiveFrames(c_int(width), c_int(height), c_int(rate), c_uint64(timestamp), c_int(num_frames))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Receive video frames from shared memory')
    parser.add_argument('--rate', type = int, default=30, help='streaming framerate')
    parser.add_argument('--timestamp', type = np.uint64, help='initial timestamp (get from transmitter)', required=True)
    parser.add_argument('--frames', type = int, default=200, help='number of frames to transmit')
    parser.add_argument('--width', type = int, default=2592, help='frame width')
    parser.add_argument('--height', type = int, default=1520, help='frame height')
    args = parser.parse_args()

    receive_frames(args.width, args.height, args.rate, np.uint64(args.timestamp), args.frames)