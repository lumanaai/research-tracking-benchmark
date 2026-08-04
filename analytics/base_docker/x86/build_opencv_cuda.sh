#!/bin/bash
set -e

# Variables
OPENCV_VERSION=4.12.0
NUM_CORES=$(nproc)

# Install dependencies
apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libcanberra-gtk* \
    libatlas-base-dev gfortran \
    python3-dev python3-numpy \
    wget unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Get OpenCV source
cd /tmp
git clone --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git
git clone --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git

# Build
cd opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN="8.0 8.6 8.9 9.0 12.0" \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_DOCS=OFF \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_cudacodec=OFF \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])") \
      -D PYTHON3_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.so \
      ..

make -j${NUM_CORES}
make install
ldconfig

# Cleanup
rm -rf /tmp/opencv /tmp/opencv_contrib
