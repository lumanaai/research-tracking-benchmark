#  FastTracker — C++ Version

A lightweight, CPU-only multi-object tracker designed for complex traffic scenes, implemented entirely in C++.

Contact: [Hamidreza Hashempoor](https://hamidreza-hashempoor.github.io/)


---

##  Dependencies

### Required Libraries

| Library | Purpose | Install Command (Ubuntu/Debian) |
|----------|----------|--------------------------------|
| **OpenCV ≥ 4.0** | Image loading, drawing, and I/O | ```bash sudo apt install libopencv-dev ``` |
| **Eigen ≥ 3.3** | Linear algebra for Kalman Filter | *(Header-only, bundled or install manually)* |
| **CMake ≥ 3.0** | Build system | ```bash sudo apt install cmake ``` |
| **g++ ≥ 8.0** | Compiler with C++11/17 support | ```bash sudo apt install build-essential ``` |

If you don’t have Eigen globally installed, just include it in your project directory and point CMake to it:


```cmake
include_directories(${PROJECT_SOURCE_DIR}/eigen-3.4-rc1/)
```


### Create and configure the build directory and make
```bash
mkdir build && cd build
cmake ..
make
```
### Prepare simple dataset and run

Make an example dataset along with required libraries like:


```
FastTracker_CPP/
├── include/
│   ├── FastTracker.h
│   ├── STrack.h
│   ├── kalmanFilter.h
│   └── lapjv.h
├── src/
│   ├── FastTracker.cpp
│   ├── STrack.cpp
│   ├── kalmanFilter.cpp
│   ├── lapjv.cpp
│   └── utils.cpp
├── main.cpp
├── CMakeLists.txt
└── eigen-3.4-rc1/        
```

Then run the project (you can change dataset directory path inside `main.cpp`)

```bash
./Tracker_proj
```


