cmake_minimum_required(VERSION 3.0)

project(tracker_proj)
# add_definitions(-std=c++11)

#### Compiler Options

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -fPIC  ")
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
link_libraries(stdc++fs)


### APP Includes
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/eigen-3.4-rc1/)

#----------------------------------------------------------------
#External Libraries----------------------------------------------
#----------------------------------------------------------------

### OpenCV
find_package(OpenCV REQUIRED)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


#----------------------------------------------------------------
#Build Libraries-------------------------------------------------
#----------------------------------------------------------------

### Fast
file(GLOB FastTracker_SRC "src/*.cpp")
add_library(FastTracker SHARED ${FastTracker_SRC})

target_link_libraries(FastTracker)

#----------------------------------------------------------------
#Main -----------------------------------------------------------
#----------------------------------------------------------------

### Main APP
add_executable(Tracker_proj ${PROJECT_SOURCE_DIR}/main.cpp)

set(Tracker_proj_LIBS 
	FastTracker
	${OpenCV_LIBS}
	X11
)

target_link_libraries(Tracker_proj 
	${Tracker_proj_LIBS}
)



add_definitions(-O1 -pthread)



