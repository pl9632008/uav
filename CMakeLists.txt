cmake_minimum_required(VERSION 3.0)
project(test)
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/outlib)
set(SRC src/infer.cu
        src/yolo.cu
        src/rotate.cpp)

enable_language(CUDA)
find_package(OpenCV REQUIRED PATHS F:/opencv/install)
include_directories(${OpenCV_INCLUDE_DIRS} include include/yolo include/mavlink2.0 include/serial include/custom include/tensorrt include/cuda)
link_directories(${PROJECT_SOURCE_DIR}/lib)

add_library(inspect SHARED ${SRC})
target_link_libraries(inspect nvinfer cudart ${OpenCV_LIBS})

add_executable(sift src/main.cpp ${SRC})
target_link_libraries(sift nvinfer cudart ${OpenCV_LIBS})

