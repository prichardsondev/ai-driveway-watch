cmake_minimum_required(VERSION 3.16)
project(ai_driveway_watch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

foreach(required_asset yoloV8.cpp yoloV8.h yolov8n.bin yolov8n.param)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${required_asset}")
        message(FATAL_ERROR
            "Missing ${required_asset}. Run ./fetch_qengineering_assets.sh first.")
    endif()
endforeach()

find_package(OpenCV REQUIRED COMPONENTS core geometry imgcodecs imgproc videoio)
find_package(ncnn REQUIRED)
find_package(Threads REQUIRED)

add_executable(driveway_ncnn_benchmark
    benchmark.cpp
    yoloV8.cpp
)

target_include_directories(driveway_ncnn_benchmark PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(driveway_ncnn_benchmark PRIVATE ${OpenCV_LIBS} ncnn)
target_compile_options(driveway_ncnn_benchmark PRIVATE -O3)

add_executable(driveway_ncnn_server
    server.cpp
    yoloV8.cpp
)

target_include_directories(driveway_ncnn_server PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(driveway_ncnn_server PRIVATE ${OpenCV_LIBS} ncnn Threads::Threads)
target_compile_options(driveway_ncnn_server PRIVATE -O3)
