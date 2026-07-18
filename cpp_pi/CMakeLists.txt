cmake_minimum_required(VERSION 3.20)
project(driveway_pi_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 5.0 REQUIRED COMPONENTS core imgproc videoio dnn)

add_executable(driveway_benchmark benchmark.cpp)
target_link_libraries(driveway_benchmark PRIVATE ${OpenCV_LIBS})

