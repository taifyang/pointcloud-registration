project(pointcloud_registration)
cmake_minimum_required(VERSION 3.16)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-w")

find_package(OpenMP)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Sophus REQUIRED)

include_directories(include ${Eigen3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS})
file(GLOB_RECURSE SOURCES "source/*.cpp")
add_executable(main main.cpp ${SOURCES})
target_link_libraries(main ${PCL_LIBRARIES} Sophus::Sophus tbb)

