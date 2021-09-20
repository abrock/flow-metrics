cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(flow-eval CXX C)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals -std=c++20 -fPIC -Wall -pedantic -g -ggdb -fopenmp")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Vigra REQUIRED)
include_directories(${Vigra_INCLUDE_DIRS})

find_package(HDF5 REQUIRED COMPONENTS C CXX HL)
include_directories(${HDF5_INCLUDE_DIRS})

find_package(Boost COMPONENTS system filesystem iostreams REQUIRED)

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})

find_package(colorScales REQUIRED)
include_directories(${COLORSCALES_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(RunningStats REQUIRED)
include_directories(${RUNNINGSTATS_INCLUDE_DIRS})

find_package(colorScales REQUIRED)
include_directories(${COLORSCALES_INCLUDE_DIRS})

find_package(SimpleJson REQUIRED)

find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)

set(HDF5_LIBRARIES "${HDF5_LIBRARIES};${HDF5_C_LIBRARIES};${HDF5_CXX_LIBRARIES};${HDF5_HL_LIBRARIES};${HDF5_CXX_HL_LIBRARIES};${HDF5_C_HL_LIBRARIES}")

include_directories("/usr/manual/include")

SET(LIBRARIES vigraimpex)

set(TWOGEOM_LIBRARY 2geom)

include_directories("/usr/local/include/2geom-0.3/")
include_directories("/usr/local/include/2geom-0.3/2geom/")
include_directories(/usr/local/include/2geom-1.0.0/2geom/)
include_directories(/usr/local/include/2geom-1.0.0/)

#add_subdirectory(libframename)
add_subdirectory(kitti-devkit)
#add_subdirectory(libColorSelect/)
#add_subdirectory(libicp/)

add_library(flowmetrics
    basename.cpp
    contourfeatures.cpp
    contourflow.cpp
    flowgt.cpp
    flowmetrics.cpp
    metric-helpers.cpp
    readgt.cpp
    )

add_executable(flow-metrics flow-metrics.cpp)

foreach(target flowmetrics flow-metrics)
target_link_libraries(${target}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_IOSTREAMS_LIBRARY}
    ${CERES_LIBRARIES}
    ${COLORSCALES_LIBRARIES}
    ${GFLAGS_LIBRARY}
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${GSL_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_C_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
    ${HDF5_C_HL_LIBRARIES}
    ${HDF5_CXX_HL_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${OpenCV_LIBS}
    ${OpenCV_EXTRA_LIBS}
    ${RUNNINGSTATS_LIBRARIES}
    ${SIMPLEJSON_LIBRARIES}
    ${TWOGEOM_LIBRARY}
    ${Vigra_LIBRARIES}
    kitti_io_flow
    ParallelTime
    )
endforeach()

target_link_libraries(flow-metrics flowmetrics)
