

find_package(Boost COMPONENTS system filesystem REQUIRED)

INCLUDE_DIRECTORIES("..")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

FIND_PACKAGE(HDF5 REQUIRED)
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})

find_package(PNG)
include_directories(${PNG_INCLUDE_DIRS})

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_Include_Dirs})

FIND_PACKAGE(HDF5 REQUIRED)
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})

FIND_PACKAGE(Vigra REQUIRED)
INCLUDE_DIRECTORIES(${Vigra_INCLUDE_DIRS})

set(KITTI_TARGETS
    convert-flow
    create-invalid-flow
    )

set(KITTI_LIBS
    mail
    io_flow
    io_disp
    log_colormap
    utils
    )

FOREACH(source ${KITTI_LIBS})
add_library(kitti_${source} ${source}.cpp)
target_link_libraries(kitti_${source}
    ${OpenCV_LIBS}
    ${OpenCV_EXTRA_LIBS}
    ${PNG_LIBRARIES}
    ${GSL_LIBRARIES}
    )
ENDFOREACH()

target_link_libraries(kitti_io_flow kitti_log_colormap)


#add_library(kitti_io_disp io_disp.cpp)

find_package(RunningStats REQUIRED)
include_directories(${RUNNINGSTATS_INCLUDE_DIRS})

FOREACH(source ${KITTI_TARGETS})
ADD_EXECUTABLE(kitti-${source} ${source}.cpp)
target_link_libraries(kitti-${source}
    ${PNG_LIBRARIES}
    kitti_mail
    kitti_io_flow
    kitti_log_colormap
    kitti_utils
    kitti_io_disp
    ${RUNNINGSTATS_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    )

FOREACH(libsource ${KITTI_LIBS})
target_link_libraries(kitti-${source} kitti_${libsource})
ENDFOREACH() # link all kitti_libs to current kitti source

ENDFOREACH() # add all kitti_executable

FOREACH(source ${CARDBOARD_TARGETS})
target_link_libraries(kitti-${source}
    flowmetrics
    ${Vigra_LIBRARIES}
    ${HDF5_C_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
    ${HDF5_LIBRARIES}
    vigraimpex ParallelTime
    )

ENDFOREACH()
