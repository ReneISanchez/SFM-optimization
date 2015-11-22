cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(sfm-gui)

find_package(Qt5Widgets) 

# ----------------------------------------------
# OpenCV
# ----------------------------------------------
find_package(OpenCV REQUIRED)


# ----------------------------------------------
# PCL
# ----------------------------------------------
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



# ----------------------------------------------
# Sources / headers
# ----------------------------------------------
set(sfm-gui_SRC src/main.cpp src/CloudViewer.cpp)
set(sfm-gui_HEADERS include)


# ----------------------------------------------
# SfM core code
# ----------------------------------------------
set(SFM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../Source" CACHE PATH "docstring")
set(SFM_INCLUDE ${SFM_PATH}/SfMLib/inc)

include_directories(${sfm-gui_HEADERS} ${SFM_INCLUDE} ${SFM_INCLUDE}/features)
link_directories(${SFM_PATH})


# ----------------------------------------------
# Flags
# ----------------------------------------------
# VTK 6 from Ubuntu 14.04 has troubles with C++11
#
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


# ----------------------------------------------
# Executable
# ----------------------------------------------
add_executable (sfm-gui ${sfm-gui_SRC})

target_link_libraries (sfm-gui ${PCL_LIBRARIES})
target_link_libraries (sfm-gui ${OpenCV_LIBRARIES})
target_link_libraries (sfm-gui Qt5::Widgets)
target_link_libraries (sfm-gui sfm)


