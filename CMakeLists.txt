cmake_minimum_required(VERSION 3.27)
project(AnythingCppInference VERSION 0.0.1)

# -------------- SETTING  ------------------#
option(USE_CUDA "Enable CUDA support" ON)
option(USE_ONNX "Enable ONNX support" ON)
add_definitions(-DUSE_ONNX_H)

set(ONNXRUNTIME_ROOT "D:/dependency/onnxruntime-win-x64-gpu-1.15.1")
set(OpenCV_DIR "D:/dependency/opencv/build")
set(JSONCPP_DIR "C:/Program Files (x86)/jsoncpp")

# -------------- GENERATE  ------------------#
# set(CMAKE_CXX_COMPILER "g++") #设置C++编译器
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRCS)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/onnx)
set(LIBS )

# -------------- DEPENDENCY  ------------------#
# opencv

FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND LIBS ${OpenCV_LIBS})

# jsoncpp
include_directories(${JSONCPP_DIR}/include)
list(APPEND LIBS ${JSONCPP_DIR}/lib/jsoncpp.lib)


# onnxruntime
if (USE_ONNX)
    include_directories(${ONNXRUNTIME_ROOT}/include)
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib)
    list(APPEND LIBS ${ONNXRUNTIME_LIB})
endif()

# message(STATUS ${LIBS})
# generate lib

add_library(${CMAKE_PROJECT_NAME} SHARED ${SRCS})
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
target_link_libraries(${CMAKE_PROJECT_NAME} ${LIBS})

# # test
add_executable(main main.cpp  )

target_link_libraries(main ${LIBS} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/RelWithDebInfo/AnythingCppInference.lib)