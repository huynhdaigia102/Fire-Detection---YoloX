cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(yolox_plugins LANGUAGES CXX)
enable_language(CXX)
include(FindCUDA)
set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)
enable_language(CUDA)
# find_package(OpenCV REQUIRED)
find_package(PkgConfig)
pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.4
                               gstreamer-sdp-1.0>=1.4
                               gstreamer-video-1.0>=1.4
                               gstreamer-app-1.0>=1.4)

if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
if(DEFINED DeepStream_DIR)
include_directories("${DeepStream_DIR}/sources/includes")
  link_directories("${DeepStream_DIR}/lib")
endif(DEFINED DeepStream_DIR)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GST_INCLUDE_DIRS})
include_directories(include/)

message(STATUS "====================${PLATFORM_TEGRA}")

aux_source_directory(gstyoloxparser gstyoloxparser_files)
add_library(gstyoloxparser SHARED
  ../../cpp/csrc/cuda/decode.h
  ../../cpp/csrc/cuda/decode.cu
  ../../cpp/csrc/cuda/nms.h
  ../../cpp/csrc/cuda/nms.cu
  ../../cpp/csrc/cuda/nms_rotated.h
  ../../cpp/csrc/cuda/nms_rotated.cu
  ../../cpp/engine.cpp
  ${gstyoloxparser_files}
)
message(STATUS "====================${gstyoloxparser_files}")
set_target_properties(gstyoloxparser PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(gstyoloxparser PUBLIC nvinfer nvinfer_plugin 
                      nvonnxparser nvdsgst_helper nvds_meta nvdsgst_meta dl
                      cuda ${CUDA_LIBRARIES} ${GST_LIBRARIES})
set_target_properties(gstyoloxparser PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
if(DEFINED PLATFORM_TEGRA)
  set(GST_INSTALL_DIR /usr/lib/aarch64-linux-gnu/gstreamer-1.0/)
else()
  set(GST_INSTALL_DIR /usr/lib/x86_64-linux-gnu/gstreamer-1.0/)
endif()

install(
  TARGETS gstyoloxparser
  LIBRARY DESTINATION ${GST_INSTALL_DIR}
)

