cmake_minimum_required (VERSION 3.22.0)

######### # essential to make it find nvcc@11.4 #########
set(CMAKE_CUDA_COMPILER /home/xingyu/usr/local/cuda-12.2/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /home/xingyu/usr/local/cuda-12.2/)

set(CMAKE_C_COMPILER /usr/bin/gcc-10)
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)

set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

project (PointConv LANGUAGES CUDA CXX)

######### CUDA LIB #########
find_package(CUDA 11.4 REQUIRED)

######### CUDNN PATH FOR TORCH #########
set(CUDNN_INCLUDE_DIR /home/xingyu/anaconda3/envs/wm/include)
set(CUDNN_INCLUDE_PATH /home/xingyu/anaconda3/envs/wm/include)
set(CUDNN_LIB_DIR /home/xingyu/anaconda3/envs/wm/lib)
set(CUDNN_PATH /home/xingyu/anaconda3/envs/wm/liblibcudnn.so)
set(CUDNN_LIBRARY_PATH /home/xingyu/anaconda3/envs/wm/lib)

######### TORCH #########
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "libtorch")
find_package(Torch REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

######### SET TARGET #########
add_executable(PointConv main.cu _ext/include/geometry.h)
set_target_properties(PointConv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(PointConv PUBLIC cxx_std_11)

######### INCLUDE #########
include_directories("libtorch/include/")
include_directories("libtorch/include/torch/csrc/api/include/")
include_directories(${Python_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/_ext/include")

######### SUBDIR #########
add_subdirectory(_ext)
target_link_libraries(PointConv _ext)

######### ADD LIBRARY #########
file(GLOB LIBTORCH_LIBS "libtorch/lib/*.a" "libtorch/lib/*.so")
target_link_libraries(PointConv ${LIBTORCH_LIBS} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
message(STATUS ${TORCH_LIBRARIES})