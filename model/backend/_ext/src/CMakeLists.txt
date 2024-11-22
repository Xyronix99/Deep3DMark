project(_ext LANGUAGES CUDA CXX)

#get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#foreach(dir ${dirs})
#    message(STATUS "dir='${dir}'")
#endforeach()

file(GLOB CUDA_FILES "*.cu")
file(GLOB CXX_FILES "*.cpp")
# cmake will not comiple cu files without this statement
add_library(_ext ${CUDA_FILES} ${CXX_FILES} geometry.cpp geometry_gpu.cu)

target_compile_features(_ext PUBLIC cxx_std_11)
set_target_properties(_ext PROPERTIES CUDA_SEPARABLE_COMPILATION ON)