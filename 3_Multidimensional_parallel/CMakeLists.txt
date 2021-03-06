
cmake_minimum_required(VERSION 3.10)
project(flip)

# if (NOT TORCH_DIR)
#   message(FATAL_ERROR "Please setup TORCH_DIR")
# else()
#   message("Found TORCH_DIR:${TORCH_DIR}")
# endif()

# RELEASE
set(CMAKE_CXX_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -mavx -mfma -mavx2")
# DEBUG
# set(CMAKE_CXX_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g -mavx -mfma -mavx2")

add_definitions(-DCPU_CAPABILITY_AVX -DCPU_CAPABILITY_AVX2)

# include_directories(${TORCH_DIR}/include ${TORCH_DIR}/include/torch/csrc/api/include)
# set(TORCH_LIBRARIES
#     "${TORCH_DIR}/lib/libtorch.so"
#     "${TORCH_DIR}/lib/libtorch_cpu.so"
#     "${TORCH_DIR}/lib/libc10.so"
# )

find_package(Torch REQUIRED)

find_package(OpenMP)

add_executable(flip main.cpp flip.cpp flip.h)
target_link_libraries(flip PUBLIC ${TORCH_LIBRARIES} OpenMP::OpenMP_CXX)
set_property(TARGET flip PROPERTY CXX_STANDARD 14)