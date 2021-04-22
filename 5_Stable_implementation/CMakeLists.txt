
cmake_minimum_required(VERSION 3.10)
project(flip)

# if (NOT TORCH_DIR)
#   message(FATAL_ERROR "Please setup TORCH_DIR")
# else()
#   message("Found TORCH_DIR:${TORCH_DIR}")
# endif()

# RELEASE
set(CMAKE_CXX_FLAGS "${CMAKE_CUDA_FLAGS} -mavx -mfma -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -DNDEBUG -O3 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow")
# DEBUG
# set(CMAKE_CXX_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g -mavx -mfma -mavx2")

add_definitions(-DCPU_CAPABILITY_AVX -DCPU_CAPABILITY_AVX2)

# include_directories(${TORCH_DIR}/include ${TORCH_DIR}/include/torch/csrc/api/include)
# set(TORCH_LIBRARIES
#     "${TORCH_DIR}/lib/libtorch.so"
#     "${TORCH_DIR}/lib/libtorch_cpu.so"
#     "${TORCH_DIR}/lib/libc10.so"
# )

# CPLUS_INCLUDE_PATH=$CUDA_HOME/include cmake .. -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DTorch_DIR=$TORCH_PATH/share/^Cake/Torch -DCMAKE_BUILD_TYPE=Debug

find_package(Torch REQUIRED)

find_package(OpenMP)

add_library(flip SHARED flip.cpp flip.h)
target_link_libraries(flip PUBLIC ${TORCH_LIBRARIES} OpenMP::OpenMP_CXX)
set_property(TARGET flip PROPERTY CXX_STANDARD 14)

add_executable(flip_test main.cpp)
target_link_libraries(flip_test PUBLIC flip)
set_property(TARGET flip_test PROPERTY CXX_STANDARD 14)

set_target_properties(flip_test
    PROPERTIES OUTPUT_NAME flip)
