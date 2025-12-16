cmake_minimum_required (VERSION 3.18)
project (ACMMP)

find_package(CUDA 10.0 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)

# Set CUDA architectures - support modern GPUs
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89)

# Ensure consistent runtime library usage
if(MSVC)
    # Force dynamic runtime library for all configurations
    foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
    endforeach()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--compiler-options;-MD)  # Match C++ dynamic runtime
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific flags
if(MSVC)
    # MSVC-specific flags
    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-DNOMINMAX)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3") # Warning level 3
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /fp:fast")
else()
    # GCC/Clang flags
    add_definitions(-Wall -Wextra -pedantic -Wno-unused-function -Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -march=native")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # Note: CUDA device code doesn't typically use OpenMP, so we don't add OpenMP flags to CUDA compilation
endif()

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    ACMMP
    main.h
    ACMMP.h
    ACMMP.cpp
    ACMMP.cu
    main.cpp
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(ACMMP
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${OpenCV_LIBS}
    )
