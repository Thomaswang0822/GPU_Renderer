cmake_minimum_required(VERSION 3.21)
project(GPU_Renderer LANGUAGES CUDA)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "86")

# Collect all source files from src/ directory
file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

# Collect all header files from include/ directory
file(GLOB_RECURSE HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
)

add_executable(${PROJECT_NAME} ${SRC_FILES} ${HEADER_FILES})

# Set include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set the language for CUDA source files
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu PROPERTIES LANGUAGE CUDA)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Generate out.ppm
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/out.ppm
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME} > ${CMAKE_CURRENT_BINARY_DIR}/out.ppm
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(out_ppm
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/out.ppm
)


# Generate out.jpg
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/out.jpg
    COMMAND ppmtojpeg ${CMAKE_CURRENT_BINARY_DIR}/out.ppm > ${CMAKE_CURRENT_BINARY_DIR}/out.jpg
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/out.ppm
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(out_jpg
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/out.jpg
)

# clean target
add_custom_target(clean_build
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/out.ppm
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/out.jpg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
