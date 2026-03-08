cmake_minimum_required(VERSION 3.23)
project(vpmr LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(CGAL REQUIRED)
if (NOT TARGET CGAL::CGAL_Core)
    add_library(CGAL_Core_compat INTERFACE)
    target_link_libraries(CGAL_Core_compat INTERFACE CGAL::CGAL)
    add_library(CGAL::CGAL_Core ALIAS CGAL_Core_compat)
endif ()
find_package(CUDAToolkit REQUIRED)

set(OPTIX_INSTALL_DIR "" CACHE PATH "Path to OptiX SDK installation")
if (NOT OPTIX_INSTALL_DIR)
    message(FATAL_ERROR "Set OPTIX_INSTALL_DIR to the OptiX SDK installation path")
endif ()
set(OPTIX_INCLUDE_DIR "${OPTIX_INSTALL_DIR}/include")
if (NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
    message(FATAL_ERROR "Cannot find optix.h in ${OPTIX_INCLUDE_DIR}")
endif ()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/libigl/cmake")
option(LIBIGL_COPYLEFT_CGAL "Build libigl with CGAL" ON)
add_subdirectory(external/libigl)

add_library(maxflow_lib STATIC external/maxflow/maxflow.cpp)
target_include_directories(maxflow_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/external/maxflow")
target_include_directories(maxflow_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/external/maxflow/maxflow")

include(cmake/embed_ptx.cmake)
embed_ptx(
    OUTPUT_TARGET vpmr_ptx
    SOURCES source/visual_measures/optix_programs.cu
)

# Core library (non-GPU sources, shared between main executable and tests)
add_library(vpmr_core STATIC
    source/mesh.cpp
    source/utils/mesh_io.cpp
    source/utils/mesh_utils.cpp
    source/utils/sampling.cpp
    source/utils/bounding_box.cpp
    source/orientation/orientation.cpp
    source/offset/offset.cpp
    source/partition/partition.cpp
    source/partition/bsp.cpp
    source/graph_cut/graph_cut.cpp
    source/simplification/simplification.cpp
    source/topology/topology.cpp
    source/attributes/attributes.cpp
)

target_include_directories(vpmr_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/source"
)

target_link_libraries(vpmr_core PUBLIC
    Eigen3::Eigen
    igl_copyleft::cgal
    CGAL::CGAL
    maxflow_lib
)

add_executable(vpmr
    source/main.cpp
    source/vpmr.cpp
    source/visual_measures/visual_measures.cpp
    source/visual_measures/optix_ray_tracer.cpp
)

target_include_directories(vpmr PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/source"
    "${OPTIX_INCLUDE_DIR}"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)

target_link_libraries(vpmr PRIVATE
    vpmr_core
    CUDA::cuda_driver
    CUDA::cudart
    vpmr_ptx
)

# Tests
option(VPMR_BUILD_TESTS "Build unit tests" ON)
if (VPMR_BUILD_TESTS)
    add_subdirectory(tests)
endif ()
