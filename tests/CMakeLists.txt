include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(vpmr_tests
    test_mesh.cpp
    test_bounding_box.cpp
    test_sampling.cpp
    test_mesh_io.cpp
    test_orientation.cpp
    test_offset.cpp
    test_bsp.cpp
    test_partition.cpp
    test_graph_cut.cpp
    test_graph_cut_diagnostic.cpp
    test_simplification.cpp
    test_topology.cpp
    test_attributes.cpp
)

target_link_libraries(vpmr_tests PRIVATE
    vpmr_core
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(vpmr_tests)
