#
# Tests
#
if (BUILD_TESTING)
    include(GoogleTest)

    # Sample tests
    add_executable(sample_tests 
        samples/sample_assert.cpp
        samples/sample_cuda_test.cu
        samples/sample_cuda_code/vector_add.cu
        samples/sample_cuda_code/vector_add.cuh
    )
    target_link_libraries(sample_tests PRIVATE GTest::gtest_main)
    gtest_discover_tests(sample_tests)
    
    # utilities
    add_executable(utilities_tests
        utilities/cuda_threads_tests.cpp
        utilities/constexpr_vec_tests.cpp
        utilities/math_tests.cu
        utilities/cuda_vec3_tests.cu
    )
    target_link_libraries(utilities_tests PRIVATE
        GTest::gtest_main
        cuda_vec3
    )
    gtest_discover_tests(utilities_tests)

endif()