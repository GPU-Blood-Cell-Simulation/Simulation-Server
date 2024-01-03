#
# Tests
#


if (BUILD_TESTING)
    

    # Sample tests
    add_executable(sample_tests 
        samples/sample_assert.cpp
        samples/sample_cuda_test.cu
        samples/sample_cuda_code/vector_add.cu
        samples/sample_cuda_code/vector_add.cuh
    )

    target_link_libraries(sample_tests PRIVATE GTest::gtest_main)
    
    include(GoogleTest)
    gtest_discover_tests(sample_tests)
    
endif()