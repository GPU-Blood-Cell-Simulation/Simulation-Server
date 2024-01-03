#
# Tests
#
include(CTest)

if (BUILD_TESTING)
    

    # Sample test
    add_executable(sample_test sample_test.cpp)
    target_link_libraries(sample_test PRIVATE GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(sample_test)
    
endif()