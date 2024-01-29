#include <gtest/gtest.h>

#include "../../src/utilities/math.cuh"


TEST(CudaMath, constCeil_noChange) {
	EXPECT_EQ(1.0, constCeil(1.0));
    EXPECT_EQ(0.0, constCeil(0.0));
    EXPECT_EQ(-1.0, constCeil(-1.0));
}


TEST(CudaMath, constCeil_change) {
	EXPECT_EQ(2.0, constCeil(1.2));
    EXPECT_EQ(1.0, constCeil(0.5));
    EXPECT_EQ(-1.0, constCeil(-1.7));
}

