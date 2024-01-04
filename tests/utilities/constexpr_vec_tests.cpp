#include <gtest/gtest.h>

#include "../../src/utilities/constexpr_vec.hpp"


TEST(ConstExp_vec, SquareBracketsOperator) {
	cvec vec = { .x=1.0, .y=2.0, .z=3.0 };

	EXPECT_EQ(1.0, vec[0]);
    EXPECT_EQ(2.0, vec[1]);
    EXPECT_EQ(3.0, vec[2]);
}


TEST(ConstExp_vec, ToGLMFunction) {
	cvec vec = { .x=1.0, .y=2.0, .z=3.0 };

    glm::vec3 result = vec.toGLM();

	EXPECT_EQ(1.0, result.x);
    EXPECT_EQ(2.0, result.y);
    EXPECT_EQ(3.0, result.z);
}


TEST(ConstExp_vec, Subtraction) {
    cvec vec1 = { .x=5.0, .y=7.0, .z=9.0 };
    cvec vec2 = { .x=1.0, .y=2.0, .z=3.0 };

    cvec diff = vec1 - vec2;

    EXPECT_EQ(4.0, diff.x);
    EXPECT_EQ(5.0, diff.y);
    EXPECT_EQ(6.0, diff.z);
}
