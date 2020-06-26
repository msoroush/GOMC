#include <gtest/gtest.h>
#include "BasicTypes.h"

TEST(BasicTypesTest, CheckDefaultConstructorTest) {
  XYZ point;
  EXPECT_EQ(point.x, 0.0);
  EXPECT_EQ(point.y, 0.0);
  EXPECT_EQ(point.z, 0.0);
}

TEST(BasicTypesTest, CheckConstructorTest) {
  XYZ point(1.0, 99.012, -991.594);
  EXPECT_EQ(point.x, 1.0);
  EXPECT_EQ(point.y, 99.012);
  EXPECT_EQ(point.z, -991.594);
}

TEST(BasicTypesTest, EqualOperatorTest) {
  XYZ point1(1.0, 99.012, -991.594);
  XYZ point2 = point1;
  EXPECT_EQ(point2.x, 1.0);
  EXPECT_EQ(point2.y, 99.012);
  EXPECT_EQ(point2.z, -991.594);
}