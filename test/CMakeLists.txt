cmake_minimum_required(VERSION 2.6)

set(This GOMCTest)

set(Sources
  BasicTypesTest.cpp
)

add_executable(${This} ${Sources})

target_link_libraries(${This} PUBLIC
  gtest_main
)

add_test(
  NAME ${This}
  COMMAND ${This}
)