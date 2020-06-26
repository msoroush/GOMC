cmake_minimum_required(VERSION 2.6)

set(This ExampleTest)

set(Sources
  ExampleTest.cpp
)

add_executable(${This} ${Sources})

target_link_libraries(${This} PUBLIC
  gtest_main
  NPT
)

add_test(
  NAME ${This}
  COMMAND ${This}
)