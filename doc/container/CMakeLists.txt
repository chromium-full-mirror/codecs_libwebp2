# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10)

# ##############################################################################
# Project

project(Container VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ##############################################################################
# Library

add_library(container_lib container.cc bit_packing.cc)

# ##############################################################################
# Tests

# GoogleTest for easy-to-read tests. It requires pthread.
find_package(GTest)
find_package(Threads)

if(GTEST_FOUND AND Threads_FOUND)
  # Test helper library
  add_library(testutil_lib testutil.cc)
  target_link_libraries(
    testutil_lib
    PUBLIC container_lib
    INTERFACE ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
  target_include_directories(testutil_lib PUBLIC ${GTEST_INCLUDE_DIRS})
  enable_testing()

  add_executable(container_test container_test.cc)
  target_link_libraries(container_test PRIVATE testutil_lib)
  target_include_directories(container_test PRIVATE ${GTEST_INCLUDE_DIRS})
  add_test(
    NAME container_test
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/container_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
else()
  message(WARNING "GoogleTest or Threads is missing, tests are disabled.")
endif()
