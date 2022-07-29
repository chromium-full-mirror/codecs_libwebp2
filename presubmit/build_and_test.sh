#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

# Fail on any error and display commands being run.
set -xe

# Get the absolute path of the parent directory of this script.
LIBWEBP2_ROOT="$(realpath "$(dirname "$0")/..")"
readonly LIBWEBP2_ROOT
readonly WORKSPACE=${WORKSPACE:-"$(mktemp -t webp2.build.XXXX -d)"}
readonly GOOGLETEST_REPO="https://github.com/google/googletest.git"
readonly BUILD_DIR="${WORKSPACE}/build"
SECONDS=0

#######################################
# Ensure GoogleTest repository is setup
#
# Globals:
#  GOOGLETEST_REPO googletest repository url
#  WORKSPACE directory where the build is done
#######################################
ensure_googletest() {
  local googletest_dir
  googletest_dir="${WORKSPACE}/googletest"

  if [[ ! -d "${googletest_dir}" ]] \
    || [[ "$(git -C ${googletest_dir} rev-parse --short HEAD)" != \
          "58d77fa" ]]; then
    rm -rf "${googletest_dir}"
    # 1.12.1 is the last release to support C++11.
    # Future releases will require at least C++14.
    git clone -b release-1.12.1 --depth 1 \
      "${GOOGLETEST_REPO}" "${googletest_dir}"
  fi

  opts+=("-DGTEST_SOURCE_DIR=${googletest_dir}")
}

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
opts=(
  "-DWP2_ENABLE_TESTS=ON"
  "-DWP2_BUILD_EXAMPLES=ON"
  "-DWP2_BUILD_EXTRAS=ON")
ensure_googletest

pushd "${BUILD_DIR}"
# CMake release
cmake "${LIBWEBP2_ROOT}" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DWP2_BUILD_SWIG_PY=ON "${opts[@]}"
make -j 8
ctest --output-on-failure -j 8

rm -rf ./*

# CMake debug
cmake "${LIBWEBP2_ROOT}" -DCMAKE_BUILD_TYPE=Debug "${opts[@]}"
make -j 8
# ctest --output-on-failure -j 8  # Tests take too much time to run in Debug.

rm -rf ./*
popd  # BUILD_DIR

duration=${SECONDS}
echo "Build and tests took ${duration} seconds."
