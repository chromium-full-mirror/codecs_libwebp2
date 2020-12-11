#!/bin/sh
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


# check forced memory failures

set -e

if [ "$#" -ge 1 ]; then
  # eval so that the passed in directory can contain variables.
  BINARY_DIR=$(eval echo ${1})
else
  # assume "tests" is the current directory
  BINARY_DIR=..
fi
if [ "$#" -ge 2 ]; then
  TESTDATA_DIR=$(eval echo ${2})
else
  TESTDATA_DIR=./testdata
fi

CWP2=${BINARY_DIR}/cwp2
DWP2=${BINARY_DIR}/dwp2
GET_DISTO=${BINARY_DIR}/get_disto

SRC_FILE=${TESTDATA_DIR}/logo.png
TMP_WP2=/tmp/test_memory.wp2
TMP_PPM=/tmp/test_memory.ppm
TMP_PNG=/tmp/test_memory.png

set +e

# test cwp2 in lossy/lossless mode
echo "*** testing cwp2"
for effort in 2 3 4 5 6; do
  for fail_at in `seq 1 50`; do
    opts="${SRC_FILE} -effort ${effort} -mem_fail ${fail_at}"
    ${CWP2} "${opts}" -q 75 &> /dev/null
    ${CWP2} "${opts}" -q 100 &> /dev/null
  done
done

# test dwp2
echo "*** testing dwp2"
${CWP2} -o ${TMP_WP2} "${SRC_FILE}" -q 75
for fail_at in `seq 1 50`; do
  ${DWP2} "${TMP_WP2}" -mem_fail ${fail_at} &> /dev/null
done

# test get_disto
echo "*** testing get_disto"
${DWP2} "${TMP_WP2}" -o "${TMP_PNG}"
for fail_at in `seq 1 50`; do
  ${GET_DISTO} "${SRC_FILE}" "${TMP_PNG}" \
               -mem_fail ${fail_at} &> /dev/null
done

rm -f ${TMP_WP2} ${TMP_PPM} ${TMP_PNG}

set -e

echo "TEST OK"
exit 0
