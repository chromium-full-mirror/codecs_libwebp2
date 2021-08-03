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

# check ICC (and metadata from command-line)

set -e

if [[ "$#" -ge 1 ]]; then
  # eval so that the passed in directory can contain variables.
  BINARY_DIR="$(eval echo "$1")"
else
  # assume "tests" is the current directory
  BINARY_DIR=..
fi
if [[ "$#" -ge 2 ]]; then
  TESTDATA_DIR="$(eval echo "$2")"
else
  TESTDATA_DIR=./testdata
fi

filesize="stat --format=\"%s\""
if [[ "${OSTYPE}" == "darwin"* ]]; then
  filesize="stat -f %z"
fi

CWP2="${BINARY_DIR}/cwp2 -effort 0"

ICC_DIR="${TESTDATA_DIR}/metadata"
# large.png contains 672 bytes of icc
SRC_FILE="${TESTDATA_DIR}/large.png"

TMP_WP2_1=/tmp/test_icc_1.wp2
TMP_WP2_2=/tmp/test_icc_2.wp2
TMP_WP2_A1=/tmp/test_icc_anim_1.wp2
TMP_WP2_A2=/tmp/test_icc_anim_2.wp2

${CWP2} "${SRC_FILE}" -o "${TMP_WP2_1}"
${CWP2} "${SRC_FILE}" -o "${TMP_WP2_2}" -nometadata
diff "${TMP_WP2_1}" "${TMP_WP2_2}" && die # file *should* differ

# Same but as an animation:
${CWP2} -f "${SRC_FILE}" 1 "${TMP_WP2_1}" 1 -o "${TMP_WP2_A1}" -effort 0
${CWP2} -f "${SRC_FILE}" 1 "${TMP_WP2_1}" 1 -o "${TMP_WP2_A2}" -effort 0 \
  -nometadata
diff "${TMP_WP2_A1}" "${TMP_WP2_A2}" && die # file *should* differ
# -f flag and animated input have different pipelines:
${CWP2} "${TMP_WP2_A1}" -o "${TMP_WP2_A2}"
${CWP2} "${TMP_WP2_A1}" -o "${TMP_WP2_A1}" -nometadata
diff "${TMP_WP2_A1}" "${TMP_WP2_A2}" && die # file *should* differ

echo "*** test replace metadata"
for type in iccp xmp exif; do
  echo "  type=${type}"
  ${CWP2} "${SRC_FILE}" -o "${TMP_WP2_2}" -metadata "${type}" \
    "${ICC_DIR}/ICC_524_type1.bin"
  diff "${TMP_WP2_1}" "${TMP_WP2_2}" && die # file *should* differ
done

echo "*** test predefined ICC"
# these predefined ICC profile only cost 2 bytes of storage
${CWP2} "${SRC_FILE}" -o "${TMP_WP2_1}" -nometadata
base_size=$(${filesize} "${TMP_WP2_1}")
for bin_file in ICC_524_type1.bin ICC_3144_type2.bin ICC_3144_type3.bin; do
  ${CWP2} "${SRC_FILE}" -o "${TMP_WP2_2}" -metadata iccp \
    "${ICC_DIR}/${bin_file}"
  size="$(${filesize} "${TMP_WP2_2}")"
  diff="$((size - base_size))"
  if [[ ${diff} != 2 ]]; then
    echo "difference between file size should be '2'.
    Got ${size} and ${base_size}."
    exit 1
  fi
done

# test invalid comands
echo "*** test invalid options"

set +e
${CWP2} "${SRC_FILE}" -metadata iccp /tmp -o "${TMP_WP2_2}"
${CWP2} "${SRC_FILE}" -metadata iccp
${CWP2} "${SRC_FILE}" -metadata
${CWP2} "${SRC_FILE}" -metadata xcas /tmp -o "${TMP_WP2_2}"

echo "TEST OK"
exit 0
