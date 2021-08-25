#!/bin/bash
#

set -ex

if [ "$#" -ge 1 ]; then
  # eval so that the passed in directory can contain variables.
  BINARY_DIR=$(eval echo ${1})
else
  BINARY_DIR=.
fi
if [ "$#" -ge 2 ]; then
  TESTDATA_DIR=$(eval echo ${2})
else
  TESTDATA_DIR=.
fi

CONVERT=${BINARY_DIR}/convert
TMP_PIC1=/tmp/convert_test1.pic
TMP_PIC2=/tmp/convert_test2.pic
TMP_AVIF=/tmp/convert_test.avif
TMP_OBU1=/tmp/convert_test1.obu
TMP_OBU2=/tmp/convert_test2.obu

# Print help
${CONVERT} -h

# Simple back-and-forth with OBU inside generic Container
for SRC_AVIF in ${TESTDATA_DIR}/*.avif; do
  ${CONVERT} ${SRC_AVIF} --info -o ${TMP_PIC1}
  ${CONVERT} ${TMP_PIC1} --info -o ${TMP_AVIF}
  ${CONVERT} ${TMP_AVIF} --info -o ${TMP_PIC2}
  cmp ${TMP_PIC1} ${TMP_PIC2}
done

# Simple back-and-forth with stripped OBU inside generic Container
for SRC_AVIF in ${TESTDATA_DIR}/*.avif; do
  ${CONVERT} ${SRC_AVIF} --info --strip-obu -o ${TMP_PIC1}
  ${CONVERT} ${TMP_PIC1} --info --strip-obu -o ${TMP_AVIF}
  ${CONVERT} ${TMP_AVIF} --info --strip-obu -o ${TMP_PIC2}
  cmp ${TMP_PIC1} ${TMP_PIC2}
done

# Simple back-and-forth with OBU
for SRC_AVIF in ${TESTDATA_DIR}/*.avif; do
  ${CONVERT} ${SRC_AVIF} --info -o ${TMP_OBU1}
  ${CONVERT} ${TMP_OBU1} --info -o ${TMP_AVIF}
  ${CONVERT} ${TMP_AVIF} --info -o ${TMP_OBU2}
  cmp ${TMP_OBU1} ${TMP_OBU2}
done

# Simple back-and-forth with stripped OBU
for SRC_AVIF in ${TESTDATA_DIR}/*.avif; do
  ${CONVERT} ${SRC_AVIF} --strip-obu --info -o ${TMP_OBU1}
  ${CONVERT} ${TMP_OBU1} --strip-obu --info -o ${TMP_AVIF}
  ${CONVERT} ${TMP_AVIF} --strip-obu --info -o ${TMP_OBU2}
  cmp ${TMP_OBU1} ${TMP_OBU2}
done

# Clean-up
rm -f ${TMP_PIC1} ${TMP_PIC2} ${TMP_AVIF} ${TMP_OBU1} ${TMP_OBU2}

echo "PASS"
