LOCAL_PATH := $(call my-dir)

################################################################################
# utils library

include $(CLEAR_VARS)

LOCAL_SRC_FILES := \
  custom_memory.cc \
  example_utils.cc \


LOCAL_CFLAGS := $(WP2_CFLAGS)
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../src $(LOCAL_PATH)/..

LOCAL_MODULE := utils_static

include $(BUILD_STATIC_LIBRARY)

ifeq ($(ENABLE_SHARED),1)
  include $(CLEAR_VARS)
  LOCAL_WHOLE_STATIC_LIBRARIES := utils_static
  LOCAL_MODULE := utils

  include $(BUILD_SHARED_LIBRARY)
endif

################################################################################
# cwp2

include $(CLEAR_VARS)

LOCAL_CFLAGS := $(WP2_CFLAGS)
LOCAL_C_INCLUDES := $(LOCAL_PATH)/../src $(LOCAL_PATH)/..
LOCAL_STATIC_LIBRARIES := wp2_static utils_static

LOCAL_SRC_FILES := cwp2.cc
LOCAL_MODULE := cwp2
include $(BUILD_EXECUTABLE)

################################################################################
# dwp2

include $(CLEAR_VARS)

LOCAL_CFLAGS := $(WP2_CFLAGS)
LOCAL_C_INCLUDES := $(LOCAL_PATH)/../src $(LOCAL_PATH)/..
LOCAL_STATIC_LIBRARIES := wp2_static utils_static
LOCAL_SRC_FILES := dwp2.cc
LOCAL_MODULE := dwp2
include $(BUILD_EXECUTABLE)

################################################################################
