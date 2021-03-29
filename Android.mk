LOCAL_PATH:= $(call my-dir)

WP2_CFLAGS := -Wall -DANDROID -DHAVE_MALLOC_H -DHAVE_PTHREAD
WP2_CFLAGS += -fvisibility=hidden

ifeq ($(APP_OPTIM),release)
  WP2_CFLAGS += -finline-functions -ffast-math \
                -ffunction-sections -fdata-sections
  ifeq ($(findstring clang,$(NDK_TOOLCHAIN_VERSION)),)
    WP2_CFLAGS += -frename-registers -s
  endif
endif

ifneq ($(findstring armeabi-v7a, $(TARGET_ARCH_ABI)),)
  # Setting LOCAL_ARM_NEON will enable -mfpu=neon which may cause illegal
  # instructions to be generated for armv7a code. Instead target the neon code
  # specifically.
  NEON := cc.neon
  USE_CPUFEATURES := yes
else
  NEON := cc
endif

enc_srcs := \
  src/enc/alpha_enc.cc \
  src/enc/analysis_enc.cc \
  src/enc/block_enc.cc \
  src/enc/bypass_enc.cc \
  src/enc/config_enc.cc \
  src/enc/distortion_enc.cc \
  src/enc/lossless_enc.cc \
  src/enc/lossy_enc.cc \
  src/enc/main_enc.cc \
  src/enc/nearlossless_enc.cc \
  src/enc/predictor_enc.cc \
  src/enc/residuals_enc.cc \
  src/enc/residuals_enc_aom.cc \
  src/enc/segment_enc.cc \
  src/enc/symbols_enc.cc \
  src/enc/syntax_enc.cc \
  src/enc/writer_enc.cc \
  src/enc/anim/anim_enc.cc \
  src/enc/anim/anim_frame.cc \
  src/enc/anim/anim_rectangle.cc \
  src/enc/lossless/analysisl_enc.cc \
  src/enc/lossless/backward_references_cost_enc.cc \
  src/enc/lossless/backward_references_enc.cc \
  src/enc/lossless/group4_enc.cc \
  src/enc/lossless/histogram_enc.cc \
  src/enc/lossless/losslessi_enc.cc \
  src/enc/lossless/palette.cc \
  src/enc/lossless/predictorl_enc.cc \
  src/enc/partitioning/partition_enc.cc \
  src/enc/partitioning/partition_score_func.cc \
  src/enc/partitioning/partition_score_func_area.cc \
  src/enc/partitioning/partition_score_func_block.cc \
  src/enc/partitioning/partition_score_func_multi.cc \
  src/enc/partitioning/partition_score_func_tile.cc \
  src/enc/partitioning/partitioner.cc \
  src/enc/partitioning/partitioner_area.cc \
  src/enc/partitioning/partitioner_exhaustive.cc \
  src/enc/partitioning/partitioner_multi.cc \
  src/enc/partitioning/partitioner_split.cc \
  src/enc/partitioning/partitioner_top_left.cc \
  src/enc/preview/preview_analysis.cc \
  src/enc/preview/preview_color.cc \
  src/enc/preview/preview_config.cc \
  src/enc/preview/preview_enc.cc \
  src/enc/preview/preview_opt.cc \
  src/enc/screen_content/screen_enc.cc \

dec_srcs := \
  src/dec/tile_dec.cc \
  src/dec/syntax_dec.cc \
  src/dec/symbols_dec.cc \
  src/dec/residuals_dec_aom.cc \
  src/dec/residuals_dec.cc \
  src/dec/main_dec.cc \
  src/dec/lossy_dec.cc \
  src/dec/lossless_dec.cc \
  src/dec/icc_dec.cc \
  src/dec/features_dec.cc \
  src/dec/bypass_dec.cc \
  src/dec/anim_dec.cc \
  src/dec/alpha_dec.cc \
  src/dec/preview/preview_dec.cc \
  src/dec/lossless/losslessi_dec.cc \
  src/dec/lossless/group4_dec.cc \
  src/dec/incr/decoder_tiles.cc \
  src/dec/incr/decoder_state.cc \
  src/dec/incr/decoder_stages.cc \
  src/dec/incr/decoder_skip.cc \
  src/dec/incr/decoder_info.cc \
  src/dec/incr/decoder_context.cc \
  src/dec/incr/decoder_api.cc \
  src/dec/filters/restoration_filter.cc \
  src/dec/filters/intratile_filter.cc \
  src/dec/filters/intertile_filter.cc \
  src/dec/filters/grain_filter.cc \
  src/dec/filters/directional_filter.cc \
  src/dec/filters/deblocking_filter.cc \
  src/dec/filters/block_map_filter.cc \
  src/dec/filters/alpha_filter.cc \

common_srcs := \
  src/common/global_params.cc \
  src/common/header_enc_dec.cc \
  src/common/integral.cc \
  src/common/progress_watcher.cc \
  src/common/symbols.cc \
  src/common/vdebug.cc \
  src/common/filters/rstr_flt_params.cc \
  src/common/lossless/color_cache.cc \
  src/common/lossy/block.cc \
  src/common/lossy/block_size.cc \
  src/common/lossy/block_size_io.cc \
  src/common/lossy/context.cc \
  src/common/lossy/predictor.cc \
  src/common/lossy/quant_mtx.cc \
  src/common/lossy/residuals.cc \
  src/common/lossy/residuals_aom.cc \
  src/common/lossy/rnd_mtx.cc \
  src/common/lossy/segment.cc \
  src/common/lossy/transforms.cc \
  src/common/preview/preview.cc \
  src/common/preview/preview_rasterizer.cc \

utils_srcs := \
  src/utils/ans.cc \
  src/utils/ans_enc.cc \
  src/utils/ans_cost_table.cc \
  src/utils/ans_utils.cc \
  src/utils/argb_buffer.cc \
  src/utils/context_switch.cc \
  src/utils/csp.cc \
  src/utils/data.cc \
  src/utils/data_source.cc \
  src/utils/data_source_context.cc \
  src/utils/data_source_stream.cc \
  src/utils/front_mgr.cc \
  src/utils/memory.cc \
  src/utils/orientation.cc \
  src/utils/plane.cc \
  src/utils/quantizer.cc \
  src/utils/split_iterator.cc \
  src/utils/status.cc \
  src/utils/thread_utils.cc \
  src/utils/utils.cc \
  src/utils/wiener.cc \

dsp_srcs := \
  src/dsp/alpha.cc \
  src/dsp/ans_dsp.cc \
  src/dsp/argb_converter.cc \
  src/dsp/cdef_dsp.cc \
  src/dsp/cpu.cc \
  src/dsp/csp_dsp.cc \
  src/dsp/deblocking_filter_dsp.cc \
  src/dsp/dec_dsp.cc \
  src/dsp/enc_dsp.cc \
  src/dsp/grain_c.cc \
  src/dsp/intra_dsp.cc \
  src/dsp/math.cc \
  src/dsp/quant_dsp.cc \
  src/dsp/raster_dsp.cc \
  src/dsp/score_dsp.cc \
  src/dsp/ssim_dsp.cc \
  src/dsp/transf_c.cc \
  src/dsp/transf_sse.cc \
  src/dsp/wiener_dsp.cc \
  src/dsp/lossless/decl_dsp.cc \
  src/dsp/lossless/dspl.cc \
  src/dsp/lossless/encl_dsp.cc \

imageio_srcs := \
  imageio/anim_image_dec.cc \
  imageio/bmpdec.cc \
  imageio/bmpenc.cc \
  imageio/file_format.cc \
  imageio/gifdec.cc \
  imageio/gifdec_metadata.cc \
  imageio/image_dec.cc \
  imageio/image_enc.cc \
  imageio/imageio_util.cc \
  imageio/jpegdec.cc \
  imageio/pngdec.cc \
  imageio/pngenc.cc \
  imageio/pnmdec.cc \
  imageio/pnmenc.cc \
  imageio/tiffdec.cc \
  imageio/tiffenc.cc \
  imageio/webpdec.cc \
  imageio/webpenc.cc \
  imageio/wp2dec.cc \

################################################################################

include $(CLEAR_VARS)

LOCAL_SRC_FILES := \
    $(enc_srcs) \
    $(dec_srcs) \
    $(common_srcs) \
    $(utils_srcs) \
    $(dsp_srcs) \
    $(imageio_srcs) \

LOCAL_CFLAGS := $(WP2_CFLAGS)
LOCAL_C_INCLUDES += $(LOCAL_PATH)/src

# prefer arm over thumb mode for performance gains
LOCAL_ARM_MODE := arm

LOCAL_MODULE := wp2_static

include $(BUILD_STATIC_LIBRARY)

ifeq ($(ENABLE_SHARED),1)
  include $(CLEAR_VARS)
  LOCAL_WHOLE_STATIC_LIBRARIES := wp2_static
  LOCAL_MODULE := wp2

  include $(BUILD_SHARED_LIBRARY)
endif

################################################################################

include $(LOCAL_PATH)/examples/Android.mk

$(call import-module,android/cpufeatures)
