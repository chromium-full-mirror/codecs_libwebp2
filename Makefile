# This makefile is a simpler alternative to the autoconf-based build
# system, for simple local building of the libraries and tools.
# It will not install the libraries system-wide, but just create the 'cwp2'
# and 'dwp2' tools in the examples/ directory, along with the static
# libraries 'src/libwebp2.a', 'src/libwebp2decoder.a'
#
# To build the library and examples, use 'make'.

#### Customizable part ####

VERSION=0.1
ARCHIVE_FILE=wp2-$(VERSION).tar.gz

EXTRA_LIBS=
EXTRA_FLAGS=

# These flags assume you have libpng, libjpeg, libtiff, libgif and libwebp
# installed. If not, either follow the install instructions below or just
# comment out the next five lines.
EXTRA_FLAGS += -DWP2_HAVE_PNG -DWP2_HAVE_JPEG -DWP2_HAVE_TIFF -DWP2_HAVE_GIF
EXTRA_FLAGS += -DWP2_HAVE_WEBP
IMG_LIBS= -lpng -lz -ljpeg -ltiff -lgif \
          -lwebp -lwebpdemux -Wno-unused-command-line-argument

# point to Googletest include dir and objects.
GTEST_DIR=../googletest
GTEST_FLAGS  = -I$(GTEST_DIR)/googletest/include
GTEST_FLAGS += -I$(GTEST_DIR)/googlemock/include
GTEST_LIBS = -L$(GTEST_DIR)/lib -lgtest -lgtest_main

# OPENGL setup
EXTRA_FLAGS += -DWP2_HAVE_OPENGL
ifeq ($(strip $(shell uname)), Darwin)
  # Work around a problem linking tables marked as common symbols,
  # cf., src/enc/yuv.[hc]
  # Failure observed with: gcc 4.2.1 and 4.0.1.
  EXTRA_FLAGS += -fno-common
  EXTRA_FLAGS += -Wno-deprecated-declarations
  # this is for using regular glut:
  #    EXTRA_FLAGS += -DHAVE_GLUT_GLUT_H
  #    GL_LIBS = -framework GLUT -framework OpenGL -lglut
  # anmd this is for freeglut:
  EXTRA_FLAGS += -DWP2_HAVE_FREEGLUT
  GL_LIBS = -lglut -lGL
else
  GL_LIBS = -lglut -lGL
endif

# SDL flags: use sdl-config if it exists
SDL_CONFIG = $(shell sdl-config --version 2> /dev/null)
ifneq ($(SDL_CONFIG),)
  SDL_LIBS = $(shell sdl-config --libs)
  SDL_FLAGS = $(shell sdl-config --cflags)
else
  # use best-guess
  SDL_LIBS = -lSDL
  SDL_FLAGS =
endif

# To install libraries on Mac OS X:
# 1. Install MacPorts (http://www.macports.org/install.php)
# 2. Run "sudo port install jpeg"
# 3. Run "sudo port install libpng"
# 4. Run "sudo port install tiff"
# 5. Run "sudo port install giflib"

# To install libraries on Linux:
# 1. Run "sudo apt-get install libjpeg62-dev"
# 2. Run "sudo apt-get install libpng12-dev"
# 3. Run "sudo apt-get install libtiff4-dev"
# 4. Run "sudo apt-get install libgif-dev"

# Uncomment for build for 32bit platform
# Alternatively, you can just use the command
# 'make -f makefile.unix EXTRA_FLAGS=-m32' to that effect.
# EXTRA_FLAGS += -m32

# Extra flags to enable byte swap for 16 bit colorspaces.
# EXTRA_FLAGS += -DWP2_SWAP_16BIT_CSP

# Extra flags to enable multi-threading
EXTRA_FLAGS += -DWP2_USE_THREAD
EXTRA_LIBS += -lpthread

# Control symbol visibility. Comment out if your compiler doesn't support it.
EXTRA_FLAGS += -fvisibility=hidden

# Extra flags to emulate C89 strictness with the full ANSI
EXTRA_FLAGS += -Wmissing-declarations
# EXTRA_FLAGS += -Wdeclaration-after-statement
EXTRA_FLAGS += -Wshadow
EXTRA_FLAGS += -Wformat-security
# EXTRA_FLAGS += -Wvla

# fetch git hash
GIT_HASH = $(shell git rev-parse --short HEAD)
ifneq ($(GIT_HASH),)
  EXTRA_FLAGS += -DWP2_GIT_VERSION=\"$(GIT_HASH)\"
endif

# SSE4.2-specific flags. If unsure about your compiler's defines, try:
#    $(CC) -dM -E - < /dev/null | grep SSE
ifeq ($(HAVE_SSE), 1)
EXTRA_FLAGS += -DWP2_HAVE_SSE
endif
EXTRA_FLAGS += -msse4

# AVX2-specific flags:
ifeq ($(HAVE_AVX2), 1)
EXTRA_FLAGS += -DWP2_HAVE_AVX2 -DWP2_HAVE_SSE
src/dsp/deblocking_filter_dsp.o: EXTRA_FLAGS += -mavx2
endif

# NEON-specific flags:
# EXTRA_FLAGS += -march=armv7-a -mfloat-abi=hard -mfpu=neon -mtune=cortex-a8
# -> seems to make the overall lib slower: -fno-split-wide-types

# MIPS (MSA) 32-bit build specific flags for mips32r5 (p5600):
# EXTRA_FLAGS += -mips32r5 -mabi=32 -mtune=p5600 -mmsa -mfp64
# EXTRA_FLAGS += -msched-weight -mload-store-pairs

# MIPS (MSA) 64-bit build specific flags for mips64r6 (i6400):
# EXTRA_FLAGS += -mips64r6 -mabi=64 -mtune=i6400 -mmsa -mfp64
# EXTRA_FLAGS += -msched-weight -mload-store-pairs

# specific flags for C/C++
EXTRA_C_FLAGS = $(EXTRA_FLAGS)
EXTRA_C_FLAGS += -Wextra -Wold-style-definition
EXTRA_C_FLAGS += -Wmissing-prototypes

EXTRA_CXX_FLAGS = $(EXTRA_FLAGS)
EXTRA_CXX_FLAGS += -std=c++11

#### Nothing should normally be changed below this line ####

AR = ar
ARFLAGS = r
CPPFLAGS = -Isrc/ -I. -Wall
ifeq ($(DEBUG), 1)
  CFLAGS += -g $(EXTRA_C_FLAGS)
  CXXFLAGS += -g $(EXTRA_CXX_FLAGS)
else
  CFLAGS += -O3 -DNDEBUG $(EXTRA_C_FLAGS)
  CXXFLAGS += -O3 -DNDEBUG $(EXTRA_CXX_FLAGS)
endif

ifeq ($(TRACE), 1)
  CFLAGS += -DWP2_TRACE
  CXXFLAGS += -DWP2_TRACE
endif

ifeq ($(ERROR_TRACE), 1)
  CFLAGS += -DWP2_ERROR_TRACE
  CXXFLAGS += -DWP2_ERROR_TRACE
endif

ifeq ($(BITTRACE), 1)
  CFLAGS += -DWP2_BITTRACE
  CXXFLAGS += -DWP2_BITTRACE
endif

ifeq ($(REDUCED), 1)
  CFLAGS += -DWP2_REDUCE_BINARY_SIZE
  CXXFLAGS += -DWP2_REDUCE_BINARY_SIZE
endif

# for AOM_ACCOUNTING/INSPECTION, we need a specially built libaom, with the
# flags CONFIG_ACCOUNTING/CONFIG_INSPECTION defined and access to inner
# headers av1/decoder/... Same for CONFIG_TUNE_VMAF and CONFIG_TUNE_BUTTERAUGLI.
ifneq ($(AOM_PATH),)
  EXTRA_INC= -I${AOM_PATH} -I${AOM_PATH}/build -DWP2_HAVE_AOM_DBG -DWP2_HAVE_AOM
  CFLAGS += ${EXTRA_INC}
  CXXFLAGS += ${EXTRA_INC}
  EXTRA_LIBS += ${AOM_PATH}/build/libaom.a
  # if you've built the libaom with CONFIG_TUNE_BUTTERAUGLI, you'll need libjxl:
#  EXTRA_LIBS += -ljxl
  # and same with CONFIG_TUNE_VMAF which requires libvmaf:
#  EXTRA_LIBS += -lvmaf
endif

# For using system-wide installed libaom:
ifeq ($(AOM), 1)
  CFLAGS += -DWP2_HAVE_AOM
  CXXFLAGS += -DWP2_HAVE_AOM
  EXTRA_LIBS += -laom
endif

# AVIF support through libavif
ifeq ($(AVIF), 1)
  CFLAGS += -DWP2_HAVE_AVIF
  CXXFLAGS += -DWP2_HAVE_AVIF
  EXTRA_LIBS += -lavif
endif

ifeq ($(SJPEG), 1)
  CFLAGS += -DWP2_HAVE_SJPEG
  CXXFLAGS += -DWP2_HAVE_SJPEG
  EXTRA_LIBS += -lsjpeg
endif

ifeq ($(ENC_DEC_MATCH), 1)
  CFLAGS += -DWP2_ENC_DEC_MATCH
  CXXFLAGS += -DWP2_ENC_DEC_MATCH
endif

EXTRA_FLAGS += -I/usr/local/include -DWP2_HAVE_OPENGL
EXTRA_LIBS  += -L/usr/local/lib

ifeq ($(USE_CLANG), 1)
  CC = clang -std=gnu99
  CXX = clang++
else
  CC = gcc -std=gnu99
  CXX = g++
endif
INSTALL = install
GROFF = /usr/bin/groff
COL = /usr/bin/col
LDFLAGS += $(EXTRA_LIBS) $(EXTRA_FLAGS) -lm

ANIM_UTIL_OBJS =

DEC_OBJS = \
    src/dec/alpha_dec.o \
    src/dec/anim_dec.o \
    src/dec/av1_dec.o \
    src/dec/bypass_dec.o \
    src/dec/features_dec.o \
    src/dec/filters/alpha_filter.o \
    src/dec/filters/block_map_filter.o \
    src/dec/filters/deblocking_filter.o \
    src/dec/filters/directional_filter.o \
    src/dec/filters/grain_filter.o \
    src/dec/filters/intertile_filter.o \
    src/dec/filters/intratile_filter.o \
    src/dec/filters/restoration_filter.o \
    src/dec/icc_dec.o \
    src/dec/incr/decoder_api.o \
    src/dec/incr/decoder_context.o \
    src/dec/incr/decoder_info.o \
    src/dec/incr/decoder_skip.o \
    src/dec/incr/decoder_stages.o \
    src/dec/incr/decoder_state.o \
    src/dec/incr/decoder_tiles.o \
    src/dec/lossless/group4_dec.o \
    src/dec/lossless/losslessi_dec.o \
    src/dec/lossless/lzw_dec.o \
    src/dec/lossless_dec.o \
    src/dec/lossy_dec.o \
    src/dec/main_dec.o \
    src/dec/residuals_dec.o \
    src/dec/residuals_dec_aom.o \
    src/dec/symbols_dec.o \
    src/dec/syntax_dec.o \
    src/dec/tile_dec.o \

DSP_DEC_OBJS = \
    src/dsp/alpha.o \
    src/dsp/ans_dsp.o \
    src/dsp/argb_converter.o \
    src/dsp/cpu.o \
    src/dsp/cdef_dsp.o \
    src/dsp/csp_dsp.o \
    src/dsp/deblocking_filter_dsp.o \
    src/dsp/dec_dsp.o \
    src/dsp/intra_dsp.o \
    src/dsp/quant_dsp.o \
    src/dsp/transf_c.o \
    src/dsp/transf_sse.o \
    src/dsp/lossless/dspl.o \
    src/dsp/lossless/decl_dsp.o \
    src/dsp/raster_dsp.o \
    src/dsp/grain_c.o \
    src/dsp/math.o \
    src/dsp/ssim_dsp.o \
    src/dsp/wiener_dsp.o \

ENC_OBJS = \
    src/enc/alpha_enc.o \
    src/enc/analysis_enc.o \
    src/enc/anim/anim_enc.o \
    src/enc/anim/anim_frame.o \
    src/enc/anim/anim_rectangle.o \
    src/enc/av1_enc.o \
    src/enc/block_enc.o \
    src/enc/bypass_enc.o \
    src/enc/config_enc.o \
    src/enc/distortion_enc.o \
    src/enc/lossless/analysisl_enc.o \
    src/enc/lossless/backward_references_cost_enc.o \
    src/enc/lossless/backward_references_enc.o \
    src/enc/lossless/group4_enc.o \
    src/enc/lossless/histogram_enc.o \
    src/enc/lossless/lzw_enc.o \
    src/enc/lossless/palette.o \
    src/enc/lossless/predictorl_enc.o \
    src/enc/lossless/losslessi_enc.o \
    src/enc/symbols_enc.o \
    src/enc/lossless_enc.o \
    src/enc/lossy_enc.o \
    src/enc/main_enc.o \
    src/enc/nearlossless_enc.o \
    src/enc/partitioning/partition_enc.o \
    src/enc/partitioning/partition_score_func.o \
    src/enc/partitioning/partition_score_func_area.o \
    src/enc/partitioning/partition_score_func_block.o \
    src/enc/partitioning/partition_score_func_multi.o \
    src/enc/partitioning/partition_score_func_tile.o \
    src/enc/partitioning/partitioner.o \
    src/enc/partitioning/partitioner_area.o \
    src/enc/partitioning/partitioner_exhaustive.o \
    src/enc/partitioning/partitioner_multi.o \
    src/enc/partitioning/partitioner_split.o \
    src/enc/partitioning/partitioner_split_recurse.o \
    src/enc/partitioning/partitioner_top_left.o \
    src/enc/residuals_enc.o \
    src/enc/residuals_enc_aom.o \
    src/enc/screen_content/screen_enc.o \
    src/enc/segment_enc.o \
    src/enc/syntax_enc.o \
    src/enc/tile_enc.o \
    src/enc/trellis.o \
    src/enc/writer_enc.o \

DSP_ENC_OBJS = \
    src/dsp/enc_dsp.o \
    src/dsp/lossless/encl_dsp.o \
    src/dsp/score_dsp.o \

EX_FORMAT_DEC_HDRS = \
    imageio/image_dec.h \
    imageio/anim_image_dec.h \

EX_FORMAT_DEC_OBJS = \
    imageio/anim_image_dec.o \
    imageio/bmpdec.o \
    imageio/gifdec.o \
    imageio/gifdec_metadata.o \
    imageio/image_dec.o \
    imageio/jpegdec.o \
    imageio/pngdec.o \
    imageio/pnmdec.o \
    imageio/tiffdec.o \
    imageio/webpdec.o \
    imageio/webpenc.o \
    imageio/wp2dec.o \

EX_FORMAT_ENC_HDRS= \
    imageio/file_format.h \
    imageio/image_enc.h \

EX_FORMAT_ENC_OBJS = \
    imageio/bmpenc.o \
    imageio/image_enc.o \
    imageio/file_format.o \
    imageio/pngenc.o \
    imageio/pnmenc.o \
    imageio/tiffenc.o \

EX_UTIL_OBJS = \

IMAGE_UTIL_HDRS= \
    imageio/imageio_util.h \

IMAGE_UTIL_OBJS = \
    imageio/imageio_util.o \

ANS_OBJS = \
    src/utils/ans.o \
    src/utils/ans_utils.o \

ANS_ENC_OBJS = \
    src/utils/ans_enc.o \
    src/utils/ans_cost_table.o \

PREVIEW_OBJS = \
    src/common/preview/preview.o \
    src/common/preview/preview_rasterizer.o \
    src/dec/preview/preview_dec.o \
    src/enc/preview/preview_analysis.o \
    src/enc/preview/preview_color.o \
    src/enc/preview/preview_config.o \
    src/enc/preview/preview_enc.o \
    src/enc/preview/preview_opt.o \

UTILS_DEC_OBJS = \
    src/utils/argb_buffer.o \
    src/utils/csp.o \
    src/utils/data.o \
    src/utils/hash.o \
    src/utils/hash_map.o \
    src/utils/context_switch.o \
    src/utils/data_source.o \
    src/utils/data_source_context.o \
    src/utils/data_source_stream.o \
    src/utils/front_mgr.o \
    src/utils/memory.o \
    src/utils/orientation.o \
    src/utils/plane.o \
    src/utils/quantizer.o \
    src/utils/split_iterator.o \
    src/utils/status.o \
    src/utils/thread_utils.o \
    src/utils/utils.o \
    src/utils/wiener.o \

UTILS_ENC_OBJS = \
    src/common/global_params.o \
    src/common/header_enc_dec.o \
    src/common/integral.o \

COMMON_DEC_OBJS = \
    src/common/filters/rstr_flt_params.o \
    src/common/lossless/color_cache.o \
    src/common/lossy/block.o \
    src/common/lossy/block_size.o \
    src/common/lossy/block_size_io.o \
    src/common/lossy/context.o \
    src/common/lossy/predictor.o \
    src/common/lossy/quant_mtx.o \
    src/common/lossy/residuals.o \
    src/common/lossy/residuals_aom.o \
    src/common/lossy/rnd_mtx.o \
    src/common/lossy/segment.o \
    src/common/lossy/transforms.o \
    src/common/progress_watcher.o \
    src/common/symbols.o \
    src/common/vdebug.o \

EXAMPLE_UTILS_HDRS = \
    examples/stopwatch.h \
    examples/example_utils.h \

EXAMPLE_UTILS_OBJS = \
    examples/custom_memory.o \
    examples/example_utils.o \

EXTRAS_OBJS = \
    extras/aom_utils.o \
    extras/ccsp_imageio.o \
    extras/wp2_dec_12b.o \
    extras/extras.o \
    extras/y4menc.o \
    extras/y4mdec.o \

LIBWP2DECODER_OBJS = $(DEC_OBJS) $(DSP_DEC_OBJS) $(UTILS_DEC_OBJS) \
                     $(ANS_OBJS) $(PREVIEW_OBJS) $(COMMON_DEC_OBJS)
LIBWP2_OBJS = $(LIBWP2DECODER_OBJS) $(ENC_OBJS) $(DSP_ENC_OBJS) \
               $(UTILS_ENC_OBJS) $(ANS_ENC_OBJS)

HDRS_INSTALLED = \
    src/wp2/base.h \
    src/wp2/debug.h \
    src/wp2/decode.h \
    src/wp2/encode.h \
    src/wp2/format_constants.h \

HDRS = \
    src/common/av1_common.h \
    src/common/color_precision.h \
    src/common/constants.h \
    src/common/filters/rstr_flt_params.h \
    src/common/lossless/color_cache.h \
    src/common/lossy/block.h \
    src/common/lossy/block_size.h \
    src/common/lossy/block_size_io.h \
    src/common/lossy/context.h \
    src/common/lossy/predictor.h \
    src/common/lossy/quant_mtx.h \
    src/common/lossy/segment.h \
    src/common/lossy/transforms.h \
    src/common/lossy/residuals.h \
    src/common/lossy/residuals_aom.h \
    src/common/lossy/rnd_mtx.h \
    src/common/lossy/aom/cdfs.inc \
    src/common/global_params.h \
    src/common/header_enc_dec.h \
    src/common/integral.h \
    src/common/preview/preview.h \
    src/common/progress_watcher.h \
    src/common/symbols.h \
    src/dsp/dsp.h \
    src/dsp/dsp_x86.h \
    src/dsp/lossless/decl_dsp.h \
    src/dsp/lossless/encl_dsp.h \
    src/dsp/lossless/dspl.h \
    src/dsp/math.h \
    src/dec/filters/alpha_filter.h \
    src/dec/filters/block_map_filter.h \
    src/dec/filters/deblocking_filter.h \
    src/dec/filters/directional_filter.h \
    src/dec/filters/grain_filter.h \
    src/dec/filters/intertile_filter.h \
    src/dec/filters/intratile_filter.h \
    src/dec/filters/restoration_filter.h \
    src/dec/incr/decoder_context.h \
    src/dec/incr/decoder_info.h \
    src/dec/incr/decoder_skip.h \
    src/dec/incr/decoder_state.h \
    src/dec/lossless/losslessi_dec.h \
    src/dec/preview/preview_dec.h \
    src/dec/residuals_dec_aom.h \
    src/dec/symbols_dec.h \
    src/dec/tile_dec.h \
    src/dec/wp2_dec_i.h \
    src/enc/analysis.h \
    src/enc/anim/anim_enc.h \
    src/enc/block_enc.h \
    src/enc/lossless/losslessi_enc.h \
    src/enc/lossless/histogram_enc.h \
    src/enc/partitioning/partition_score_func.h \
    src/enc/partitioning/partition_score_func_area.h \
    src/enc/partitioning/partition_score_func_block.h \
    src/enc/partitioning/partition_score_func_multi.h \
    src/enc/partitioning/partition_score_func_tile.h \
    src/enc/partitioning/partitioner.h \
    src/enc/partitioning/partitioner_area.h \
    src/enc/partitioning/partitioner_exhaustive.h \
    src/enc/partitioning/partitioner_multi.h \
    src/enc/partitioning/partitioner_split.h \
    src/enc/partitioning/partitioner_split_recurse.h \
    src/enc/partitioning/partitioner_top_left.h \
    src/enc/preview/preview_enc.h \
    src/enc/residuals_enc_aom.h \
    src/enc/screen_content/screen_enc.h \
    src/enc/symbols_enc.h \
    src/enc/tile_enc.h \
    src/enc/trellis.h \
    src/enc/wp2_enc_i.h \
    src/utils/ans.h \
    src/utils/ans_utils.h \
    src/utils/context_switch.h \
    src/utils/csp.h \
    src/utils/data_source.h \
    src/utils/data_source_context.h \
    src/utils/data_source_stream.h \
    src/utils/front_mgr.h \
    src/utils/hash.h \
    src/utils/orientation.h \
    src/utils/plane.h \
    src/utils/random.h \
    src/utils/split_iterator.h \
    src/utils/split_iterator.cc \
    src/utils/stats.h \
    src/utils/thread_utils.h \
    src/utils/utils.h \
    src/utils/vector.h \
    src/utils/wiener.h \
    $(HDRS_INSTALLED) \

OUT_LIBS =
OUT_LIBS += imageio/libimageio_util.a
OUT_LIBS += imageio/libimagedec.a
OUT_LIBS += imageio/libimageenc.a
OUT_LIBS += extras/libextras.a
OUT_LIBS += examples/example_utils.a
OUT_LIBS += src/libwebp2decoder.a
OUT_LIBS += src/libwebp2.a
# libs for testing. Let's just include everything we could need
BASE_LIBS =
BASE_LIBS += extras/libextras.a
BASE_LIBS += imageio/libimageenc.a
BASE_LIBS += imageio/libimagedec.a
BASE_LIBS += imageio/libimageio_util.a
BASE_LIBS += examples/example_utils.a
BASE_LIBS += src/libwebp2.a

TEST_OBJS =
TEST_OBJS += tests/include/helpers_incr.o
TEST_OBJS += tests/include/helpers_filter.o
TEST_OBJS += tests/include/helpers.o
TEST_OBJS += $(BASE_LIBS)

TEST_CXX_FLAGS =
TEST_CXX_FLAGS += $(GTEST_FLAGS) -lpthread -Wno-unused-command-line-argument

OTHER_EXAMPLES =

OUTPUT = $(OUT_LIBS) $(OUT_EXAMPLES)
ifeq ($(MAKECMDGOALS),clean)
  OUTPUT += $(EXTRA_EXAMPLES) $(OTHER_EXAMPLES)
  OUTPUT += $(EXTRAS_LIB)
endif

all: ex $(EXTRA_EXAMPLES) $(OTHER_EXAMPLES) $(CC_TESTS) $(SH_TESTS)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

%.o: %.cc $(HDRS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

imageio/libimagedec.a: $(EX_FORMAT_DEC_OBJS) $(EX_FORMAT_DEC_HDRS)
imageio/libimageenc.a: $(EX_FORMAT_ENC_OBJS) $(EX_FORMAT_ENC_HDRS)
imageio/libimageio_util.a: $(IMAGE_UTIL_OBJS) $(IMAGE_UTIL_HDRS)
examples/example_utils.a: $(EXAMPLE_UTILS_OBJS) $(EXAMPLE_UTILS_HDRS)
extras/libextras.a: $(EXTRAS_OBJS)
src/libwebp2decoder.a: $(LIBWP2DECODER_OBJS)
src/libwebp2.a: $(LIBWP2_OBJS)

%.a:
	$(AR) $(ARFLAGS) $@ $^

###############################################################################
# examples
###############################################################################

examples/cwp2: examples/cwp2.o
examples/cwp2: $(BASE_LIBS)
examples/cwp2: override EXTRA_LIBS += $(IMG_LIBS)

examples/dwp2: examples/dwp2.o
examples/dwp2: $(BASE_LIBS)
examples/dwp2: override EXTRA_LIBS += $(IMG_LIBS)

examples/vwp2: examples/vwp2.o
examples/vwp2: $(BASE_LIBS)
examples/vwp2: extras/libextras.a
examples/vwp2: override EXTRA_LIBS += $(IMG_LIBS) $(GL_LIBS)

examples/get_disto: examples/get_disto.o
examples/get_disto: $(BASE_LIBS)
examples/get_disto: override EXTRA_LIBS += $(IMG_LIBS)

examples/rd_curve: examples/rd_curve.o
examples/rd_curve: $(BASE_LIBS)
examples/rd_curve: extras/libextras.a
examples/rd_curve: override EXTRA_LIBS += $(IMG_LIBS)

OUT_EXAMPLES =
OUT_EXAMPLES += examples/cwp2
OUT_EXAMPLES += examples/dwp2
OUT_EXAMPLES += examples/vwp2
OUT_EXAMPLES += examples/get_disto
OUT_EXAMPLES += examples/rd_curve

ex: $(OUT_EXAMPLES)

###############################################################################
# extras
###############################################################################

extras/vwp2_sdl: extras/vwp2_sdl.o
extras/vwp2_sdl: extras/wp2_to_sdl.o
extras/vwp2_sdl: $(BASE_LIBS)
extras/vwp2_sdl: override EXTRA_FLAGS += -DWP2_HAVE_SDL $(SDL_FLAGS)
extras/vwp2_sdl: override EXTRA_LIBS += $(SDL_LIBS) $(IMG_LIBS)

extras/av1enc: extras/av1enc.o
extras/av1enc: $(BASE_LIBS)
extras/av1enc: override EXTRA_LIBS += $(IMG_LIBS) -lpthread

extras/mk_preview: extras/mk_preview.o
extras/mk_preview: $(BASE_LIBS)
extras/mk_preview: override EXTRA_LIBS += $(IMG_LIBS) -lpthread

extras/ewp2: extras/ewp2.o
extras/ewp2: $(BASE_LIBS)
extras/ewp2: override EXTRA_LIBS += $(IMG_LIBS) -lpthread

EXTRA_EXAMPLES =
EXTRA_EXAMPLES += extras/vwp2_sdl
EXTRA_EXAMPLES += extras/av1enc
EXTRA_EXAMPLES += extras/mk_preview
EXTRA_EXAMPLES += extras/ewp2

EXTRAS_LIB = extras/libextras.a

extras: $(EXTRAS_LIB) $(EXTRA_EXAMPLES)

###############################################################################

$(OUT_EXAMPLES) $(EXTRA_EXAMPLES) $(OTHER_EXAMPLES):
	$(CXX) -o $@ $^ $(LDFLAGS)

###############################################################################
# tests
###############################################################################

dist: DESTDIR := dist
dist: OUT_EXAMPLES += $(EXTRA_EXAMPLES)
dist: all
	$(INSTALL) -m755 -d $(DESTDIR)/include/wp2 \
	           $(DESTDIR)/bin $(DESTDIR)/doc $(DESTDIR)/lib
	$(INSTALL) -m755 -s $(OUT_EXAMPLES) $(DESTDIR)/bin
	$(INSTALL) -m644 $(HDRS_INSTALLED) $(DESTDIR)/include/wp2
	$(INSTALL) -m644 src/libwebp2.a $(DESTDIR)/lib
	umask 022; \
	for m in man/[cd]wp2.1; do \
	  basenam=$$(basename $$m .1); \
	  $(GROFF) -t -e -man -T utf8 $$m \
	    | $(COL) -bx >$(DESTDIR)/doc/$${basenam}.txt; \
	  $(GROFF) -t -e -man -T html $$m \
	    | $(COL) -bx >$(DESTDIR)/doc/$${basenam}.html; \
	done

###############################################################################
# tests
###############################################################################

CC_TESTS =
CC_TESTS += test_alloc test_ans test_extras test_av1
CC_TESTS += test_math test_transf test_utils test_front_mgr
CC_TESTS += test_preview_api test_optim test_csp test_dsp test_vector
CC_TESTS += test_symbols test_segment test_quant_mtx test_incr_dec test_api
CC_TESTS += test_coded_block test_distortion test_predictor test_score
CC_TESTS += test_anim test_anim_incr test_imageio_anim test_example_utils
CC_TESTS += test_orientation test_partitions test_uv_mode test_aom_residuals
CC_TESTS += test_residuals test_efforts test_progress test_intertile_filter
CC_TESTS += test_directional_filter test_deblocking_filter test_plane
CC_TESTS += test_header test_custom_csp test_custom_csp_dec test_malloc_fail_at
CC_TESTS += test_custom_csp_enc test_custom_csp_enc_dec test_dec
CC_TESTS += test_thread test_pessimization test_formats test_global_params
CC_TESTS += test_data test_data_source test_data_source_stream test_alpha_filter
CC_TESTS += test_anim_rectangle test_block_size test_context test_buffer
CC_TESTS += test_imageio test_imageio_conversion test_imageio_util
CC_TESTS += test_simple_enc_dec test_simple_encode test_slow_enc test_trellis
CC_TESTS += test_partition_score_func test_tile_shapes test_version
CC_TESTS += test_syntax_writer test_anim_frame test_diffusion test_analysis
CC_TESTS += test_decoder_api test_restoration_filter test_color_precision
CC_TESTS += test_dsp_speed test_metadata test_bypass test_screen test_anim_skip
CC_TESTS += test_split_iterator test_slow_lossless_enc test_debug_options
CC_TESTS += test_average_color test_varint
# lossless tests:
CC_TESTS += lossless/test_color_cache lossless/test_palette
CC_TESTS += lossless/test_group4 lossless/test_histogram

# template for generating test binary rules
define TEST_binary_template
tests/$(1): tests/$$(addsuffix .o, $(1)) $(TEST_OBJS)
tests/$(1): override EXTRA_LIBS += $(IMG_LIBS) $(GTEST_LIBS)
tests/$(1): override EXTRA_CXX_FLAGS += $(TEST_CXX_FLAGS)
tests/$(1):
	$$(CXX) -o $(2) $$^ $$(LDFLAGS)
endef

define TEST_template  # generate the rules with same name for .cc and binary
$(call TEST_binary_template,$(1),$$@)
endef

define TEST_template_full   # generate the rules for binary and test script
$(call TEST_template,$(1))
$(1): tests/$(1)
	cd tests && ./$(1)
endef

$(foreach ttest, $(CC_TESTS), $(eval $(call TEST_template_full,$(ttest))))

# special sh tests, with specially named binaries

$(eval $(call TEST_binary_template,test_assert,tests/test_assert_bin))
test_assert: tests/test_assert
	cd tests && ./test_assert.sh

$(eval $(call TEST_binary_template,ansz,tests/ansz))
test_ansz: tests/ansz
	cd tests && ./test_ansz.sh

test_io: $(OUT_EXAMPLES) $(EXTRA_EXAMPLES) $(OTHER_EXAMPLES)
	cd tests && ./test_io.sh ../examples
test_io_pnm: $(OUT_EXAMPLES) $(EXTRA_EXAMPLES) $(OTHER_EXAMPLES)
	cd tests && ./test_io_pnm.sh ../examples
test_cmd: $(OUT_EXAMPLES) $(EXTRA_EXAMPLES) $(OTHER_EXAMPLES)
	cd tests && ./test_cmd.sh ../examples > /dev/null
test_enc: $(OUT_EXAMPLES) $(EXTRA_EXAMPLES) $(OTHER_EXAMPLES)
	cd tests && ./test_enc.sh ../examples
test_enc_dir: $(OUT_EXAMPLES) $(EXTRA_EXAMPLES) $(OTHER_EXAMPLES)
	cd tests && ./test_enc_dir.sh ../examples
test_enc_anim: $(OUT_EXAMPLES) $(EXTRA_EXAMPLES) $(OTHER_EXAMPLES)
	cd tests && ./test_enc_anim.sh ../examples
test_memory: $(OUT_EXAMPLES) $(EXTRA_EXAMPLES) $(OTHER_EXAMPLES)
	cd tests && ./test_memory.sh ../examples
test_icc: $(OUT_EXAMPLES) $(EXTRA_EXAMPLES) $(OTHER_EXAMPLES)
	cd tests && ./test_icc.sh ../examples

SH_TESTS =
SH_TESTS += test_assert test_ansz test_io_pnm test_cmd
SH_TESTS += test_enc test_enc_dir test_enc_anim test_memory test_icc

test: $(CC_TESTS) $(SH_TESTS)

###############################################################################

ex_clean:
	for ttest in $(CC_TESTS); do $(RM) tests/$$ttest tests/$$ttest.o; done
	for ttest in ansz test_assert_bin; do $(RM) tests/$$ttest; done
	$(RM) tests/*.o tests/*~

clean: ex_clean
	$(RM) $(OUTPUT) *~
	$(RM) -rf Testing
	for d in examples extras imageio src src/dec src/dsp src/enc src/wp2 \
	         src/utils tests man doc swig cmake src/enc/lossless \
	         src/dec/filters src/dec/incr src/dec/lossless src/dsp/lossless \
	         src/utils/lossless src/common/lossless src/common/lossy \
	         src/enc/preview src/enc/partitioning src/dec/preview \
	         src/common/preview src/common/filters/ src/common \
	         src/enc/anim src/enc/screen_content tests/include \
	         src/common/lossy/aom; \
	do \
           $(RM) -f $$d/*.o $$d/*~ $$d/*.a; \
        done
	$(RM) -f wp2_js/*.bc wp2_js/*~

DIST_FILES= \
            API.txt \
            Makefile \
            README.md \
            README.wp2_js \
            LICENSE \
            CMakeLists.txt \
            cmake/android.cmake \
            cmake/compiler.cmake \
            cmake/cpu.cmake \
            cmake/thread.cmake \
            cmake/wp2Config.cmake.in \
            doc/vwp2.webp \
            examples/cwp2.cc \
            examples/dwp2.cc \
            examples/vwp2.cc \
            examples/get_disto.cc \
            examples/rd_curve.cc \
            examples/custom_memory.cc \
            examples/example_utils.cc \
            examples/example_utils.h \
            examples/stopwatch.h \
            extras/aom_utils.h \
            extras/aom_utils.cc \
            extras/av1enc.cc \
            extras/mk_preview.cc \
            extras/ccsp_imageio.h \
            extras/ccsp_imageio.cc \
            extras/extras.h \
            extras/ewp2.cc \
            extras/extras.cc \
            extras/vwp2_sdl.cc \
            extras/wp2_dec_12b.cc \
            extras/wp2_to_sdl.h \
            extras/wp2_to_sdl.cc \
            extras/y4mdec.cc \
            extras/y4menc.cc \
            imageio/anim_image_dec.cc \
            imageio/anim_image_dec.h \
            imageio/bmpdec.cc \
            imageio/bmpenc.cc \
            imageio/file_format.cc \
            imageio/file_format.h \
            imageio/gifdec.cc \
            imageio/gifdec_metadata.cc \
            imageio/image_dec.cc \
            imageio/image_dec.h \
            imageio/image_enc.cc \
            imageio/image_enc.h \
            imageio/imageio_util.cc \
            imageio/imageio_util.h \
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
            man/dwp2.1 \
            man/cwp2.1 \
            man/vwp2.1 \
            src/common/av1_common.h \
            src/common/color_precision.h \
            src/common/constants.h \
            src/common/filters/rstr_flt_params.h \
            src/common/filters/rstr_flt_params.cc \
            src/common/global_params.h \
            src/common/global_params.cc \
            src/common/header_enc_dec.h \
            src/common/header_enc_dec.cc \
            src/common/integral.h \
            src/common/integral.cc \
            src/common/lossless/color_cache.h \
            src/common/lossless/color_cache.cc \
            src/common/lossy/aom/cdfs.inc \
            src/common/lossy/block.h \
            src/common/lossy/block.cc \
            src/common/lossy/block_size.h \
            src/common/lossy/block_size.cc \
            src/common/lossy/block_size_io.h \
            src/common/lossy/block_size_io.cc \
            src/common/lossy/context.h \
            src/common/lossy/context.cc \
            src/common/lossy/predictor.h \
            src/common/lossy/predictor.cc \
            src/common/lossy/quant_mtx.h \
            src/common/lossy/quant_mtx.cc \
            src/common/lossy/residuals.h \
            src/common/lossy/residuals.cc \
            src/common/lossy/residuals_aom.h \
            src/common/lossy/residuals_aom.cc \
            src/common/lossy/rnd_mtx.h \
            src/common/lossy/rnd_mtx.cc \
            src/common/lossy/segment.h \
            src/common/lossy/segment.cc \
            src/common/lossy/transforms.h \
            src/common/lossy/transforms.cc \
            src/common/preview/preview.h \
            src/common/preview/preview.cc \
            src/common/preview/preview_rasterizer.cc \
            src/common/progress_watcher.h \
            src/common/progress_watcher.cc \
            src/common/symbols.h \
            src/common/symbols.cc \
            src/common/vdebug.cc \
            src/common/vdebug.h \
            src/dec/alpha_dec.cc \
            src/dec/anim_dec.cc \
            src/dec/av1_dec.cc \
            src/dec/bypass_dec.cc \
            src/dec/features_dec.cc \
            src/dec/filters/alpha_filter.h \
            src/dec/filters/alpha_filter.cc \
            src/dec/filters/block_map_filter.h \
            src/dec/filters/block_map_filter.cc \
            src/dec/filters/deblocking_filter.h \
            src/dec/filters/deblocking_filter.cc \
            src/dec/filters/directional_filter.h \
            src/dec/filters/directional_filter.cc \
            src/dec/filters/grain_filter.h \
            src/dec/filters/grain_filter.cc \
            src/dec/filters/intertile_filter.h \
            src/dec/filters/intertile_filter.cc \
            src/dec/filters/intratile_filter.h \
            src/dec/filters/intratile_filter.cc \
            src/dec/filters/restoration_filter.h \
            src/dec/filters/restoration_filter.cc \
            src/dec/icc_dec.cc \
            src/dec/incr/decoder_api.cc \
            src/dec/incr/decoder_context.h \
            src/dec/incr/decoder_context.cc \
            src/dec/incr/decoder_info.h \
            src/dec/incr/decoder_info.cc \
            src/dec/incr/decoder_skip.h \
            src/dec/incr/decoder_skip.cc \
            src/dec/incr/decoder_stages.cc \
            src/dec/incr/decoder_state.h \
            src/dec/incr/decoder_state.cc \
            src/dec/incr/decoder_tiles.cc \
            src/dec/lossless_dec.cc \
            src/dec/lossy_dec.cc \
            src/dec/main_dec.cc \
            src/dec/preview/preview_dec.h \
            src/dec/preview/preview_dec.cc \
            src/dec/residuals_dec.cc \
            src/dec/residuals_dec_aom.h \
            src/dec/residuals_dec_aom.cc \
            src/dec/symbols_dec.h \
            src/dec/symbols_dec.cc \
            src/dec/syntax_dec.cc \
            src/dec/tile_dec.h \
            src/dec/tile_dec.cc \
            src/dec/wp2_dec_i.h \
            src/dsp/alpha.cc \
            src/dsp/ans_dsp.cc \
            src/dsp/argb_converter.cc \
            src/dsp/cpu.cc \
            src/dsp/cdef_dsp.cc \
            src/dsp/csp_dsp.cc \
            src/dsp/deblocking_filter_dsp.cc \
            src/dsp/dec_dsp.cc \
            src/dsp/dsp.h \
            src/dsp/dsp_x86.h \
            src/dsp/enc_dsp.cc \
            src/dsp/raster_dsp.cc \
            src/dsp/grain_c.cc \
            src/dsp/math.cc \
            src/dsp/math.h \
            src/dsp/score_dsp.cc \
            src/dsp/ssim_dsp.cc \
            src/dsp/intra_dsp.cc  \
            src/dsp/quant_dsp.cc  \
            src/dsp/transf_c.cc  \
            src/dsp/transf_sse.cc  \
            src/dsp/wiener_dsp.cc \
            src/enc/alpha_enc.cc \
            src/enc/analysis.h \
            src/enc/analysis_enc.cc \
            src/enc/anim/anim_enc.h \
            src/enc/anim/anim_enc.cc \
            src/enc/anim/anim_frame.cc \
            src/enc/anim/anim_rectangle.cc \
            src/enc/av1_enc.cc \
            src/enc/block_enc.h \
            src/enc/block_enc.cc \
            src/enc/bypass_enc.cc \
            src/enc/config_enc.cc \
            src/enc/distortion_enc.cc \
            src/enc/lossless_enc.cc \
            src/enc/lossy_enc.cc \
            src/enc/main_enc.cc \
            src/enc/nearlossless_enc.cc \
            src/enc/partitioning/partition_enc.cc \
            src/enc/partitioning/partition_score_func.h \
            src/enc/partitioning/partition_score_func.cc \
            src/enc/partitioning/partition_score_func_area.h \
            src/enc/partitioning/partition_score_func_area.cc \
            src/enc/partitioning/partition_score_func_block.h \
            src/enc/partitioning/partition_score_func_block.cc \
            src/enc/partitioning/partition_score_func_multi.h \
            src/enc/partitioning/partition_score_func_multi.cc \
            src/enc/partitioning/partition_score_func_tile.h \
            src/enc/partitioning/partition_score_func_tile.cc \
            src/enc/partitioning/partitioner.h \
            src/enc/partitioning/partitioner.cc \
            src/enc/partitioning/partitioner_area.h \
            src/enc/partitioning/partitioner_area.cc \
            src/enc/partitioning/partitioner_exhaustive.h \
            src/enc/partitioning/partitioner_exhaustive.cc \
            src/enc/partitioning/partitioner_multi.h \
            src/enc/partitioning/partitioner_multi.cc \
            src/enc/partitioning/partitioner_split.h \
            src/enc/partitioning/partitioner_split.cc \
            src/enc/partitioning/partitioner_split_recurse.h \
            src/enc/partitioning/partitioner_split_recurse.cc \
            src/enc/partitioning/partitioner_top_left.h \
            src/enc/partitioning/partitioner_top_left.cc \
            src/enc/syntax_enc.cc \
            src/enc/preview/preview_analysis.cc \
            src/enc/preview/preview_color.cc \
            src/enc/preview/preview_config.cc \
            src/enc/preview/preview_enc.h \
            src/enc/preview/preview_enc.cc \
            src/enc/preview/preview_opt.cc \
            src/enc/residuals_enc.cc \
            src/enc/residuals_enc_aom.h \
            src/enc/residuals_enc_aom.cc \
            src/enc/screen_content/screen_enc.h \
            src/enc/screen_content/screen_enc.cc \
            src/enc/segment_enc.cc \
            src/enc/symbols_enc.h \
            src/enc/symbols_enc.cc \
            src/enc/tile_enc.h \
            src/enc/tile_enc.cc \
            src/enc/trellis.h \
            src/enc/trellis.cc \
            src/enc/wp2_enc_i.h \
            src/enc/writer_enc.cc \
            src/utils/ans.h \
            src/utils/ans.cc \
            src/utils/ans_enc.h \
            src/utils/ans_enc.cc \
            src/utils/ans_cost_table.cc \
            src/utils/ans_utils.h \
            src/utils/ans_utils.cc \
            src/utils/context_switch.h \
            src/utils/context_switch.cc \
            src/utils/csp.h \
            src/utils/csp.cc \
            src/utils/data.cc \
            src/utils/data_source.h \
            src/utils/data_source.cc \
            src/utils/data_source_context.h \
            src/utils/data_source_context.cc \
            src/utils/data_source_stream.h \
            src/utils/data_source_stream.cc \
            src/utils/argb_buffer.cc \
            src/utils/front_mgr.h \
            src/utils/front_mgr.cc \
            src/utils/hash.h \
            src/utils/memory.cc \
            src/utils/orientation.cc \
            src/utils/orientation.h \
            src/utils/plane.cc \
            src/utils/plane.h \
            src/utils/quantizer.cc \
            src/utils/quantizer.h \
            src/utils/random.h \
            src/utils/split_iterator.h \
            src/utils/split_iterator.cc \
            src/utils/stats.h \
            src/utils/status.cc \
            src/utils/thread_utils.cc \
            src/utils/thread_utils.h \
            src/utils/utils.cc \
            src/utils/utils.h \
            src/utils/vector.h \
            src/utils/wiener.h \
            src/utils/wiener.cc \
            src/wp2/base.h \
            src/wp2/debug.h \
            src/wp2/decode.h \
            src/wp2/encode.h \
            src/wp2/format_constants.h \
            src/dec/lossless/group4_dec.cc \
            src/dec/lossless/losslessi_dec.cc \
            src/dec/lossless/losslessi_dec.h \
            src/dsp/lossless/decl_dsp.h \
            src/dsp/lossless/decl_dsp.cc \
            src/dsp/lossless/dspl.h \
            src/dsp/lossless/dspl.cc \
            src/dsp/lossless/encl_dsp.h \
            src/dsp/lossless/encl_dsp.cc \
            src/enc/lossless/analysisl_enc.cc \
            src/enc/lossless/histogram_enc.h \
            src/enc/lossless/predictorl_enc.cc \
            src/enc/lossless/losslessi_enc.cc \
            src/enc/symbols_enc.h \
            src/enc/symbols_enc.cc \
            src/enc/lossless/losslessi_enc.h \
            src/enc/lossless/backward_references_cost_enc.cc \
            src/enc/lossless/backward_references_enc.cc \
            src/enc/lossless/backward_references_enc.h \
            src/enc/lossless/group4_enc.cc \
            src/enc/lossless/histogram_enc.cc \
            src/enc/lossless/palette.h \
            src/enc/lossless/palette.cc \
            swig/examples/pycwp2.py \
            swig/examples/pydwp2.py \
            swig/examples/pyvwp2.py \
            swig/CMakeLists.txt \
            swig/libwebp2.i \
            swig/README.md \
            tests/fuzz/common_tags.dict \
            tests/fuzz/common_wp2_tags.dict \
            tests/fuzz/fuzz_anim.cc \
            tests/fuzz/fuzz_anim_and_dec_config.cc \
            tests/fuzz/fuzz_anim_and_enc_config.cc \
            tests/fuzz/fuzz_anim_enc_config.cc \
            tests/fuzz/fuzz_image.cc \
            tests/fuzz/fuzz_image_and_dec_config.cc \
            tests/fuzz/fuzz_image_and_enc_config.cc \
            tests/fuzz/fuzz_image_enc_config.cc \
            tests/fuzz/fuzz_imageio_anim_dec.cc \
            tests/fuzz/fuzz_imageio_image_dec_enc.cc \
            tests/fuzz/fuzz_imageio_image_enc.cc \
            tests/fuzz/fuzz_utils.cc \
            tests/fuzz/fuzz_utils.h \
            tests/fuzz/fuzz_utils_temp_file.h \
            tests/fuzz/repro_fuzz.cc \
            tests/include/helpers.h \
            tests/include/helpers.cc \
            tests/include/helpers_filter.h \
            tests/include/helpers_filter.cc \
            tests/include/helpers_incr.h \
            tests/include/helpers_incr.cc \
            tests/CMakeLists.txt \
            tests/ansz.cc \
            tests/test_alloc.cc \
            tests/test_alpha_filter.cc \
            tests/test_analysis.cc \
            tests/test_anim.cc \
            tests/test_anim_frame.cc \
            tests/test_anim_incr.cc \
            tests/test_anim_rectangle.cc \
            tests/test_anim_skip.cc \
            tests/test_ans.cc \
            tests/test_ansz.sh \
            tests/test_aom_residuals.cc \
            tests/test_api.cc \
            tests/test_assert.cc \
            tests/test_assert.sh \
            tests/test_average_color.cc \
            tests/test_block_size.cc \
            tests/test_buffer.cc \
            tests/test_bypass.cc \
            tests/test_cmd.sh \
            tests/test_trellis.cc \
            tests/lossless/test_color_cache.cc \
            tests/lossless/test_histogram.cc \
            tests/lossless/test_group4.cc \
            tests/lossless/test_palette.cc \
            tests/test_av1.cc \
            tests/test_coded_block.cc \
            tests/test_color_precision.cc \
            tests/test_context.cc \
            tests/test_csp.cc \
            tests/test_custom_csp.cc \
            tests/test_custom_csp_dec.cc \
            tests/test_custom_csp_enc.cc \
            tests/test_custom_csp_enc_dec.cc \
            tests/test_dsp.cc \
            tests/test_dsp_speed.cc \
            tests/test_data.cc \
            tests/test_data_source.cc \
            tests/test_data_source_stream.cc \
            tests/test_deblocking_filter.cc \
            tests/test_dec.cc \
            tests/test_decoder_api.cc \
            tests/test_diffusion.cc \
            tests/test_directional_filter.cc \
            tests/test_distortion.cc \
            tests/test_enc.sh \
            tests/test_enc_anim.sh \
            tests/test_enc_dir.sh \
            tests/test_enc_ewp2.sh \
            tests/test_example_utils.cc \
            tests/test_extras.cc \
            tests/test_formats.cc \
            tests/test_front_mgr.cc \
            tests/test_header.cc \
            tests/test_global_params.cc \
            tests/test_icc.sh \
            tests/test_imageio.cc \
            tests/test_imageio_anim.cc \
            tests/test_imageio_conversion.cc \
            tests/test_imageio_util.cc \
            tests/test_incr_dec.cc \
            tests/test_intertile_filter.cc \
            tests/test_io.sh \
            tests/test_io_pnm.sh \
            tests/test_io_ewp2.sh \
            tests/test_math.cc \
            tests/test_malloc_fail_at.cc \
            tests/test_memory.sh \
            tests/test_metadata.cc \
            tests/test_debug_options.cc \
            tests/test_optim.cc \
            tests/test_orientation.cc \
            tests/test_partitions.cc \
            tests/test_partition_score_func.cc \
            tests/test_pessimization.cc \
            tests/test_plane.cc \
            tests/test_pessimization.cc \
            tests/test_predictor.cc \
            tests/test_preview_api.cc \
            tests/test_progress.cc \
            tests/test_score.cc \
            tests/test_screen.cc \
            tests/test_quant_mtx.cc \
            tests/test_restoration_filter.cc \
            tests/test_residuals.cc \
            tests/test_segment.cc \
            tests/test_simple_enc_dec.cc \
            tests/test_simple_encode.cc \
            tests/test_slow_enc.cc \
            tests/test_slow_lossless_enc.cc \
            tests/test_split_iterator.cc \
            tests/test_stuttering_server.sh \
            tests/test_efforts.cc \
            tests/test_symbols.cc \
            tests/test_swig_examples.sh \
            tests/test_swig_python.py \
            tests/test_syntax_writer.cc \
            tests/test_thread.cc \
            tests/test_tile_shapes.cc \
            tests/test_transf.cc \
            tests/test_utils.cc \
            tests/test_uv_mode.cc \
            tests/test_varint.cc \
            tests/test_vector.cc \
            tests/test_version.cc \
            tests/testdata \
            tests/testdata/metadata/ICC_524_type1.bin \
            tests/testdata/metadata/ICC_3144_type2.bin \
            tests/testdata/metadata/ICC_3144_type3.bin \
            tests/testdata/alpha_ramp.lossy.webp \
            tests/testdata/alpha_ramp.pam \
            tests/testdata/alpha_ramp.png \
            tests/testdata/alpha_ramp.tiff \
            tests/testdata/alpha_ramp.webp \
            tests/testdata/animation0.gif \
            tests/testdata/animation0.webp \
            tests/testdata/animation1.webp \
            tests/testdata/source0.pgm \
            tests/testdata/source0.ppm \
            tests/testdata/source1.itl.png \
            tests/testdata/source1.png \
            tests/testdata/source1_1x1.png \
            tests/testdata/source1_1x48.png \
            tests/testdata/source1_32x32.png \
            tests/testdata/source1_4x4.png \
            tests/testdata/source1_64x1.png \
            tests/testdata/source1_64x48.png \
            tests/testdata/source2.tiff \
            tests/testdata/source3.jpg \
            tests/testdata/source4.ll.webp \
            tests/testdata/source4.webp \
            tests/testdata/specific \
            tests/testdata/specific/bug449_1001.pam \
            tests/testdata/specific/bug449_1001.pgm \
            tests/testdata/specific/bug449_1001.ppm \
            tests/testdata/specific/bug449_1001_g.pam \
            tests/testdata/specific/bug449_255.pam \
            tests/testdata/specific/bug449_255.pgm \
            tests/testdata/specific/bug449_255.ppm \
            tests/testdata/specific/bug449_255_g.pam \
            tests/testdata/specific/bug449_37.pam \
            tests/testdata/specific/bug449_37.pgm \
            tests/testdata/specific/bug449_37.ppm \
            tests/testdata/specific/bug449_37_g.pam \
            tests/testdata/specific/bug449_65535.pam \
            tests/testdata/specific/bug449_65535.pgm \
            tests/testdata/specific/bug449_65535.ppm \
            tests/testdata/specific/bug449_65535_g.pam \
            tests/testdata/specific/bug449_ref.png \
            tests/testdata/specific/bug449_ref_gray.png \
            tests/testdata/specific/bug449_ref_gray_alpha.png \
            tests/testdata/specific/bug449_ref_noalpha.png \
            tests/tools/stuttering_http_server.py \
            wp2_js/index.html \
            wp2_js/index_wasm.html \
            wp2_js/test_wp2_js.wp2 \
            wp2_js/test_wp2_wasm.wp2 \
            wp2_js/preview.html \
            Android.mk \
            Application.mk \
            examples/Android.mk \

pak: clean
ifeq ($(strip $(shell uname)), Darwin)
	COPYFILE_DISABLE=1 tar -s ,.*,wp2-$(VERSION)\/~,p -czf $(ARCHIVE_FILE) \
  $(DIST_FILES)
else
	tar --transform="s#^#wp2-$(VERSION)/#" -czf $(ARCHIVE_FILE) $(DIST_FILES)
endif
	@echo "GENERATED ARCHIVE: $(ARCHIVE_FILE)"

.PHONY: all clean dist ex pak
.SUFFIXES:
