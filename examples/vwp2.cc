// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// -----------------------------------------------------------------------------
//
// WP2 visualization tool
// usage: vwp2 input.png [-q quality][...]
//
// Author: Skal (pascal.massimino@gmail.com)

#if defined(__unix__) || defined(__CYGWIN__)
#define _POSIX_C_SOURCE 200112L  // for setenv
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#ifdef HAVE_CONFIG_H
#include "src/wp2/config.h"
#endif

#include "examples/example_utils.h"
#include "imageio/image_dec.h"
#include "imageio/image_enc.h"
#include "imageio/imageio_util.h"
#include "src/common/color_precision.h"
#include "src/common/constants.h"
#include "src/common/lossy/block_size.h"
#include "src/dec/wp2_dec_i.h"
#include "src/dsp/dsp.h"
#include "src/dsp/math.h"
#include "src/enc/wp2_enc_i.h"
#include "src/utils/plane.h"
#include "src/utils/random.h"
#include "src/utils/utils.h"
#include "src/wp2/decode.h"
#include "src/wp2/encode.h"

#if defined(WP2_HAVE_AOM)
#include "extras/aom_utils.h"
#endif

#if defined(WP2_HAVE_OPENGL) && !defined(WP2_REDUCE_BINARY_SIZE)
#if defined(WP2_HAVE_FREEGLUT)

#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>

#else

#define GL_SILENCE_DEPRECATION  // to avoid deprecation warnings on MacOS
#if defined(HAVE_GLUT_GLUT_H)
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif
#endif  // !WP2_HAVE_FREEGLUT

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif

namespace WP2 {
namespace {

// Scroll wheel events also sends mouse click events as button 3 and 4 (which
// don't seem to have a GLUT constants).
constexpr int kScrollWheelUp = 3;
constexpr int kScrollWheelDown = 4;

//------------------------------------------------------------------------------
// Visual Debug: set DecoderInfo::visual_debug to get extra visual data into
// DecoderInfo::debug_output during decoding.

struct VDToken {
  VDToken(const char* const token_name,  // NOLINT - not explicit
          std::vector<VDToken>&& next_tokens = std::vector<VDToken>())
      : name(token_name),
        next(std::move(next_tokens)),
        longest_next_name_size(
            next.empty() ? 0
                         : (int32_t)std::max_element(
                               next.begin(), next.end(),
                               [](const VDToken& lhs, const VDToken& rhs) {
                                 return lhs.name.size() < rhs.name.size();
                               })->name.size()) {}

  const std::string name;
  const std::vector<VDToken> next;       // Direct children.
  const int32_t longest_next_name_size;  // Longest name size amongst 'next'.
};

std::vector<VDToken> GetVDTokensCfl() {
  return {"best-prediction", "best-residuals", "best-slope", "best-intercept"};
}

std::vector<VDToken> GetVDTokensChannelCommon() {
  return {{"compressed"},
          {"prediction", {{"raw"}, {"modes", {"long", "short"}}}},
          {"residuals"},
          {"encoder", {"original-residuals", "prediction-scores"}}};
}

std::vector<VDToken> GetVDTokensAlpha() {
  std::vector<VDToken> r = GetVDTokensChannelCommon();
  r.emplace_back<VDToken>({"is_lossy"});
  r.emplace_back<VDToken>({"lossy"});
  r.emplace_back<VDToken>({"lossless"});
  return r;
}

std::vector<VDToken> GetVDTokensUV() {
  std::vector<VDToken> r = GetVDTokensChannelCommon();
  r.emplace_back("coeff-method");
  r.emplace_back<VDToken>(
      {"chroma-from-luma", {"prediction", "slope", "intercept"}});
  return r;
}

std::vector<VDToken> GetVDTokensY() {
  std::vector<VDToken> r = GetVDTokensChannelCommon();
  r.emplace_back("coeff-method");
  return r;
}

// Tree of possible 'visual_debug' paths.
const VDToken kVisualDebug  // NOLINT - non-trivially destructible global var
    {"",
     {{"decompressed"},  // No vdebug.
      {"blocks",
       {{"partition", {"split-tf", "blocks-only"}},
        "segment-ids",
        {"quantization", {"y", "u", "v", "a"}},
        "is420",
        {"encoder", {"is420-scores", "lambda-mult"}}}},
      {"transform", {"xy", "x", "y"}},
      {"y", GetVDTokensY()},
      {"u", GetVDTokensUV()},
      {"v", GetVDTokensUV()},
      {"a", GetVDTokensAlpha()},
      {"filters",
       {{"filter-block-map", {{"bpp"}, {"res", {"y", "u", "v", "a"}}}},
        {"deblocking-filter",
         {{"diff", {"yuv", "a"}},
          {"strength",
           {{"horizontal", {"y", "u", "v", "a"}},
            {"vertical", {"y", "u", "v", "a"}}}}}},
        {"directional-filter", {"diff", "strength", "direction", "variance"}},
        {"restoration-filter", {"diff", "strength"}},
        {"intertile-filter",
         {"diff",
          {"strength",
           {{"horizontal", {"y", "u", "v", "a"}},
            {"vertical", {"y", "u", "v", "a"}}}}}},
        {"grain-filter", {"diff"}},
        {"alpha-filter", {"diff"}}}},
      {"encoder",
       {{"partition",
         {"method",
          {"pass",
           {"all", "0",  "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8", "9",
            "10",  "11", "12", "13", "14", "15", "16", "17", "18", "19"}},
          "order",
          "score",
          {"multi",
           {{"raw-quant-dct", {"32x32", "16x16", "8x8", "4x4"}},
            {"luma-alpha-gradient", {"32x32", "16x16", "8x8", "4x4"}},
            {"narrow-std-dev", {"32x32", "16x16", "8x8", "4x4"}},
            {"quant-dct", {"32x32", "16x16", "8x8", "4x4"}},
            {"direction", {"32x32", "16x16", "8x8", "4x4"}},
            {"analysis", {"spread-5x5", "direction-4x4"}}}}}},
        {"segmentation", {"variance", "score", "id", "blocks"}},
        {"error-diffusion",
         {{"u", {"propagated-error", "new-error"}},
          {"v", {"propagated-error", "new-error"}}}},
        {"chroma-from-luma",
         {{"u", GetVDTokensCfl()},
          {"v", GetVDTokensCfl()},
          {"a", GetVDTokensCfl()}}}}},
      {"compressed", {"a", "r", "g", "b"}},
      // Requires the reference image to be copied to DecoderInfo::debug_output.
      {"original",
       {"diff",
        "a",
        "r",
        "g",
        "b",
        "y",
        "u",
        "v",
        {"histogram", {"y", "u", "v"}}}},
      {"lossless",
       {{"symbols", "clusters", {"transformed", {"regular", "force-opaque"}}}}},
      // Requires WP2_BITTRACE to be defined during the compilation.
      {"bits-per-pixel", {{"overall"}, {"coeffs", {"y", "u", "v"}}}},
      {"error-map",
       {"PSNR",
        "SSIM",
        "MSSSIM",
        "LSIM",
        {"diff-with-alt", {"PSNR", "SSIM", "MSSSIM", "LSIM"}}}}}};

// Verifies that 'str' starts with 'token' and that the ending or the next
// character (if any) is the separator '/'.
bool StrStartsWithToken(const std::string& str, const std::string& token) {
  return (std::strncmp(str.c_str(), token.c_str(), token.size()) == 0 &&
          (str.size() <= token.size() || str[token.size()] == '/' ||
           (!token.empty() && token.back() == '/')));
}

// Recursively retrieves the paths to all leaves.
void GetAllLeavesPaths(const VDToken& node, const std::string& current_path,
                       std::vector<std::string>* const paths) {
  for (const VDToken& next : node.next) {
    if (next.next.empty()) {
      paths->push_back(current_path + next.name);
    } else {
      GetAllLeavesPaths(next, current_path + next.name + '/', paths);
    }
  }
}

// Returns the path to the first leaf matching 'mask' or to an adjacent one
// depending on 'leaf_offset'.
std::string GetLeaf(const std::string& mask, int32_t leaf_offset = 0) {
  // Printing all paths to an array then parsing it is not great but simple.
  std::vector<std::string> leaves_paths;
  GetAllLeavesPaths(kVisualDebug, "", &leaves_paths);

  for (uint32_t i = 0; i < leaves_paths.size(); ++i) {
    if (StrStartsWithToken(leaves_paths[i], mask)) {
      return leaves_paths[((int32_t)(i + leaves_paths.size()) + leaf_offset) %
                          leaves_paths.size()];
    }
  }
  return "";
}

//------------------------------------------------------------------------------

constexpr uint32_t kNumSparks = 1000;

// Unfortunate global variables. Gathered into a struct for comfort.
class Params {
 public:
  Params() : current_file_(~0u) {
    enc_config_.thread_level = 64;
    enc_config_.info = &einfo_;
    dec_config_.thread_level = 64;
    dec_config_.info = &dinfo_;
#if defined(WP2_BITTRACE)
    einfo_.store_blocks = true;  // Preemptively to avoid (slow) reencoding.
#endif
  }
  WP2_NO_DISCARD bool SetCurrentFile(size_t file_number);
  WP2_NO_DISCARD bool SetBitstream(const char* const file_name);
  WP2_NO_DISCARD bool SetAltFile(const char* const file_name);
  WP2_NO_DISCARD bool SetAltImage();
  WP2_NO_DISCARD bool DumpCurrentCanvas(const char* const file_path);
  void ShiftAlt();   // move alt1 to alt2
  void SetQuality(float incr);
  WP2_NO_DISCARD bool EncodeImage();
  WP2_NO_DISCARD bool DecodeOutput();
  WP2_NO_DISCARD bool EncodeAndDecode();
  WP2_NO_DISCARD bool DecodeAndSwap(ArgbBuffer* const buffer);
  WP2_NO_DISCARD bool GetOriginal();
  WP2_NO_DISCARD bool GetCompressed();
  WP2_NO_DISCARD bool ComputeYUVDistortion();

  const ArgbBuffer& GetBuffer() const;
  WP2_NO_DISCARD bool GetCurrentCanvas(ArgbBuffer* const buffer);

  WP2_NO_DISCARD bool EncodeWebP(bool match_size);
  float webp_distortion_ = 0.f;
  uint32_t webp_size_ = 0;
  WP2_NO_DISCARD bool EncodeAV1(bool copy_partition, float av1_quality,
                                size_t* const av1_file_size = nullptr);
  WP2_NO_DISCARD bool EncodeAV1ToMatch(bool copy_partition,
                                       size_t target_file_size);

  WP2Status status_ = WP2_STATUS_OK;

  enum Show {
    kDebug,
    kOriginal,
    kCompressed,
    kPreview,
    kPreviewColor,
    kAlt1, kAlt2,
    kWebP,
    kInfo,
    kHelp,
    kMenu
  };
  Show show_ = kDebug;
  int mouse_x_ = 20, mouse_y_ = 20;  // In pixels, starting from top left.
  int mouse_x_down_ = 20, mouse_y_down_ = 20;  // Position at last GLUT_DOWN.
  bool moved_since_last_down_ = false;
  std::string message_;
  int message_end_time_;
  bool show_interface_ = true;

  // Converts from pixel sizes to relative sizes as used by OpenGL.
  float ToRelativeX(int32_t px) const { return px * 2.f / viewport_width_; }
  float ToRelativeY(int32_t px) const { return px * 2.f / viewport_height_; }
  // Converts from pixel positions to absolute positions as used by OpenGL.
  float ToAbsoluteX(int32_t px) const { return ToRelativeX(px) - 1.f; }
  float ToAbsoluteY(int32_t px) const { return 1.f - ToRelativeY(px); }
  // Converts from viewport pixel position to image pixel coordinates.
  uint32_t ToImageX(int viewport_x) const;
  uint32_t ToImageY(int viewport_y) const;
  // Same as glRectf() but with 'rect' in pixels, top-left being (0, 0).
  void glRect(const Rectangle& rect, uint32_t thickness = 1) const;

  float sparks_x_[kNumSparks] = {0};
  float sparks_y_[kNumSparks] = {0};
  uint32_t sparks_index_ = 0;
  bool display_sparks_ = false;

  void DisplaySparks();

  enum {
    kDisplayNone,
    kDisplayHeader,
    kDisplayYCoeffs,
    kDisplayUCoeffs,
    kDisplayVCoeffs,
    kDisplayACoeffs,
    kDisplayPredModes,
    kDisplayGrid,
    kDisplayNum
  } display_block_ = kDisplayNone;
  void DisplayBlockInfo(std::vector<std::string>* const msg);
  // side-func to help DisplayBlockInfo()
  void PrintBlockHeader(const BlockInfo& b,
                        std::vector<std::string>* const msg) const;
  void PrintResiduals(const BlockInfo& b, const BlockInfo& b_enc,
                        std::vector<std::string>* const msg) const;
  void PrintPredModes(const BlockInfo& b,
                      std::vector<std::string>* const msg) const;

  const char* partition_file_path_ = "/tmp/partition.txt";
  const char* dump_wp2_path_ = "/tmp/dump.wp2";
  const char* dump_png_path_ = "/tmp/dump.png";
  // If not empty, is the current block being drawn with the mouse.
  Rectangle forcing_block_;
  // Same as EncoderInfo::force_partition but not yet encoded.
  std::vector<Rectangle> force_partition_;
  std::vector<EncoderInfo::ForcedSegment> force_segment_;
  std::vector<EncoderInfo::ForcedPredictor> force_predictor_;
  std::vector<EncoderInfo::ForcedTransform> force_transform_;
  void ClearForcedElements();
  bool IsDebugViewVisible() const;
  bool IsPartitionVisible() const;
  bool IsSegmentVisible() const;
  bool IsPredictorVisible() const;
  bool IsTransformVisible() const;
  void DisplayPartition() const;
  void DisplayForcedSegments() const;
  void DisplayForcedPredictors() const;
  void DisplayForcedTransforms() const;
#if defined(WP2_BITTRACE)
  // Displays a sort of bar graph at the bottom of the screen showing the
  // relative size of different syntax elements.
  // 'traces' contains the size in bits taken by each (group of) symbols.
  void DisplayBitTraces() const;
#endif
  uint32_t getLineThickness() const;
  bool ReadAndConvertPartition(bool read_only = true);
  // Checks if either the encoding or decoding visual debug config match the
  // given token.
  bool VDMatchEncDec(const char* token) const;

  bool IsHamburgerMenuVisible() const;
  bool SetMenuSelection(const std::string& selection);
  void DisplayMenu();
  bool DisplaySubMenu(int32_t x, int32_t y, const VDToken& prefix,
                      std::string* const current_selection);

  // Adds a temporary 'message' on screen.
  void SetMessage(const std::string& message, bool even_if_show_info = false,
                  int duration_ms = 2000);
  // Same as above with different default arguments.
  void AddInfo(const std::string& message);
  void SetError(const std::string& message);

  // Top-level window refresh.
  void PrintInfo();

  void PrintMessages(const std::vector<std::string>& msg, bool print_lower,
                     const float text_color[4], const float bg_color[4],
                     bool small = false, bool outline = false);
  std::vector<std::string> GetHelp();

  // Reshapes the window based on current image size and zoom level.
  void ReshapeWindow();
  void ApplyZoomLevel();
  void UpdateViewZoomLevel(int incr);

  // Event handling.
  void HandleKey(unsigned char key, int pos_x, int pos_y);
  void HandleKeyUp(unsigned char key, int pos_x, int pos_y);
  void HandleMouseMove(int x, int y, bool button_pressed);

  // Functions below return true if the click event was handled, false if
  // nothing was done.
  bool HandleSegmentClick(int button, int state, uint32_t img_x_down,
                          uint32_t img_y_down, uint32_t img_x, uint32_t img_y,
                          int incr = 1);
  bool HandlePredictorClick(int button, int state, uint32_t img_x_down,
                            uint32_t img_y_down, uint32_t img_x, uint32_t img_y,
                            int incr);
  bool HandleTransformClick(int button, int state, uint32_t img_x_down,
                            uint32_t img_y_down, uint32_t img_x, uint32_t img_y,
                            int incr);
  bool HandlePartitionClick(int button, int state, uint32_t img_x_down,
                            uint32_t img_y_down, uint32_t img_x,
                            uint32_t img_y);
  bool HandleMenuClick(int button, int state, uint32_t img_x_down,
                       uint32_t img_y_down, uint32_t img_x, uint32_t img_y);

  void HandleMouseClick(int button, int state, int x, int y);
  void HandleMouseWheel(bool is_up, int x, int y);
  void HandleDisplay();

  enum Background {
    kCheckerboard,
    kWhite,
    kBlack,
    kPink,
    kBackgroundNum,
  } background_ = kCheckerboard;

  // Debug environment value. Can be retrieved with 'atoi(getenv("WP2DBG"))'.
  uint32_t wp2dbg_value_ = 0;

  size_t current_file_;
  std::vector<std::string> files_;
  std::string input_;   // input file
  std::string output_;  // currently encoded file

  float riskiness_ = 0.f;

  uint32_t enc_duration_ms_;  // encoding time in ms
  uint32_t dec_duration_ms_;  // decoding time in ms

  float quality_ = 75.;
  float alpha_quality_ = 100.;
  EncoderConfig enc_config_;  // encoding parameters
  DecoderConfig dec_config_;
  std::string menu_selection_ = "decompressed";  // hovered menu selection
  std::string menu_selection_prev_ = "decompressed";  // last menu selected
  std::string visual_debug_ = "decompressed";    // displayed debug output

  bool view_only_ = false;
  bool use_premultiplied_ = false;  // original samples in ARGB
  ArgbBuffer in_ = ArgbBuffer(use_premultiplied_ ? WP2_Argb_32 : WP2_ARGB_32);
  ArgbBuffer in_yuv_;  // original alpha, Y, U or V samples
  // image shown on spacebar (can be a view on 'in' or 'in_yuv')
  ArgbBuffer original_;
  ArgbBuffer out_;      // recompressed samples
  ArgbBuffer out_yuv_;  // recompressed samples
  ArgbBuffer preview_;
  Argb32b preview_color_;
  ArgbBuffer webp_;  // result of WebP compression
  // alternate comparison pictures & messages
  ArgbBuffer alt1_, alt2_;
  std::string alt1_msg_, alt2_msg_;

  EncoderInfo einfo_;
  DecoderInfo dinfo_;
  std::string original_selection_info_;
  MetricType distortion_metric_ = PSNR;
  float distortion_[5] = {0.f}, yuv_distortion_[3] = {0.f};
  int bit_trace_ = 0;
  uint32_t bit_trace_level_ = 0;
  uint32_t visual_bit_trace_level_ = 3;

  uint32_t width_, height_;   // Pixel with/height of the image.
  Rectangle view_rect_;       // Part of the image being shown.
  Rectangle view_rect_down_;  // View rect when mouse was clicked. For dragging.
  int view_zoom_level_ = 0;   // Zoom level used to determine view_rect_.
  // Display zoom level: 1 image pixel = 2^zoom_level pixels shown.
  int zoom_level_ = 0;
  // Size of viewport which is view_rect_.width/height * 2^zoom_level
  uint32_t viewport_width_, viewport_height_;

  bool crop_ = false;
  Rectangle crop_area_;
  bool keep_metadata_ = false;
};

// Made accessible to plain C function callbacks of glut.
Params* global_params = nullptr;

bool Params::SetMenuSelection(const std::string& selection) {
  menu_selection_ = selection;
  assert(!menu_selection_.empty());
  visual_debug_ = menu_selection_;
  dinfo_.visual_debug = visual_debug_.c_str();
  einfo_.visual_debug = visual_debug_.c_str();

  if (!VDMatch(enc_config_, "encoder")) {
    einfo_.visual_debug = nullptr;
  }
  if (VDMatch(dec_config_, "decompressed") || VDMatch(dec_config_, "encoder")) {
    // "decompressed" is in fact the regular output.
    // "encoder" is reserved to EncoderConfig.
    dinfo_.visual_debug = nullptr;
  }
  if (VDMatch(enc_config_, "encoder")) {
    if (!EncodeAndDecode()) return false;
  } else {
    if (!DecodeOutput()) return false;
  }
  return true;
}

//------------------------------------------------------------------------------
// Drawing tools

// Displays 'text' starting at pixel (x, y) in OpenGL coordinates [-1:1].
void PrintString(float x, float y, const std::string& text,
                 bool small = false) {
  glRasterPos2f(x, y);
  void* const font = small ? GLUT_BITMAP_8_BY_13 : GLUT_BITMAP_9_BY_15;
  for (char c : text) glutBitmapCharacter(font, c);
}

uint32_t Params::ToImageX(int viewport_x) const {
  return view_rect_.x +
         DivRound((uint32_t)Clamp<int>(viewport_x, 0, viewport_width_ - 1) *
                      (view_rect_.width - 1),
                  viewport_width_ - 1);
}
uint32_t Params::ToImageY(int viewport_y) const {
  return view_rect_.y +
         DivRound((uint32_t)Clamp<int>(viewport_y, 0, viewport_height_ - 1) *
                      (view_rect_.height - 1),
                  viewport_height_ - 1);
}

// Floors to block-aligned coordinate.
uint32_t AlignWithBlocks(uint32_t coord) {
  return coord - coord % kMinBlockSizePix;
}

void Params::DisplaySparks() {
  if (!display_sparks_) return;

  static UniformIntDistribution random(/*seed=*/0u);

  sparks_x_[sparks_index_] =
      ToRelativeX(mouse_x_ + random.Get<int32_t>(-5, 5)) - 1.f;
  sparks_y_[sparks_index_] = -(ToRelativeY(mouse_y_) - 1.f);
  sparks_index_ = (sparks_index_ + 1) % kNumSparks;

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  for (uint32_t i = 0; i < kNumSparks; ++i) {
    if (sparks_x_[i] != 0 && sparks_y_[i] != 0) {
      sparks_y_[i] -= ToRelativeY(2);

      glColor4f(random.Get<int32_t>(100, 255) / 255.f,
                random.Get<int32_t>(100, 255) / 255.f,
                random.Get<int32_t>(100, 255) / 255.f, 1.f);
      glRectf(sparks_x_[i] - ToRelativeX(1), sparks_y_[i] - ToRelativeY(1),
              sparks_x_[i] + ToRelativeX(1), sparks_y_[i] + ToRelativeY(1));
      if (sparks_y_[i] < -1) {
        sparks_x_[i] = sparks_y_[i] = 0;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Hamburger menu

constexpr int32_t kMenuCharWidth = 9;
constexpr int32_t kMenuItemHeight = 20;
constexpr int32_t kMenuBorderWidth = 2;

// Renders a submenu at (x, y) with children of 'prefix'. Highlights the node
// matching the 'current_selection' or sets it to the hovered entry.
// Returns true if the mouse at 'mouse_x, mouse_y' is inside the submenu.
bool Params::DisplaySubMenu(int32_t x, int32_t y, const VDToken& prefix,
                            std::string* const current_selection) {
  const int32_t longest_name_size = prefix.longest_next_name_size;
  const int32_t submenu_width =
      (longest_name_size + 3) * kMenuCharWidth + 2 * kMenuBorderWidth;
  const int32_t submenu_height = prefix.next.size() * kMenuItemHeight;
  const int32_t left = x, right = x + submenu_width - 1;  // Inclusive.

  const bool mouse_is_nearby =
      (mouse_x_ >= left && mouse_x_ <= right + 50 && mouse_y_ + 150 >= y &&
       mouse_y_ <= y + submenu_height + 150);
  const bool mouse_in_column = (mouse_is_nearby && mouse_x_ <= right);
  const bool mouse_in_submenu =
      (mouse_in_column && mouse_y_ >= y && mouse_y_ <= y + submenu_height - 1);
  bool was_hovered = false;  // True if an element is hovered by the mouse.

  // Background.
  const float kMenuAlpha = .8;
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glColor4f(0.f, 0.f, 0.f, kMenuAlpha);
  glRectf(ToAbsoluteX(x), ToAbsoluteY(y),
          ToAbsoluteX(x + submenu_width), ToAbsoluteY(y + submenu_height));

  // For each branch in the current VisualDebug tree node.
  int32_t top = y;
  for (const VDToken& token : prefix.next) {
    const int32_t bottom = top + kMenuItemHeight - 1;  // Inclusive.

    // Set 'current_selection' if hovered by the mouse.
    const bool is_hovered =
        (mouse_in_submenu && mouse_y_ >= top && mouse_y_ <= bottom);
    if (is_hovered) {
      assert(!was_hovered);  // Check that only one item is hovered.
      *current_selection = token.name;
      if (!token.next.empty()) *current_selection += "/";
      was_hovered = true;
    }

    // Do not set as selected if another entry in submenu is hovered.
    bool is_selected =
        (is_hovered || (!mouse_in_column &&
                        StrStartsWithToken(*current_selection, token.name)));

    std::string displayed_name = token.name;
    if (!token.next.empty()) {
      displayed_name.append(longest_name_size - token.name.size(), ' ');
      displayed_name += " /";
    }
    if (is_hovered) {
      glColor4f(1.0f, 1.0f, 1.0f, kMenuAlpha);
    } else if (is_selected) {
      glColor4f(0.2f, 0.8f, 0.2f, kMenuAlpha);
    } else {
      glColor4f(0.7f, 0.7f, 0.7f, kMenuAlpha);
    }
    PrintString(ToAbsoluteX(left + kMenuBorderWidth), ToAbsoluteY(bottom - 5),
                displayed_name);

    // Display submenu if any and selected.
    if (is_selected && !token.next.empty()) {
      std::string suffix =
          current_selection->substr(/*pos*/ token.name.size() + 1);
      *current_selection =
          current_selection->substr(/*pos*/ 0, /*n*/ token.name.size() + 1);
      was_hovered |= DisplaySubMenu(right + 1, top, token, &suffix);
      *current_selection += suffix;
    }

    top += kMenuItemHeight;
  }

  if (mouse_in_submenu) assert(was_hovered);
  if (mouse_is_nearby && !was_hovered) *current_selection = "";
  return was_hovered || mouse_is_nearby;
}

// Renders the menu and selected or hovered submenus.
void Params::DisplayMenu() {
  const bool is_hamburger_hovered =
      (mouse_x_ >= 0 && mouse_x_ < kMenuItemHeight && mouse_y_ >= 0 &&
       mouse_y_ < kMenuItemHeight);

  // Show the hamburger menu.
  if (IsHamburgerMenuVisible()) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glColor4f(0.f, 0.f, 0.f, .8f);
    glRectf(-1, 1, -1 + ToRelativeX(kMenuItemHeight),
            1 - ToRelativeY(kMenuItemHeight));
    glColor4f(1.f, 1.f, 1.f, .8f);

    for (uint32_t i = 0; i < 3; ++i) {
      glRectf(-1 + ToRelativeX(2), 1 - ToRelativeY(i * 5 + 3),
              -1 + ToRelativeX(kMenuItemHeight - 2),
              1 - ToRelativeY(i * 5 + 5));
    }

    // Show menu items if hamburger menu is hovered.
    if (is_hamburger_hovered) show_ = kMenu;
  } else if (show_ == kMenu) {
    show_ = kDebug;
    menu_selection_ = visual_debug_;
  }

  // Show menu items.
  if (show_ == kMenu) {
    const bool menu_hovered =
        DisplaySubMenu(0, kMenuItemHeight, kVisualDebug, &menu_selection_);
    if (!menu_hovered && !is_hamburger_hovered) {
      show_ = kDebug;
      menu_selection_ = visual_debug_;
    }
  }
}

//------------------------------------------------------------------------------
// Messages

constexpr uint32_t kFontWidthPix = 9;

// Split 'str' by the line break delimiter into 'msg' entries.
void SplitIntoMessages(const std::string& str,
                       std::vector<std::string>* const msg) {
  std::string::size_type from = 0, to;
  while (from < str.size()) {
    if ((to = str.find('\n', from)) == std::string::npos) to = str.size();
    msg->emplace_back(str.substr(from, to - from));
    from = to + 1;
  }
  // Remove any empty line at the back.
  while (!msg->empty() && msg->back().empty()) msg->pop_back();
}

void Params::PrintMessages(const std::vector<std::string>& msg,
                           bool print_lower, const float text_color[4],
                           const float bg_color[4], bool small, bool outline) {
  if (msg.empty()) return;

  const float line_height = ToRelativeY(19);
  const float line_start =
      1.f - ToRelativeY(print_lower ? 25 : 5) - line_height;
  const float left_start = -1.f + ToRelativeX(10);

  // Background unless fully transparent
  if (bg_color[3] > 0) {
    const float y_top = line_start + line_height;
    const float x_left = left_start - ToRelativeX(5);
    float HY = msg.size() * line_height;
    float HX = 0;
    for (const std::string& str : msg) HX = std::max(HX, (float)str.size());
    HX *= ToRelativeX(kFontWidthPix);
    HX += ToRelativeX(10);
    HY += ToRelativeY(10);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glColor4f(bg_color[0], bg_color[1], bg_color[2], bg_color[3]);
    glRectf(x_left, y_top, x_left + HX, y_top - HY);
  }

  if (outline) {
    glColor4f(bg_color[0], bg_color[1], bg_color[2], text_color[3]);
    for (int dy = -1; dy <= 1; dy += 2) {
      for (int dx = -1; dx <= 1; dx += 2) {
        for (size_t i = 0; i < msg.size(); ++i) {
          const float position = line_start - i * line_height;
          PrintString(left_start + ToRelativeX(dx), position + ToRelativeY(dy),
                      msg[i], small);
        }
      }
    }
  }

  // Text
  glColor4f(text_color[0], text_color[1], text_color[2], text_color[3]);
  for (size_t i = 0; i < msg.size(); ++i) {
    const float position = line_start - i * line_height;
    PrintString(left_start, position, msg[i], small);
  }
}

#if defined(WP2_BITTRACE)
constexpr uint32_t kBitTraceMaxLevels = 10;

void Params::DisplayBitTraces() const {
  double total_size = 0;
  for (const auto& p : dinfo_.bit_traces) total_size += p.second.bits;

  std::map<std::string, float> traces;
  for (const auto& p : dinfo_.bit_traces) {
    uint32_t level = 0;
    size_t i = 0;
    while (level < visual_bit_trace_level_ && i != std::string::npos) {
      i = p.first.find("/", (i == 0) ? 0 : i + 1);
      traces[p.first.substr(0, i)] += p.second.bits;
      ++level;
    }
  }

  const float line_height = ToRelativeY(35);

  const float mouse_x = ToRelativeX(mouse_x_) - 1;
  const float mouse_y = -(ToRelativeY(mouse_y_) - 1);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  // First pass draws the bars, second pass draws the labels, third pass draws
  // hovered label.
  for (uint32_t pass = 0; pass < 3; ++pass) {
    std::map<std::string, float> left_pos;
    uint32_t index[kBitTraceMaxLevels] = {0};

    for (const auto& v : traces) {
      const float relative_size = v.second / total_size;
      const float width = relative_size * 2;
      const int level = std::count(v.first.begin(), v.first.end(), '/');
      std::string prefix = "";
      if (level > 0) {
        prefix = v.first.substr(0, v.first.find_last_of("/"));
      }

      const float bottom = -1 + line_height * level;
      const float top = bottom + line_height;
      const float left = left_pos[prefix] - 1;
      const float right = left + width;
      constexpr float kInfoAlpha = 0.8f;

      if (pass == 0) {
        // First pass : draw the bars.
        const GLfloat color = index[level] % 2;
        glColor4f(color, color, color, kInfoAlpha);
        glRectf(left, top, right, bottom);
        // Add a black line at the top.
        glColor4f(0.f, 0.f, 0.f, kInfoAlpha);
        glRectf(left, top, right, top - ToRelativeY(1));
      } else if (pass == 1) {
        // Second pass: draw the labels.
        glColor4f(1.f, 0.f, 0.f, kInfoAlpha);
        const float top_pos = bottom + ToRelativeY(10);

        std::string title = v.first;
        if (level > 0) {
          title = title.substr(title.find_last_of("/") + 1);
        }
        const uint32_t num_chars = std::max(
            0.f,
            std::floor((width - ToRelativeX(4)) / ToRelativeX(kFontWidthPix)));
        if (num_chars > 1) {
          if (num_chars < title.length()) {
            title = title.substr(0, num_chars - 1);
            title += ".";  // To show that it's truncated.
          }
          PrintString(left_pos[prefix] - 1 + ToRelativeX(2), top_pos, title);
        }
      } else if (pass == 2 && mouse_x > left && mouse_x < right &&
                 mouse_y < top && mouse_y > bottom) {
        // Third pass: draw hover label.
        std::string title = v.first;
        if (level > 0) {
          title = title.substr(title.find_last_of("/") + 1);
        }
        title += SPrintf(" (%.f B, %.2f%%)", title.c_str(), v.second / 8.f,
                         relative_size * 100);
        const float title_width = title.size() * ToRelativeX(kFontWidthPix);
        const float title_x =
            (mouse_x + title_width > 1) ? 1 - title_width : mouse_x;
        // Draw a black outline.
        glColor4f(0.f, 0.f, 0.f, 1.f);
        static constexpr int kBorderWidth = 2;
        for (int dy = -kBorderWidth; dy <= kBorderWidth; ++dy) {
          for (int dx = -kBorderWidth; dx <= kBorderWidth; ++dx) {
            PrintString(title_x + ToRelativeX(dx), mouse_y + ToRelativeY(dy),
                        title);
          }
        }
        glColor4f(1.f, 1.f, 0.f, 1.f);
        PrintString(title_x, mouse_y, title);
      }
      ++index[level];
      left_pos[v.first] = left_pos[prefix];
      left_pos[prefix] += width;
    }
  }
}
#endif

//------------------------------------------------------------------------------

std::vector<std::string> Params::GetHelp() {
  std::vector<std::string> m;
  // Adding a new shortcut? These are still available: j, p, u, y
  m.emplace_back("Keyboard shortcuts:");
  m.emplace_back(
      "  'i' ............... overlay file information ('I' for YUV disto)");
  m.emplace_back("  'v'/'V' ........... cycle through visual debugging");
  m.emplace_back("  'e'/'E' ........... shortcut for error map");
  m.emplace_back(
      "  space ............. show the original uncompressed picture");
  m.emplace_back("  tab ............... show the compressed picture");
  m.emplace_back("  alt+1 / alt+2 ..... show alternate picture(s)");
  m.emplace_back(
      "  'a' ............... saves current canvas as alternate picture");
  m.emplace_back(std::string("  'A' ............... saves current canvas to ") +
                 dump_png_path_);
  m.emplace_back(std::string("  alt+'a' ........... loads ") + dump_png_path_ +
                 std::string(" as alternate picture"));
  m.emplace_back(
      "  up/down ........... change the compression factor by +/- 1 units");
  m.emplace_back(
      "  left/right ........ change the compression factor by +/- 10 units");
  m.emplace_back("  ctrl + arrows ..... change alpha compression factor");
  m.emplace_back("  's' ............... change encoding effort parameter");
  m.emplace_back("  't' ............... toggle segment id mode");
  m.emplace_back(
      "  '1' ............... cycle colorspace type "
      "(YCoCg,YCbCr,Custom,YIQ)");
  m.emplace_back("  '2' ............... cycle UV-Mode (Adapt,420,Sharp,444)");
  m.emplace_back("  '3' ............... change tile size");
  m.emplace_back("  '4' ............... change the partition method");
  m.emplace_back("  '5' ............... change the partition set");
  m.emplace_back("  '6' ............... toggle block snapping");
  m.emplace_back("  '7' ............... increase number of segments");
  m.emplace_back("  '8' ............... increase SNS value");
  m.emplace_back("  '9' ............... increase error diffusion strength");
  m.emplace_back("  '0' ............... toggle perceptual tuning");
  m.emplace_back("  'm' ............... toggle use of random matrix");
#if !defined(_WIN32)
  m.emplace_back("  'x' ............... toggle the env variable WP2DBG");
#endif
  m.emplace_back("  'f'/'F'/alt+'f/F' . toggle filters");
  m.emplace_back("  'g'/'G' ........... decrease/reset grain amplitude");
  m.emplace_back("  'b' ............... display block side-info");
  m.emplace_back("  'k' ............... toggle metadata");
  m.emplace_back("  'o' ............... change background color (for alpha)");
  m.emplace_back("  '\\' ............... show the preview if available");
  m.emplace_back("  '|' ............... show the preview color if available");
  m.emplace_back("  'z'/'Z' ........... zoom in/out (resize window)");
  m.emplace_back("  'n'/'N' ........... change distortion metric");
#if defined(WP2_HAVE_WEBP)
  m.emplace_back("  'w'/'W' ........... show WebP equivalent in size / disto");
#endif
#if defined(WP2_HAVE_AOM)
  m.emplace_back("  'r' ............... show AV1 equivalent in quality factor");
  m.emplace_back("  alt+'r' ........... show AV1 with same file size (slow)");
#endif
#if defined(WP2_HAVE_AOM_DBG)
  m.emplace_back("  'R' ............... copy AV1 partition (luma transforms)");
#endif
  m.emplace_back("  '`' ............... jump to previously selected menu");
  m.emplace_back("  '+'/'_' ........... go to next/previous file");
  m.emplace_back(std::string("  'l'/'L' ........... "
                             "load or dump the current bitstream as ") +
                 dump_wp2_path_);
  m.emplace_back("  'h' ............... show this help message");
  m.emplace_back("  'H' ............... toggle interface display");
  m.emplace_back("  'q' / 'Q' / ESC ... quit");

  m.emplace_back("");
  m.emplace_back("Block layout edition:");
  m.emplace_back("  Editor is shown by bringing the \"blocks/partitioning\"");
  m.emplace_back("  menu ('v' key) or the \"display block\" mode ('b' key).");
  m.emplace_back("  Force blocks position and size by drawing rectangles with");
  m.emplace_back("  the right mouse button. Remove one with right or middle ");
  m.emplace_back("  click. Press 'c' to convert rectangles (dashed outline)");
  m.emplace_back("  into actual forced encoded blocks (magenta outline).");
  m.emplace_back("  Press 'd' to dump or 'D' to load the blocks in file at");
  m.emplace_back(std::string("  ") + partition_file_path_);
  m.emplace_back("Segment edition: In segment-ids view, right click to cycle");
  m.emplace_back("  through segments, then press 'c' to apply. ");
  m.emplace_back("  Middle click to remove.");
  m.emplace_back("Predictor edition: In prediction view, right click to cycle");
  m.emplace_back("  through predictors, then press 'c' to apply. Middle click");
  m.emplace_back("  to remove. A red square means predictor was ignored ");
  m.emplace_back("  because context is constant.");
  m.emplace_back("Transform edition: In transform view, right click to cycle");
  m.emplace_back("  through transform pairs, then press 'c' to apply.");
  m.emplace_back("  Middle click to remove.");
  return m;
}

//------------------------------------------------------------------------------
// Info

const char* const kCSPString[] = {"YCoCg", "YCbCr", "Custom", "YIQ"};
const char* const kUVModeString[] = {"UVAdapt", "UV420", "UV444", "UVAuto"};
const char* const kTileShapeString[] = {"128", "256", "512", "Wide", "Auto"};
const char* const kPartitionSetString[] = {
    "Small squares",  "Small rects", "All rects", "Thick rects",
    "Medium squares", "All squares", "Some rects"};
const char* const kOnOff[] = {"Off", "On"};
const char* const kSegmentModes[] = {"Auto", "Explicit", "Implicit"};

STATIC_ASSERT_ARRAY_SIZE(kCSPString, kNumCspTypes);
STATIC_ASSERT_ARRAY_SIZE(kUVModeString, EncoderConfig::NumUVMode);
STATIC_ASSERT_ARRAY_SIZE(kPartitionSetString, NUM_PARTITION_SETS);

void Params::PrintInfo() {
  DisplaySparks();
  std::vector<std::string> msg;
  DisplayBlockInfo(&msg);
  DisplayPartition();
  DisplayForcedSegments();
  DisplayForcedPredictors();
  DisplayForcedTransforms();
  DisplayMenu();

  if (show_ == kMenu ||
      (forcing_block_.width > 0 && forcing_block_.height > 0)) {
    // Do not print messages if something else is displayed instead.
    return;
  }

  bool print_lower = IsHamburgerMenuVisible();
  float text_color[4] = {.0f, .0f, .0f, .9f};
  float bg_color[4] = {.8f, .8f, .9f, .9f};
  bool small = false;

  if (display_block_ != kDisplayNone) {
    bg_color[0] = .8f;
    bg_color[1] = .7f;
    bg_color[2] = .7f;
    bg_color[3] = .8f;
    small = (display_block_ > kDisplayHeader);
  } else if (show_ == kHelp) {
    const std::vector<std::string> help = GetHelp();
    msg.insert(msg.end(), help.begin(), help.end());
  } else if (show_ == kOriginal) {
    msg.emplace_back("- O R I G I N A L -");
    if (!original_selection_info_.empty()) {
      SplitIntoMessages(original_selection_info_, &msg);
      print_lower = true;  // Match standard kDebug text formatting.
      small = true;
    }
  } else if (show_ == kCompressed) {
    msg.emplace_back("- Compressed -");
  } else if (show_ == kPreview) {
    msg.emplace_back("- Preview -");
  } else if (show_ == kPreviewColor) {
    msg.emplace_back("- Preview color -");
  } else if (show_ == kWebP) {
    msg.emplace_back(SPrintf("- WebP [size=%u %s=%.2f] -", webp_size_,
                             kMetricNames[distortion_metric_],
                             webp_distortion_));
    msg.emplace_back(SPrintf("[ref WP2: size=%u %s=%.2f]", output_.size(),
                             kMetricNames[distortion_metric_], distortion_[4]));
  } else if (show_ == kInfo) {
    bg_color[0] = .8f;
    bg_color[1] = .9f;
    bg_color[2] = .9f;
    bg_color[3] = .8f;
    small = true;

    msg.emplace_back(files_[current_file_]);

    msg.emplace_back(SPrintf("Dimension: %d x %d", width_, height_));
    if (viewport_width_ != width_ || viewport_height_ != height_) {
      msg.emplace_back(SPrintf(" (displayed as %d x %d)",
                               viewport_width_, viewport_height_));
    }

    if (!view_only_) {
      msg.emplace_back(SPrintf(
                       "Quality: %3.1f [effort=%d     sns=%5.1f diffusion=%d",
                       quality_, enc_config_.effort, enc_config_.sns,
                       enc_config_.error_diffusion));
      msg.emplace_back(
          SPrintf("              segments=%u/%d(%c) snap=%s]",
                  dinfo_.num_segments, enc_config_.segments,
                  kSegmentModes[(int)enc_config_.segment_id_mode][0],
                  kOnOff[enc_config_.partition_snapping]));
#if defined(WP2_BITTRACE)
      msg.emplace_back(SPrintf(" %lu blocks", einfo_.blocks.size()));
#endif

      if (in_.HasTransparency()) {
        msg.emplace_back(SPrintf("Alpha quality: %.1f (%s=%.2f)",
            alpha_quality_, kMetricNames[distortion_metric_],
            distortion_[0]));
      }
    } else {
      msg.emplace_back(" -- WARNING: Showing original WP2 image --");
    }

    msg.emplace_back(SPrintf("Size: %u [%.2f bpp] (enc %u ms + dec %u ms)",
        (uint32_t)output_.size(), 8.f * output_.size() / (width_ * height_),
        enc_duration_ms_, dec_duration_ms_));
    msg.emplace_back(SPrintf(" (%.1f%% of original)",
                             100.f * output_.size() / input_.size()));

    if (in_.metadata_.IsEmpty()) {
      assert(out_.metadata_.IsEmpty());
    } else {
      const Metadata& mi = in_.metadata_;
      const Metadata& mo = out_.metadata_;
      msg.emplace_back(SPrintf(   // 6 spaces to align with "Size: ".
          "      %s %u B of metadata (ICCP %u, XMP %u, EXIF %u)",
          keep_metadata_ ? "included" : "ignored",
          (uint32_t)(mi.iccp.size + mi.xmp.size + mi.exif.size),
          (uint32_t)mi.iccp.size, (uint32_t)mi.xmp.size,
          (uint32_t)mi.exif.size));
      if (!keep_metadata_) {
        assert(out_.metadata_.IsEmpty());
      } else if (mi.iccp.size != mo.iccp.size || mi.xmp.size != mo.xmp.size ||
                 mi.exif.size != mo.exif.size) {
        msg.emplace_back(SPrintf(
            " decoded as ICCP %u, XMP %u, EXIF %u",
            (uint32_t)mo.iccp.size, (uint32_t)mo.xmp.size,
            (uint32_t)mo.exif.size));
      }
    }

#if defined(WP2_ENC_DEC_MATCH)
    msg.emplace_back(SPrintf(   // 6 spaces to align with "Size: ".
             "      Warning: size is inflated by WP2_ENC_DEC_MATCH");
#endif
    msg.emplace_back(SPrintf("Filters: dblk=%s, drct=%s, rstr=%s, alpha=%s",
        kOnOff[dec_config_.enable_deblocking_filter],
        kOnOff[dec_config_.enable_directional_filter],
        kOnOff[dec_config_.enable_restoration_filter],
        kOnOff[enc_config_.enable_alpha_filter]));

    if (!view_only_) {
      msg.emplace_back(SPrintf("%s=%.2f  (a %.1f, r %.1f, g %.1f, b %.1f)",
          kMetricNames[distortion_metric_], distortion_[4], distortion_[0],
          distortion_[1], distortion_[2], distortion_[3]));
      if (yuv_distortion_[0] > 0.f) {  // Maybe it is unavailable.
        msg.emplace_back(SPrintf("  (y %.1f, u %.1f, v %.1f)",
            yuv_distortion_[0], yuv_distortion_[1], yuv_distortion_[2]));
      }

      msg.emplace_back(SPrintf("CSP: %s   UV: %s  Partition: %s, %s  Tile: %s",
          kCSPString[(int)enc_config_.csp_type],
          kUVModeString[(int)enc_config_.uv_mode],
          kPartitionMethodString[enc_config_.partition_method],
          kPartitionSetString[enc_config_.partition_set],
          kTileShapeString[enc_config_.tile_shape]));
      msg.emplace_back(SPrintf("Perceptual:%s  RndMtx: %s  Grain: %s Amp: %d",
          kOnOff[enc_config_.tune_perceptual],
          kOnOff[enc_config_.use_random_matrix],
          kOnOff[enc_config_.store_grain], dec_config_.grain_amplitude));
     }
#if defined(WP2_BITTRACE)
    DisplayBitTraces();
#endif  // WP2_BITTRACE
  } else if (show_ == kAlt1) {
    msg.emplace_back(alt1_msg_);
  } else if (show_ == kAlt2) {
    msg.emplace_back(alt2_msg_);
  } else {
    bg_color[0] = .9f;
    bg_color[1] = .9f;
    bg_color[2] = .9f;
    bg_color[3] = .8f;
    small = true;

    if (VDMatch(enc_config_, "")) {
      msg.emplace_back(einfo_.visual_debug);
      SplitIntoMessages(einfo_.selection_info, &msg);
    } else if (VDMatch(dec_config_, "")) {
      msg.emplace_back(dinfo_.visual_debug);
      SplitIntoMessages(dinfo_.selection_info, &msg);
    }
  }

  if (!message_.empty()) {
    msg.push_back(message_);
    if (glutGet(GLUT_ELAPSED_TIME) > message_end_time_) {
      message_.clear();
    }
  }

  PrintMessages(msg, print_lower, text_color, bg_color, small);
}

void Params::SetMessage(const std::string& message, bool even_if_show_info,
                        int duration_ms) {
  if (show_ != kInfo || even_if_show_info) {
    message_ = message;
    message_end_time_ = glutGet(GLUT_ELAPSED_TIME) + duration_ms;
  }
}

void Params::AddInfo(const std::string& message) {
  SetMessage(message, /*even_if_show_info=*/true, /*duration_ms=*/5000);
}

void Params::SetError(const std::string& message) {
  SetMessage(message, /*even_if_show_info=*/true,
             /*duration_ms=*/60000);

  // Reset outputs so that nothing misleading is displayed.
  for (ArgbBuffer* const buffer : {&out_, &dinfo_.debug_output}) {
    WP2_ASSERT_STATUS(buffer->Resize(in_.width(), in_.height()));
    buffer->Fill({0, 0, 0, 0});
  }
#if defined(WP2_BITTRACE)
  einfo_.blocks.clear();
  dinfo_.bit_traces.clear();
  dinfo_.blocks.clear();
#endif  // WP2_BITTRACE
  dinfo_.header_size = 0;
  dinfo_.selection_info.clear();
}

//------------------------------------------------------------------------------
// Timer callbacks

void DisplayLoop(int delay_ms) {
  glutTimerFunc((unsigned int)delay_ms, DisplayLoop, delay_ms);
  glutPostRedisplay();
}

//------------------------------------------------------------------------------
// error maps

WP2Status ComputeErrorMap(MetricType metric, const ArgbBuffer& original,
                          const ArgbBuffer& decoded,
                          ArgbBuffer* const error_map,
                          DecoderInfo* const info = nullptr) {
  WP2_CHECK_OK(decoded.width() == original.width() &&
               decoded.height() == original.height(), WP2_STATUS_BAD_DIMENSION);
  WP2_CHECK_STATUS(error_map->Resize(original.width(), original.height()));
  const double norm = (metric == PSNR) ? 5. : (metric == LSIM) ? 5. : 8.;
  for (uint32_t y = 0; y < error_map->height(); ++y) {
    uint8_t* const dst = error_map->GetRow8(y);
    for (uint32_t x = 0; x < error_map->width(); ++x) {
      float d;
      WP2_CHECK_STATUS(decoded.GetDistortion(original, x, y, metric, &d));
      const uint8_t v = (uint8_t)std::lround(Clamp(255. - d * norm, 0., 255.));
      dst[4 * x + 0] = 255;  // TODO(skal): handle alpha properly
      dst[4 * x + 1] = dst[4 * x + 2] = dst[4 * x + 3] = v;

      if (info != nullptr && info->selection.x == x && info->selection.y == y) {
        info->selection_info =
            SPrintf("%s at %u, %u: %f\n", kMetricNames[metric], x, y, d);
      }
    }
  }
  return WP2_STATUS_OK;
}

// Sets 'diff' to be a shadowed version of 'original' but with a greener or
// redder tint depending on 'error_a' or 'error_b' being the lowest.
void ComputeDiffPixel(const uint8_t original[4], uint8_t error_a,
                      uint8_t error_b, uint8_t diff[4]) {
  diff[0] = kAlphaMax;
  for (uint32_t i : {1, 2, 3}) {
    // Darkened/greyed out version of the original sample.
    diff[i] = RightShiftRound((63u << 1) + ((uint32_t)original[i] << 6), 7);
  }
  if (error_a > error_b) {
    diff[1] = (uint8_t)Clamp(diff[1] + (error_a - error_b) * 10u, 0u, 255u);
  } else if (error_b > error_a) {
    diff[2] = (uint8_t)Clamp(diff[2] + (error_b - error_a) * 10u, 0u, 255u);
  }
}

// Fills 'diff_map' with 'original' pixels that are greener if 'decoded' is
// closer than 'alt', redder otherwise.
WP2Status ComputeDiffMap(MetricType metric, const ArgbBuffer& original,
                         const ArgbBuffer& decoded, const ArgbBuffer& alt,
                         ArgbBuffer* const diff_map,
                         DecoderInfo* const info = nullptr) {
  if (info != nullptr) {
    float disto_dec[5] = {}, disto_alt[5] = {};
    WP2_CHECK_STATUS(decoded.GetDistortion(original, metric, disto_dec));
    WP2_CHECK_STATUS(alt.GetDistortion(original, metric, disto_alt));
    info->selection_info =
        SPrintf("%s of decoded: %5.2f (A %4.1f, R %4.1f, G %4.1f, B %4.1f)\n",
                kMetricNames[metric], disto_dec[4], disto_dec[0], disto_dec[1],
                disto_dec[2], disto_dec[3]);
    info->selection_info +=
        SPrintf("%s of alt:     %5.2f (A %4.1f, R %4.1f, G %4.1f, B %4.1f)\n",
                kMetricNames[metric], disto_alt[4], disto_alt[0], disto_alt[1],
                disto_alt[2], disto_alt[3]);
  }

  ArgbBuffer error_map_decoded, error_map_alt;
  WP2_CHECK_STATUS(
      ComputeErrorMap(metric, original, decoded, &error_map_decoded));
  WP2_CHECK_STATUS(ComputeErrorMap(metric, original, alt, &error_map_alt));
  WP2_CHECK_STATUS(diff_map->Resize(original.width(), original.height()));
  const uint32_t bpp = WP2FormatBpp(original.format());
  for (uint32_t y = 0; y < diff_map->height(); ++y) {
    for (uint32_t x = 0; x < diff_map->width(); ++x) {
      const uint32_t i = x * bpp + 1;
      const uint8_t err_dec = (error_map_decoded.GetRow8(y))[i];
      const uint8_t err_alt = (error_map_alt.GetRow8(y))[i];
      ComputeDiffPixel(&(original.GetRow8(y))[x * bpp], err_dec, err_alt,
                       &(diff_map->GetRow8(y))[x * bpp]);

      if (info != nullptr && info->selection.x == x && info->selection.y == y) {
        info->selection_info +=
            SPrintf("Normalized diff at %3u,%3u: decoded %u alt %u\n", x, y,
                    err_dec, err_alt);
      }
    }
  }

  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// compression

bool Params::EncodeImage() {
  enc_config_.quality = quality_;
  enc_config_.alpha_quality = alpha_quality_;
  if (!enc_config_.IsValid()) {
    SetError("Error: Invalid configuration");
    fprintf(stderr, "Invalid configuration\n");
    return false;
  }

  const double start = GetStopwatchTime();
  if (view_only_) {
    output_ = input_;
  } else {
    output_.clear();
    einfo_.force_partition = force_partition_;
    einfo_.force_segment = force_segment_;
    einfo_.force_predictor = force_predictor_;
    einfo_.force_transform = force_transform_;

    StringWriter writer(&output_);
    Metadata metadata;
    using std::swap;
    if (!keep_metadata_) swap(metadata, in_.metadata_);
    const WP2Status enc_status = Encode(in_, &writer, enc_config_);
    if (!keep_metadata_) swap(metadata, in_.metadata_);
    if (enc_status != WP2_STATUS_OK) {
      SetError("Error: Compression failed");
      fprintf(stderr, "Compression failed: %s\n", WP2GetStatusText(enc_status));
      return false;
    }
  }
  enc_duration_ms_ =
      (uint32_t)std::lround(1000. * (GetStopwatchTime() - start));
  return true;
}

bool Params::EncodeAndDecode() {
  if (!EncodeImage()) return false;
  return DecodeOutput();
}

bool Params::DecodeOutput() {
  const double start = GetStopwatchTime();

  if (view_only_ && GuessImageFormat((const uint8_t*)output_.data(),
                                     output_.size()) != FileFormat::WP2) {
    status_ = out_.ConvertFrom(in_);
  } else {
    if (VDMatch(dec_config_, "original")) {
      // The reference image is needed in 'debug_output' if the 'visual_debug'
      // is set to "original".
      status_ = dinfo_.debug_output.ConvertFrom(in_);
      if (status_ != WP2_STATUS_OK) {
        SetError("Error: Out of memory");
        return false;
      }
    }

    status_ = Decode(output_, &out_, dec_config_);
    if (bit_trace_ > 0) {
#if defined(WP2_BITTRACE)
      PrintBitTraces(dinfo_, output_.size(), /*sort_values=*/true,
                     /*use_bytes=*/bit_trace_ == 2, /*show_histograms=*/false,
                     /*short_version=*/false, bit_trace_level_);
#endif
    }
  }
  dec_duration_ms_ =
      (uint32_t)std::lround(1000. * (GetStopwatchTime() - start));

  if (status_ != WP2_STATUS_OK) {
    SetError("Error: Decompression failed");
    fprintf(stderr, "Decompression failed\n");
    return false;
  }

  if (out_.GetDistortionBlackOrWhiteBackground(in_, distortion_metric_,
                                               distortion_) != WP2_STATUS_OK) {
    fprintf(stderr, "Error while computing the distortion.\n");
  }
  if (VDMatch(dec_config_, "error-map")) {
    const MetricType metric = VDMatch(dec_config_, "PSNR")     ? PSNR
                              : VDMatch(dec_config_, "LSIM")   ? LSIM
                              : VDMatch(dec_config_, "MSSSIM") ? MSSSIM
                                                               : SSIM;
    ArgbBuffer in_premultiplied(WP2_Argb_32);
    if (in_premultiplied.ConvertFrom(in_) != WP2_STATUS_OK) {
      SetError("Error: Conversion to Argb failed!");
      fprintf(stderr, "Conversion to Argb failed!\n");
      return false;
    }
    if (VDMatch(dec_config_, "diff-with-alt")) {
      if (alt1_.IsEmpty()) {
        SetError("alt-1 is empty");
      } else if (ComputeDiffMap(metric, in_premultiplied, out_, alt1_,
                                &dec_config_.info->debug_output,
                                dec_config_.info) != WP2_STATUS_OK) {
        SetError("Error: Diff map computation failed");
        fprintf(stderr, "Diff map computation failed\n");
        return false;
      }
    } else if (ComputeErrorMap(metric, in_premultiplied, out_,
                               &dec_config_.info->debug_output,
                               dec_config_.info) != WP2_STATUS_OK) {
      SetError("Error: Error map computation failed");
      fprintf(stderr, "Error map computation failed\n");
      return false;
    }
  }
  if (!force_segment_.empty() && !dinfo_.explicit_segment_ids) {
    force_segment_.clear();
  }

  if (!ComputeYUVDistortion()) return false;
  return true;
}

bool Params::ComputeYUVDistortion() {
  if (!view_only_ && show_ == kInfo) {  // Only displayed during 'kInfo'.
    const char* const kOrigVD[] = {"original/y", "original/u", "original/v"};
    const char* const kDecVD[] = {"y/compressed", "u/compressed",
                                  "v/compressed"};
    ArgbBuffer original_yuv, decompressed_yuv;
    for (Channel c : {kYChannel, kUChannel, kVChannel}) {
      DecoderInfo decoder_info;
      dec_config_.info = &decoder_info;
      decoder_info.visual_debug = kOrigVD[c];
      if (!DecodeAndSwap(&original_yuv)) return false;
      decoder_info.visual_debug = kDecVD[c];
      if (!DecodeAndSwap(&decompressed_yuv)) return false;
      dec_config_.info = &dinfo_;
      float disto[5];
      WP2_ASSERT_STATUS(decompressed_yuv.GetDistortion(
          original_yuv, distortion_metric_, disto));
      yuv_distortion_[c] = disto[1];
    }
  } else {
    yuv_distortion_[0] = yuv_distortion_[1] = yuv_distortion_[2] = 0.f;
  }
  return true;
}

bool Params::VDMatchEncDec(const char* token) const {
  return VDMatch(dec_config_, token) || VDMatch(enc_config_, token);
}

bool Params::DecodeAndSwap(ArgbBuffer* const buffer) {
  status_ = buffer->Swap(&dec_config_.info->debug_output);  // Avoid realloc
  assert(status_ == WP2_STATUS_OK);
  status_ = dec_config_.info->debug_output.ConvertFrom(in_);
  assert(status_ == WP2_STATUS_OK);
  status_ = Decode(output_, &out_, dec_config_);
  if (status_ != WP2_STATUS_OK) {
    fprintf(stderr, "Decompression failed?!\n");
    return false;
  }
  status_ = buffer->Swap(&dec_config_.info->debug_output);
  assert(status_ == WP2_STATUS_OK);
  return true;
}

bool Params::GetOriginal() {
  if (VDMatch(dec_config_, "residuals")) {
    EncoderInfo encoder_info;
    Channel c = VDChannel(dec_config_);
    const std::string kChannelName[] = {"y", "u", "v", "a"};
    const std::string vdebug = kChannelName[c] + "/encoder/original-residuals";
    encoder_info.visual_debug = vdebug.c_str();
    encoder_info.selection = dinfo_.selection;
    enc_config_.info = &encoder_info;

    // Avoid realloc
    if (in_yuv_.Swap(&enc_config_.info->debug_output) != WP2_STATUS_OK) {
      return false;
    }
    if (!EncodeImage()) return false;
    if (in_yuv_.Swap(&enc_config_.info->debug_output) != WP2_STATUS_OK) {
      return false;
    }

    original_selection_info_ = encoder_info.selection_info;
    enc_config_.info = &einfo_;
    if (original_.SetView(in_yuv_) != WP2_STATUS_OK) return false;
    return true;
  }

  DecoderInfo decoder_info;
  // clang-format off
  decoder_info.visual_debug = VDMatchEncDec("a") ? "original/a" :
                              VDMatchEncDec("y") ? "original/y" :
                              VDMatchEncDec("u") ? "original/u" :
                              VDMatchEncDec("v") ? "original/v" :
                              VDMatchEncDec("r") ? "original/r" :
                              VDMatchEncDec("g") ? "original/g" :
                              VDMatchEncDec("b") ? "original/b" : nullptr;
  // clang-format on
  decoder_info.selection =
      (VDMatchEncDec("prediction/modes/long") ||
       VDMatchEncDec("prediction/raw") || VDMatchEncDec("compressed"))
          ? dinfo_.selection
          : Rectangle();
  if (decoder_info.visual_debug != nullptr) {
    dec_config_.info = &decoder_info;
    if (!DecodeAndSwap(&in_yuv_)) return false;
    original_selection_info_ = decoder_info.selection_info;
    dec_config_.info = &dinfo_;
    if (original_.SetView(in_yuv_) != WP2_STATUS_OK) return false;
  } else {
    original_selection_info_.clear();
    // They use different formats so we can't use a view.
    if (original_.ConvertFrom(in_) != WP2_STATUS_OK) return false;
  }
  return true;
}

bool Params::GetCompressed() {
  DecoderInfo decoder_info;
  // clang-format off
  decoder_info.visual_debug = VDMatchEncDec("a") ? "a/compressed" :
                              VDMatchEncDec("y") ? "y/compressed" :
                              VDMatchEncDec("u") ? "u/compressed" :
                              VDMatchEncDec("v") ? "v/compressed" : nullptr;
  // clang-format on
  if (decoder_info.visual_debug != nullptr) {
    dec_config_.info = &decoder_info;
    if (!DecodeAndSwap(&out_yuv_)) return false;
    dec_config_.info = &dinfo_;
  } else {
    out_yuv_.Deallocate();
  }
  return true;
}

bool Params::EncodeWebP(bool match_size) {
#if defined(WP2_HAVE_WEBP)
  WebPConfig webp_config;
  if (!WebPConfigPreset(&webp_config, WEBP_PRESET_DEFAULT, quality_)) {
    return false;
  }
  if (enc_config_.thread_level > 0) ++webp_config.thread_level;
  webp_config.method = (enc_config_.effort * 6 + 4) / 9;
  webp_config.segments = std::min(enc_config_.segments, 4);
  webp_config.sns_strength = (int)std::lround(enc_config_.sns);
  webp_config.pass = 6;
  if (match_size) {
    webp_config.target_size = (int)output_.size();
  } else {
    if (distortion_metric_ != PSNR) return false;
    webp_config.target_PSNR = distortion_[4];
  }
  if (!WebPValidateConfig(&webp_config)) return false;

  MemoryWriter writer;
  status_ = CompressWebP(in_, webp_config, &writer);
  if (status_ != WP2_STATUS_OK) return false;
  webp_size_ = writer.size_;

  status_ = ReadImage(writer.mem_, writer.size_, &webp_, FileFormat::WEBP);
  if (status_ != WP2_STATUS_OK) return false;

  float disto[5];
  status_ =
      webp_.GetDistortionBlackOrWhiteBackground(in_, distortion_metric_, disto);
  if (status_ != WP2_STATUS_OK) return false;
  webp_distortion_ = disto[4];

  show_ = kWebP;
  return true;
#else
  return false;
#endif  // WP2_HAVE_WEBP
}

bool Params::EncodeAV1(bool copy_partition, float av1_quality,
                       size_t* const av1_file_size) {
#if !defined(WP2_HAVE_AOM_DBG)
  if (copy_partition) return false;
#endif
#if defined(WP2_HAVE_AOM)
  ParamsAV1 cfg;
  cfg.quality = av1_quality;
  // The settings below seem to give the best results (yes, even threads).
  cfg.effort = 9;
  cfg.threads = 1;
  cfg.pass = 2;
  cfg.use_yuv444 = (enc_config_.uv_mode != EncoderConfig::UVMode420);
  cfg.draw_blocks = (display_block_ != kDisplayNone);
  cfg.draw_transforms = (display_block_ == kDisplayYCoeffs);
  if (copy_partition) force_partition_.clear();

  printf("Compressing AV1 at quality %.1f...\n", cfg.quality);
  std::string tmp;
  double timing[2];
  ShiftAlt();
  if (CompressAV1(
          in_, cfg, &alt1_, &tmp, timing, /*blocks=*/nullptr,
          /*transforms=*/(copy_partition ? &force_partition_ : nullptr)) !=
      WP2_STATUS_OK) {
    return false;
  }
  printf("Done compressing AV1 in just %.2f seconds (%u bytes)!\n", timing[0],
         (uint32_t)tmp.size());

  if (av1_file_size != nullptr) *av1_file_size = tmp.size();
  alt1_msg_ = SPrintf("AV1 - size = %u", (uint32_t)tmp.size());

  if (!cfg.draw_blocks && !cfg.draw_transforms) {
    // Only print distortion if the output image is unaltered by debug overlays.
    float disto[5];
    if (alt1_.GetDistortionBlackOrWhiteBackground(in_, distortion_metric_,
                                                  disto) != WP2_STATUS_OK) {
      return false;
    }
    alt1_msg_ +=
        SPrintf("  %s = %.1f", kMetricNames[distortion_metric_], disto[4]);
  }

  if (copy_partition) {
    ConvertPartition(in_.width(), in_.height(), enc_config_.partition_set,
                     /*ignore_invalid=*/true, &force_partition_);
  }

  show_ = kAlt1;
  return true;
#else
  (void)copy_partition;
  return false;
#endif  // WP2_HAVE_AOM
}

bool Params::EncodeAV1ToMatch(bool copy_partition, size_t target_file_size) {
  // Begin by testing 0, it's probably a good choice.
  int lo_q = -100, hi_q = 100, mid_q;
  do {
    mid_q = (lo_q + hi_q) / 2;
    size_t file_size;
    if (!EncodeAV1(copy_partition, mid_q, &file_size)) {
      AddInfo("AV1 not available");
      return false;
    }
    if (std::abs((float)file_size - target_file_size) <
        0.001f * target_file_size) {
      break;
    } else if (file_size < target_file_size) {
      lo_q = mid_q + 1;
    } else {
      hi_q = mid_q;
    }
  } while (lo_q >= 0 && lo_q < hi_q);
  return true;
}

//------------------------------------------------------------------------------

void Params::UpdateViewZoomLevel(int incr) {
  if (view_zoom_level_ + incr <= 0) {
    zoom_level_ -= view_zoom_level_;
    view_zoom_level_ = 0;
  } else if ((width_ >> (view_zoom_level_ + incr) > 1) &&
             (height_ >> (view_zoom_level_ + incr) > 1)) {
    view_zoom_level_ += incr;
    // Also change the zoom level so the window size doesn't change.
    zoom_level_ += incr;
  }

  if (view_zoom_level_ == 0) {
    view_rect_ = {0, 0, width_, height_};
  } else {
    const int view_w = std::ceil((float)width_ / (1 << view_zoom_level_));
    const int view_h = std::ceil((float)height_ / (1 << view_zoom_level_));
    const int img_x_down = ToImageX(mouse_x_down_);
    const int img_y_down = ToImageY(mouse_y_down_);
    // Compute view x/y so that the last point that was clicked stays in the
    // same place.
    const int view_x =
        Clamp<int>(img_x_down - view_w * mouse_x_down_ / viewport_width_, 0,
                   width_ - view_w);
    const int view_y =
        Clamp<int>(img_y_down - view_h * mouse_y_down_ / viewport_height_, 0,
                   height_ - view_h);
    view_rect_ = {(uint32_t)view_x, (uint32_t)view_y, (uint32_t)view_w,
                  (uint32_t)view_h};
  }
}

void Params::ApplyZoomLevel() {
  if (zoom_level_ >= 0) {
    viewport_width_ = view_rect_.width << zoom_level_;
    viewport_height_ = view_rect_.height << zoom_level_;
  } else {
    viewport_width_ = view_rect_.width >> -zoom_level_;
    viewport_height_ = view_rect_.height >> -zoom_level_;
  }
}

bool Params::SetCurrentFile(size_t file) {
  if (file >= files_.size()) file = files_.size() - 1;
  if (file != current_file_) {
    ClearForcedElements();
    current_file_ = file;
    const std::string& file_name = files_[file];
    status_ = IoUtilReadFile(file_name.c_str(), &input_);
    if (status_ != WP2_STATUS_OK) {
      fprintf(stderr, "Could not read the input file %s\n", file_name.c_str());
      return false;
    }
    const uint8_t* data = (const uint8_t*)input_.data();
    const size_t data_size = input_.size();

    const FileFormat format = GuessImageFormat(data, data_size);
    if (format == FileFormat::UNSUPPORTED) {
      fprintf(stderr, "Unsupported input format\n");
      return false;
    }
    if (ReadImage(data, data_size, &in_, format) != WP2_STATUS_OK) {
      fprintf(stderr, "Could not decode the input file %s\n",
              file_name.c_str());
      return false;
    }

    if (crop_ && in_.SetView(in_, crop_area_) != WP2_STATUS_OK) {
      fprintf(stderr, "Error! Cropping operation failed. Skipping.");
    }

    preview_color_ = {0x00u, 0x00u, 0x00u, 0x00u};
    preview_.Deallocate();
    if (format == FileFormat::WP2) {
      BitstreamFeatures wp2_features;
      if (wp2_features.Read(data, data_size) != WP2_STATUS_OK) {
        fprintf(stderr, "Inconsistent state: could not read WP2 features.\n");
        return false;
      }
      if (wp2_features.has_preview &&
          ExtractPreview(data, data_size, &preview_) != WP2_STATUS_OK) {
        fprintf(stderr, "Could not decode preview of %s\n",
                file_name.c_str());
        preview_.Deallocate();
      }
      preview_color_ = ToArgb32b(wp2_features.preview_color);
    }
    width_ = in_.width();
    height_ = in_.height();
    view_zoom_level_ = 0;
    view_rect_ = {0, 0, width_, height_};

    if (!EncodeAndDecode()) return false;
  }
  return (status_ == WP2_STATUS_OK);
}

void Params::ShiftAlt() {
  const WP2Status s = alt1_.Swap(&alt2_);
  (void)s;
  assert(s == WP2_STATUS_OK);
  std::swap(alt1_msg_, alt2_msg_);
  alt1_msg_.clear();
  alt1_.Deallocate();
}

bool Params::SetBitstream(const char* const file_name) {
  std::string data;
  status_ = IoUtilReadFile(file_name, &data);
  if (status_ != WP2_STATUS_OK) {
    fprintf(stderr, "Could not read the file %s\n", file_name);
    return false;
  }

  ArgbBuffer tmp;
  status_ = Decode(data, &tmp);
  if (status_ != WP2_STATUS_OK) {
    fprintf(stderr, "Could not decode the file %s\n", file_name);
    return false;
  }

  ClearForcedElements();
  alt1_msg_.clear();
  alt1_.Deallocate();
  alt2_msg_.clear();
  alt2_.Deallocate();

  view_only_ = true;
  using std::swap;
  swap(data, input_);
  data.clear();
  status_ = tmp.Swap(&in_);
  return (status_ == WP2_STATUS_OK);
}

bool Params::SetAltFile(const char* const file_name) {
  ShiftAlt();
  status_ = ReadImage(file_name, &alt1_);
  if (status_ != WP2_STATUS_OK) {
    fprintf(stderr, "Could not decode the alternate file %s\n", file_name);
    return false;
  }
  const uint32_t w = alt1_.width(), h = alt1_.height();
  if (w != width_ || h != height_) {
    alt1_.Deallocate();
    fprintf(stderr,
            "Alternate picture has incompatible dimensions "
            " (%dx%d vs expected %dx%d)\n",
            w, h, width_, height_);
    return false;
  }
  alt1_msg_ = SPrintf("- Alt Pic (%s) -", file_name);
  return true;
}

// Copies the current canvas to the alternative buffer.
bool Params::SetAltImage() {
  if (show_ == kAlt1 || show_ == kAlt2) return false;
  ArgbBuffer tmp;
  if (GetCurrentCanvas(&tmp)) {
    ShiftAlt();
    status_ = tmp.IsView() ? alt1_.CopyFrom(tmp) : alt1_.Swap(&tmp);
    if (status_ == WP2_STATUS_OK) alt1_msg_ = "- Saved canvas -";
  }
  if (status_ != WP2_STATUS_OK) {
    fprintf(stderr, "Could not copy the alternate image\n");
    return false;
  }
  return true;
}

bool Params::DumpCurrentCanvas(const char* const file_path) {
  ArgbBuffer tmp;
  if (GetCurrentCanvas(&tmp)) {
    status_ = SaveImage(tmp, file_path, /*overwrite=*/true);
  }

  if (status_ != WP2_STATUS_OK) {
    fprintf(stderr, "Could not dump the current canvas\n");
    return false;
  }
  return true;
}

// Returns the currently displayed buffer.
const ArgbBuffer& Params::GetBuffer() const {
  if (show_ == kOriginal) return original_;
  if (show_ == kPreview) return preview_;
  if (show_ == kWebP) return webp_;
  if (show_ == kAlt1 && !alt1_.IsEmpty()) return alt1_;
  if (show_ == kAlt2 && !alt2_.IsEmpty()) return alt2_;
  if (show_ == kCompressed) return out_yuv_.IsEmpty() ? out_ : out_yuv_;
  if (VDMatch(dec_config_, "")) return dec_config_.info->debug_output;
  if (VDMatch(enc_config_, "")) return enc_config_.info->debug_output;
  return out_;
}

// Copies the current canvas to 'buffer'.
bool Params::GetCurrentCanvas(ArgbBuffer* const buffer) {
  if (show_ == kPreviewColor) {
    ShiftAlt();
    status_ = buffer->Resize(out_.width(), out_.height());
    if (status_ == WP2_STATUS_OK) buffer->Fill(preview_color_);
  } else if (GetBuffer().format() == buffer->format()) {
    status_ = buffer->SetView(GetBuffer());
  } else {
    status_ = buffer->ConvertFrom(GetBuffer());
  }
  return (status_ == WP2_STATUS_OK);
}

//------------------------------------------------------------------------------
// Blocks

void Params::glRect(const Rectangle& rect, uint32_t thickness) const {
  if (!view_rect_.Contains(rect.x, rect.y)) return;
  const float x = 2.f * (rect.x - view_rect_.x) / view_rect_.width - 1.f;
  // Compensate OpenGL bottom-left origin by shifting by one pixel vertically.
  const float y =
      1.f - 2.f * (rect.y - view_rect_.y + rect.height) / view_rect_.height -
      1.f / viewport_height_;
  const float w = 2.f * rect.width / view_rect_.width;
  const float h = 2.f * rect.height / view_rect_.height;

  // Thickness
  const float px_w = 2.f / viewport_width_;
  const float px_h = 2.f / viewport_height_;
  for (uint32_t j = 0; j < thickness; ++j) {
    for (uint32_t i = 0; i < thickness; ++i) {
      glRectf(x + i * px_w, y - j * px_h, x + w + i * px_w, y + h - j * px_h);
    }
  }
}

//------------------------------------------------------------------------------
// DisplayBlockInfo()

#if defined(WP2_BITTRACE)

// Helper class for the creation of a nice looking paragraph.
class ParagraphMaker {
 public:
  // Initializes with a potential prefix that will define an indentation for the
  // rest of the paragraph.
  void Init(const std::string& str = "") {
    strs_.clear();
    strs_.push_back(str);
    indentation_ = std::string(str.length(), ' ');
  }
  // Appends a string to the last string of the list.
  void Append(const std::string& str) {
    assert(!strs_.empty());
    strs_.back() += str;
  }
  // Appends a new element to the list.
  void AppendToList(const std::string& str) {
    assert(!strs_.empty());
    strs_.back() += str + ", ";
    if (strs_.back().size() >= 50u) strs_.push_back(indentation_);
  }
  // Push back all the elements of the list to the msg.
  void PushBackTo(std::vector<std::string>* const msg) {
    if (strs_.empty()) return;
    // Remove the last element if empty.
    if (strs_.back() == indentation_) strs_.pop_back();
    if (strs_.empty()) return;
    // Remove the last ', '.
    const uint32_t index = strs_.back().size() - 2;
    if (strs_.back().substr(index) == ", ") {
      strs_.back() = strs_.back().substr(0, index);
    }
    msg->insert(msg->end(), strs_.begin(), strs_.end());
  }

 private:
  std::vector<std::string> strs_;
  std::string indentation_;
};


constexpr uint32_t kLastChannel = 4;
constexpr const char* const kEncNames[] = {
  "X", "Mtd0", "Mtd1", "DC  ", "Zero"
};

// Isolate the bit costs about residuals (Y, U, V, A, remaining stuff).
static void IsolateBitTraces(const BlockInfo& b,
    std::map<const std::string, LabelStats> bit_traces[],
    double bit_sums[]) {
  for (const auto& bt : b.bit_traces) {
    std::string label = bt.first;
    uint32_t channel;
    for (channel = 0; channel < kLastChannel; ++channel) {
      const std::string prefix = kCoeffsStr[channel];

      if (label.size() >= prefix.size() &&
          std::strncmp(label.c_str(), prefix.c_str(), prefix.size()) == 0) {
        // 'label' starts with 'prefix', remove 'prefix' from it.
        assert(label.size() > prefix.size() + 1 &&
               label[prefix.size()] == '/');
        label = label.substr(prefix.size() + 1);
        break;
      }
    }

    // Merge by first token.
    const size_t slash_pos = label.find('/');
    if (slash_pos != std::string::npos) label = label.substr(0, slash_pos);

    bit_traces[channel][label].bits += bt.second.bits;
    bit_traces[channel][label].num_occurrences += bt.second.num_occurrences;
    bit_sums[channel] += bt.second.bits;
  }
}

void Params::PrintBlockHeader(const BlockInfo& b,
                              std::vector<std::string>* const msg) const {
  std::map<const std::string, LabelStats> bit_traces[kLastChannel + 1];
  double bit_sums[kLastChannel + 1] = {0};
  IsolateBitTraces(b, bit_traces, bit_sums);

  ParagraphMaker pm;
  pm.Init("Costs: ");
  for (uint32_t c = 0; c < (b.has_lossy_alpha ? kLastChannel : 3); ++c) {
    pm.AppendToList(SPrintf("%s: %.1f", kCoeffsStr[c], bit_sums[c]));
  }
  for (const auto& p : bit_traces[kLastChannel]) {
    pm.AppendToList(SPrintf("%s: %.1f", p.first.c_str(), p.second.bits));
  }
  pm.PushBackTo(msg);
  msg->push_back(SPrintf("segment id: %d (%s %s)", b.segment_id,
                          b.is420 ? "is420" : "",
                          b.has_lossy_alpha ? "has_alpha_res" : ""));
  msg->push_back(SPrintf("transform: %s %s",
                          WP2TransformNames[b.tf_x],
                          WP2TransformNames[b.tf_y]));
  for (uint32_t tf_i = 0; tf_i < 4; ++tf_i) {
    if (b.encoding_method[kYChannel][tf_i] == -1) continue;
    msg->push_back(
      SPrintf("Transform %u. Mthd: Y=%s U=%s V=%s (A=%s)", tf_i,
              kEncNames[b.encoding_method[kYChannel][tf_i] + 1],
              kEncNames[b.encoding_method[kUChannel][tf_i] + 1],
              kEncNames[b.encoding_method[kVChannel][tf_i] + 1],
              kEncNames[b.encoding_method[kAChannel][tf_i] + 1]));
  }
}

void Params::PrintResiduals(const BlockInfo& b, const BlockInfo& b_enc,
                            std::vector<std::string>* const msg) const {
  std::map<const std::string, LabelStats> bit_traces[kLastChannel + 1];
  double bit_sums[kLastChannel + 1] = {0};
  IsolateBitTraces(b, bit_traces, bit_sums);

  ParagraphMaker pm;
  const uint32_t channel =
      (display_block_ == kDisplayYCoeffs) ? kYChannel :
      (display_block_ == kDisplayUCoeffs) ? kUChannel :
      (display_block_ == kDisplayVCoeffs) ? kVChannel : kAChannel;

  msg->push_back(SPrintf("%s channel.", kChannelStr[channel]));
  for (uint32_t tf_i = 0; tf_i < 4; ++tf_i) {
    if (!b.residual_info[channel][tf_i].empty()) {
      pm.Init();
      // Display info about the residuals (index of last element ...).
      pm.Append(SPrintf("Transform %u. Residuals: ", tf_i));
      for (const std::string& str : b.residual_info[channel][tf_i]) {
        pm.AppendToList(str);
      }
      pm.PushBackTo(msg);
    }
  }
  // Display stats about residuals.
  {
    typedef decltype(b.bit_traces)::value_type StatType;
    std::vector<StatType> stats(bit_traces[channel].begin(),
                                bit_traces[channel].end());
    // Sort those stats from lowest to highest usage.
    std::vector<uint32_t> indices(stats.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [stats](uint32_t i1, uint32_t i2) {
                return (stats[i1].second.bits < stats[i2].second.bits);
              });
    pm.Init("Costs: ");
    pm.Append(SPrintf("all: %.1f, ", bit_sums[channel]));
    // Display stats as bit cost (/num of occurences = individual bit cost).
    for (uint32_t i : indices) {
      const auto& p = stats[i];
      pm.AppendToList(
          SPrintf("%s: %.1f (/%d=%.1f)", p.first.c_str(), p.second.bits,
                  p.second.num_occurrences,
                  p.second.bits / p.second.num_occurrences));
    }
    pm.PushBackTo(msg);
  }
  const int bw = b.rect.width, bh = b.rect.height;
  for (uint32_t tf_i = 0; tf_i < 4; ++tf_i) {
    if (b.encoding_method[channel][tf_i] == -1) continue;
    msg->push_back(SPrintf("Enc %u. Mthd: %s", tf_i,
                           kEncNames[b.encoding_method[channel][tf_i] + 1]));
    const uint32_t scale =
        (b.is420 && (channel == 1 || channel == 2)) ? 2 : 1;
    const BlockSize split_size = GetSplitSize(
        GetBlockSize(bw / kMinBlockSizePix, bh / kMinBlockSizePix),
        b.split_tf[channel]);
    const uint32_t w = BlockWidthPix(split_size) / scale;
    const uint32_t h = BlockHeightPix(split_size) / scale;
    const bool show_original = (show_ == kOriginal);
    if (show_original || b.encoding_method[channel][tf_i] !=
        (int8_t)EncodingMethod::kAllZero) {
      int16_t res[kMaxBlockSizePix2];
      int32_t coeffs[kMaxBlockSizePix2];
      const uint32_t num_coeffs = w * h;
      std::copy(&b_enc.original_res[channel][tf_i][0],
                &b_enc.original_res[channel][tf_i][0] + num_coeffs,
                &res[0]);
      const bool reduced_transform =
          (channel == kUChannel || channel == kVChannel) && b_enc.is420;
      WP2Transform2D(res,
                     (WP2TransformType)b_enc.tf_x,
                     (WP2TransformType)b_enc.tf_y,
                     w, h, coeffs, reduced_transform);

      const float norm = 2. * (int32_t)std::sqrt(num_coeffs);
      for (uint32_t j = 0; j < h; ++j) {
        std::string line;
        for (uint32_t i = 0; i < w; ++i) {
          const uint32_t idx = i + w * j;
          if (show_original) {
            const int32_t v = coeffs[idx];
            if (std::abs(v) >= (norm * 1.00)) {
              line += SPrintf("%3d ", (int)(v / norm));
            } else {
              const char symbol = (std::abs(v) >= (norm * 0.50)) ? 'X'
                                : (std::abs(v) >= (norm * 0.25)) ? 'x'
                                                                 : '.';
              line += SPrintf("  %c ", symbol);
            }
          } else {
            const int32_t v = b_enc.coeffs[channel][tf_i][idx];
            line += v ? SPrintf("%3d ", v) : "  . ";
          }
        }
        msg->push_back(line);
      }
      if (show_original) {
        msg->push_back(SPrintf("original coeffs in units of %.1f", norm));
      }
    } else {
      msg->push_back(" - no coeffs -");
    }
  }
}

void Params::PrintPredModes(const BlockInfo& b,
                            std::vector<std::string>* const msg) const {
  msg->push_back(SPrintf("Chroma prediction mode: %d", b.uv_pred));
  msg->push_back(SPrintf("Luma prediction: %d", b.y_pred));
  if (b.has_lossy_alpha) {
    msg->push_back(SPrintf("Alpha prediction: %d", b.a_pred));
  } else {
    msg->push_back("-no alpha-");
  }

  msg->push_back(b.y_context_is_constant ? "Y context is constant"
                                         : "Y context is not constant");
  const int optimize_modes =
      (b.y_context_is_constant || enc_config_.effort == 0) ? 0 :
      (enc_config_.effort == 1)                            ? 1 : 2;
  msg->push_back(
      SPrintf("OptimizeModes%d() Y score: %.1f (%.1f per pixel)",
              optimize_modes, b.pred_scores[kYChannel],
              b.pred_scores[kYChannel] / b.rect.GetArea()));
  msg->push_back(SPrintf("FindBestUVModes() score: %.1f (%.1f per pixel)",
                         b.pred_scores[kUChannel],
                         b.pred_scores[kUChannel] / b.rect.GetArea()));
}

#endif  // WP2_BITTRACE

void Params::DisplayBlockInfo(std::vector<std::string>* const msg) {
  if (display_block_ == kDisplayNone) return;
#if defined(WP2_BITTRACE)
  assert(dinfo_.store_blocks && einfo_.store_blocks &&
         einfo_.blocks.size() == dinfo_.blocks.size());
  // Sort encoded and decoded blocks for easy comparison.
  for (std::vector<BlockInfo>* blocks : {&dinfo_.blocks, &einfo_.blocks}) {
    std::sort(blocks->begin(), blocks->end(),
              [](const BlockInfo& l, const BlockInfo& r) {
                return (l.rect.x < r.rect.x) ||
                       (l.rect.x == r.rect.x && l.rect.y < r.rect.y);
              });
  }

  const int selection_x = (int)dinfo_.selection.x;
  const int selection_y = (int)dinfo_.selection.y;
  for (uint32_t index = 0; index < dinfo_.blocks.size(); ++index) {
    const BlockInfo& b = dinfo_.blocks[index];
    const BlockInfo& b_enc = einfo_.blocks[index];
    assert(b.rect == b_enc.rect);
    assert(b.segment_id == b_enc.segment_id);
    assert(b.y_context_is_constant == b_enc.y_context_is_constant);

    const int bx = b.rect.x, by = b.rect.y;
    const int bw = b.rect.width, bh = b.rect.height;
    if (!view_rect_.Contains(bx, by)) continue;

    const bool block_selected = b.rect.Contains(selection_x, selection_y);
    if (display_block_ == kDisplayGrid || !block_selected) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glColor4f(0.0f, 0.2f, 0.1f, 0.2f);
      glRect(b.rect);
      continue;
    }

    bool forced = false;
    for (const Rectangle& rect : einfo_.force_partition) {
      if (rect == b.rect) {
        forced = true;
        break;
      }
    }
    msg->push_back(SPrintf("size: %2d x %2d  pos: %3d x %3d  %s",
                           bw, bh, bx, by, forced ? "(forced)" : ""));
    float disto[5];
    const Rectangle rect = b.rect.ClipWith({0, 0, width_, height_});
    WP2_ASSERT_STATUS(out_.GetDistortionBlackOrWhiteBackground(
        in_, rect, distortion_metric_, disto));
    msg->push_back(SPrintf("bits:%5.1f (%.2f bpp) %s:%5.1f", b.bits,
                           b.bits / b.rect.GetArea(),
                           kMetricNames[distortion_metric_], disto[4]));

    if (display_block_ == kDisplayHeader) {
      PrintBlockHeader(b, msg);
    } else if (display_block_ <= kDisplayACoeffs) {
      if (display_block_ == kDisplayACoeffs && !b.has_lossy_alpha) {
        msg->push_back("- no alpha -");
      } else {
        PrintResiduals(b, b_enc, msg);
      }
    } else if (display_block_ == kDisplayPredModes) {
      PrintPredModes(b_enc, msg);
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glColor4f(1.f, 0.f, 0.f, 0.5f);  // red
    glRect(b.rect);
  }
#endif  // WP2_BITTRACE
}

//------------------------------------------------------------------------------

void Params::ClearForcedElements() {
  force_partition_.clear();
  force_segment_.clear();
  force_predictor_.clear();
  force_transform_.clear();
  forcing_block_ = Rectangle();
}

bool Params::IsDebugViewVisible() const {
  return (show_ == kDebug || show_ == kInfo || show_ == kHelp ||
          show_ == kMenu);
}

bool Params::IsPartitionVisible() const {
  return (display_block_ != kDisplayNone ||
          (IsDebugViewVisible() && VDMatch(dec_config_, "partition"))) &&
         !IsSegmentVisible() && !IsPredictorVisible() && !IsTransformVisible();
}

bool Params::IsSegmentVisible() const {
  return IsDebugViewVisible() && VDMatch(dec_config_, "segment-ids");
}

bool Params::IsPredictorVisible() const {
  return IsDebugViewVisible() && VDMatch(dec_config_, "prediction");
}

bool Params::IsTransformVisible() const {
  return IsDebugViewVisible() && VDMatch(dec_config_, "transform");
}

void Params::DisplayPartition() const {
  if (!IsPartitionVisible()) return;

  // Adapt to zoom.
  const uint32_t thick = getLineThickness();
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  // Encoded forced blocks.
  glColor4f(1.0f, 0.2f, 1.0f, 1.0f);
  for (const Rectangle& rect : einfo_.force_partition) {
    if (!view_rect_.Contains(rect.x, rect.y)) continue;
    glRect(rect, thick);
  }

  glLineStipple(2 * thick, 0xAAAA);
  glEnable(GL_LINE_STIPPLE);

  // All blocks, including those that haven't been applied yet.
  glColor4f(0.6f, 1.0f, 0.6f, 1.0f);
  for (const Rectangle& rect : force_partition_) glRect(rect, thick);

  // If a block is currently being drawn with the mouse.
  if (forcing_block_.width > 0 && forcing_block_.height > 0) {
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glRect(forcing_block_, thick);
  }

  glDisable(GL_LINE_STIPPLE);
}

void Params::DisplayForcedSegments() const {
  if (!IsSegmentVisible()) return;

#if defined(WP2_BITTRACE)
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  const uint32_t thick = getLineThickness();
  glLineStipple(2 * thick, 0xAAAA);

  for (const auto& segment : force_segment_) {
    if (!view_rect_.Contains(segment.x, segment.y)) continue;
    const BlockInfo* const b = einfo_.FindBlock(segment.x, segment.y);
    if (b != nullptr) {
      const Rectangle rect{b->rect.x, b->rect.y,
                           b->rect.width - 1, b->rect.height - 1};
      glColor4f(0.f, 0.f, 0.f, 1.f);
      glRect(rect, thick * 2);
      const Argb32b color = kSegmentColors[segment.segment_id];
      glColor4f(color.r / 255.f, color.g / 255.f, color.b / 255.f, 1.f);
      glRect(rect, thick);
    }
  }
#endif
}

void Params::DisplayForcedPredictors() const {
  if (!IsPredictorVisible()) return;

#if defined(WP2_BITTRACE)
  Channel channel = VDChannel(dec_config_);
  if (channel == kVChannel) channel = kUChannel;

  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  for (const auto& predictor : force_predictor_) {
    if (predictor.channel != channel) continue;
    if (!view_rect_.Contains(predictor.x, predictor.y)) continue;
    const BlockInfo* const b = einfo_.FindBlock(predictor.x, predictor.y);
    if (b != nullptr) {
      if (channel == kYChannel && b->y_context_is_constant) {
        glColor4f(1.f, 0.f, 0.f, 1.f);
      } else {
        glColor4f(0.f, 1.f, 0.f, 1.f);
      }
      glRect(b->rect, 4.f);

      const std::vector<std::string>& predictors =
            (channel == kYChannel) ? dinfo_.y_predictors :
            (channel == kAChannel) ? dinfo_.a_predictors :
                                     dinfo_.uv_predictors;
      const uint32_t predictor_id =
            std::min(predictor.predictor_id, (uint8_t)(predictors.size() - 1));

      const float color = (float)predictor_id / (predictors.size() - 1);
      glColor4f(color, color, color, 1.f);
      glRect(b->rect, 2.f);

      const std::string& title = predictors[predictor_id];
      glColor4f(1.f, 0.f, 1.f, 1.f);
      PrintString((predictor.x - view_rect_.x) * 2.f / view_rect_.width - 1,
                  1 - (predictor.y - view_rect_.y) * 2.f / view_rect_.height,
                  title);
    }
  }
#endif
}

void Params::DisplayForcedTransforms() const {
  if (!IsTransformVisible()) return;

#if defined(WP2_BITTRACE)
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  const uint32_t thick = getLineThickness();
  glLineStipple(2 * thick, 0xAAAA);

  for (const auto& forced : force_transform_) {
    if (!view_rect_.Contains(forced.x, forced.y)) continue;
    const BlockInfo* const b = einfo_.FindBlock(forced.x, forced.y);
    if (b != nullptr) {
      const Rectangle rect{b->rect.x, b->rect.y,
                           b->rect.width - 1, b->rect.height - 1};
      glColor4f(1.f, 1.f, 1.f, 1.f);
      glRect(rect, thick * 2);

      const std::string title = SPrintf("%s_%s",
          WP2TransformNames[kTfX[forced.transform]],
          WP2TransformNames[kTfY[forced.transform]]);
      glColor4f(1.f, 0.f, 1.f, 1.f);
      PrintString((forced.x - view_rect_.x) * 2.f / view_rect_.width - 1,
                  1 - (forced.y - view_rect_.y) * 2.f / view_rect_.height,
                  title);
    }
  }
#endif
}

uint32_t Params::getLineThickness() const {
  return (display_block_ != kDisplayNone)
             ? 1u
             : std::max(1u, std::min(viewport_width_ / view_rect_.width,
                                     viewport_height_ / view_rect_.height));
}

bool Params::IsHamburgerMenuVisible() const {
  return (show_ == kDebug || show_ == kMenu ||
          (show_ == kOriginal && display_block_ != kDisplayNone)) &&
         (forcing_block_.width == 0 || forcing_block_.height == 0);
}

//------------------------------------------------------------------------------

bool Params::ReadAndConvertPartition(bool read_only) {
  force_partition_.clear();
  if (!ReadPartition(partition_file_path_, read_only, &force_partition_)) {
    return false;
  }
  ConvertPartition(in_.width(), in_.height(), enc_config_.partition_set,
                   /*ignore_invalid=*/false, &force_partition_);
  return true;
}

//------------------------------------------------------------------------------
// Handlers

// T must be cast-able to and from 'int'

template <class T>
void ModifyInRange(T& value, int max, int increment) {
  int v = (int)value + increment;
  v = (v + max) % max;  // deals with negative values
  value = (T)v;
}
template <class T>
void Modify(T& value, int max, bool increment, int increment_step = 1) {
  ModifyInRange(value, max, increment ? increment_step : -increment_step);
}
// For ranges
template <class T>
void ModifyR(T& value, T max, bool increment, int increment_step = 1) {
  if (glutGetModifiers() == GLUT_ACTIVE_ALT) {
    // Toggle on/off (GLUT doesn't handle shift+alt+key: can't use 'increment')
    value = (T)(((int)value > (int)max / 2) ? 0 : (int)max - 1);
  } else {
    Modify(value, max, increment, increment_step);
  }
}

void Params::ReshapeWindow() {
  ApplyZoomLevel();
  glutReshapeWindow(viewport_width_, viewport_height_);
  // Update viewport_width/height based on actual window dimensions. These can
  // be different from the ones we requested because of the size of the screen
  // or because the window is in fullscreen mode.
  viewport_width_ = glutGet(GLUT_WINDOW_WIDTH);
  viewport_height_ = glutGet(GLUT_WINDOW_HEIGHT);
  glutPostRedisplay();
}

void Params::HandleKey(unsigned char key, int pos_x, int pos_y) {
  mouse_x_ = pos_x;
  mouse_y_ = pos_y;

  // Ignore key strokes if the menu is opened.
  if (show_ == kMenu) return;

  const bool is_ctrl = (glutGetModifiers() == GLUT_ACTIVE_CTRL);
  const bool is_alt = (glutGetModifiers() == GLUT_ACTIVE_ALT);
  (void)is_ctrl;

  // first, the keys with special modifiers
  if (is_alt && key >= '0' && key <= '9') {
    if (key == '1' && !alt1_.IsEmpty()) {
      show_ = kAlt1;
    } else if (key == '2' && !alt2_.IsEmpty()) {
      show_ = kAlt2;
    }
    glutPostRedisplay();
  } else if (key == 'q' || key == 'Q' || key == 27 /* Esc */) {
    if (key == 27 /* Esc */ &&
        (einfo_.selection.width > 0 || einfo_.selection.height > 0)) {
      einfo_.selection = dinfo_.selection = {0, 0, 0, 0};
      if (VDMatch(enc_config_, "encoder")) {
        if (!EncodeAndDecode()) return;
      } else {
        if (!DecodeOutput()) return;
      }
      AddInfo("Selection cleared. Press escape again to quit.");
    } else {
#if defined(WP2_HAVE_FREEGLUT)
      glutLeaveMainLoop();
#else
      exit(0);
#endif
    }
  } else if (key == ' ') {
    show_ = kOriginal;
    if (!GetOriginal()) AddInfo("An error occurred");
    glutPostRedisplay();
  } else if (key == 'W' || key == 'w') {
    if (!EncodeWebP(key == 'w')) AddInfo("WebP not available");
    glutPostRedisplay();
  } else if (key == 'r' || key == 'R') {
    const bool binary_search = is_alt;
    const bool copy_partition = (key == 'R');
    if (binary_search ? EncodeAV1ToMatch(copy_partition, output_.size())
                      : EncodeAV1(copy_partition, quality_)) {
      AddInfo(copy_partition ? "Copied partition (press 'c' to use it)"
                             : "Stored AV1 as alt-1 image");
    } else {
      AddInfo("AV1 not available");
    }
    glutPostRedisplay();
  } else if (key == '\\' && !preview_.IsEmpty()) {
    show_ = kPreview;
    glutPostRedisplay();
  } else if (key == '|' && preview_color_.a == 0xFFu) {
    show_ = kPreviewColor;
    glutPostRedisplay();
  } else if (key == 13) {  // return
    if (!alt1_.IsEmpty()) {
      show_ = kAlt1;
      glutPostRedisplay();
    }
  } else if (key == '\t') {
    show_ = kCompressed;
    if (!GetCompressed()) AddInfo("Unable to display compressed image");
    glutPostRedisplay();
  } else if (key == 'a' || key == 'A') {
    if (is_alt) {
      if (SetAltFile(dump_png_path_)) {
        AddInfo(SPrintf("Loaded alt-1 from '%s'", dump_png_path_));
      } else {
        AddInfo(SPrintf("Could not load '%s'", dump_png_path_));
      }
    } else if (key == 'A') {
      if (DumpCurrentCanvas(dump_png_path_)) {
        AddInfo(SPrintf("Dumped canvas to '%s'", dump_png_path_));
      } else {
        AddInfo(SPrintf("Could not dump to '%s'", dump_png_path_));
      }
    } else {
      if (SetAltImage()) {
        AddInfo("Saved canvas to alt-1 image");
      } else {
        AddInfo("Unable to save canvas");
      }
      glutPostRedisplay();
    }
  } else if (key == 'b' || key == 'B') {
    const bool need_update = (dinfo_.store_blocks == 0);
    Modify(display_block_, kDisplayNum, key == 'b');
    dinfo_.store_blocks = (display_block_ != kDisplayNone);
    if (need_update) {
      if (!DecodeOutput()) return;
    }
    glutPostRedisplay();
  } else if (key == 'E' || key == 'e') {
    std::string selection =
        GetLeaf(menu_selection_, /*leaf_offset=*/(key == 'e') ? 1 : -1);
    if (selection.find("error-map") != 0) {  // NOLINT
      selection = "error-map/PSNR";
    }
    menu_selection_prev_ = menu_selection_;
    if (!SetMenuSelection(selection)) return;
  } else if (key == 's' || key == 'S') {
    ModifyR(enc_config_.effort, 10, key == 's');
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("Effort: %d", enc_config_.effort));
  } else if (key == 'o' || key == 'O') {
    ModifyR(background_, kBackgroundNum, key == 'o');
    const char* const kBackgroundName[] = {"Checkerboard", "Clearly White",
                                           "Just Black", "Not Pink"};
    STATIC_ASSERT_ARRAY_SIZE(kBackgroundName, kBackgroundNum);
    SetMessage(SPrintf("Background: %s", kBackgroundName[background_]));
  } else if (key == 'l') {
    if (SetBitstream(dump_wp2_path_)) {
      AddInfo(SPrintf("Loaded file from '%s'", dump_wp2_path_));
      if (!EncodeAndDecode()) return;  // 'view_only' so just copy.
    } else {
      AddInfo(SPrintf("Couldn't load file from '%s'", dump_wp2_path_));
    }
  } else if (key == 'L') {
    if (output_.empty() ||
        IoUtilWriteFile(output_, dump_wp2_path_,
                        /*overwrite=*/true) != WP2_STATUS_OK) {
      AddInfo(SPrintf("Couldn't save dump file '%s'", dump_wp2_path_));
    } else {
      AddInfo(SPrintf("Saved bits as '%s'", dump_wp2_path_));
    }
  } else if (key == 'M' || key == 'm') {
    enc_config_.use_random_matrix = !enc_config_.use_random_matrix;
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("Use random matrix: %d", enc_config_.use_random_matrix));
#if !defined(_WIN32)
  } else if (key == 'x' || key == 'X') {
    Modify(wp2dbg_value_, 2u, key == 'x');
    setenv("WP2DBG", (wp2dbg_value_ == 0) ? "0" : "1", 1);
    if (!EncodeAndDecode()) return;
    AddInfo(SPrintf("export WP2DBG=%d", wp2dbg_value_));
#endif
  } else if (key == 'f' || key == 'F') {
    if (glutGetModifiers() == GLUT_ACTIVE_ALT) {
      dec_config_.enable_restoration_filter =
          !dec_config_.enable_restoration_filter;
    } else if (glutGetModifiers() == (GLUT_ACTIVE_ALT | GLUT_ACTIVE_SHIFT)) {
      enc_config_.enable_alpha_filter = !enc_config_.enable_alpha_filter;
    } else if (key == 'F') {
      dec_config_.enable_directional_filter =
          !dec_config_.enable_directional_filter;
    } else {
      dec_config_.enable_deblocking_filter =
          !dec_config_.enable_deblocking_filter;
    }
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("Filters: dblk=%s, drct=%s, rstr=%s, alpha=%s",
                       kOnOff[dec_config_.enable_deblocking_filter],
                       kOnOff[dec_config_.enable_directional_filter],
                       kOnOff[dec_config_.enable_restoration_filter],
                       kOnOff[enc_config_.enable_alpha_filter]));
  } else if (key == 'g' || key == 'G') {
    if (key == 'G') {
      dec_config_.grain_amplitude = 0;
    } else {
      ModifyR(dec_config_.grain_amplitude, (uint8_t)101, false, 20);
    }
    enc_config_.store_grain = (dec_config_.grain_amplitude > 0);
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("Grain amplitude: %d", dec_config_.grain_amplitude));
  } else if (key == 'c' || key == 'C') {
    if (!EncodeAndDecode()) return;
    AddInfo(SPrintf("Forcing %u blocks %u segments %u predictors %u transforms",
                    (uint32_t)einfo_.force_partition.size(),
                    (uint32_t)einfo_.force_segment.size(),
                    (uint32_t)einfo_.force_predictor.size(),
                    (uint32_t)einfo_.force_transform.size()));
  } else if (key == 'd') {
    if (!WritePartition(force_partition_, partition_file_path_)) {
      fprintf(stderr, "Error: Unable to write partition file");
    } else {
      AddInfo(SPrintf("Dumped %u blocks", (uint32_t)force_partition_.size()));
    }
  } else if (key == 'D') {
    if (ReadAndConvertPartition(/*read_only=*/true)) {
      AddInfo(SPrintf("Loaded %u blocks", (uint32_t)force_partition_.size()));
    } else {
      AddInfo("Error: Unable to parse partition file");
    }
  } else if (key == '+' || key == '_') {
    const int file = (int)current_file_ + (key == '+' ? 1 : -1);
    if (file >= 0 && file < (int)files_.size()) {
      if (SetCurrentFile(file)) {
        ReshapeWindow();
      } else {
        AddInfo(SPrintf("Unable to read file %d", file));
      }
    } else {
      AddInfo(SPrintf("File %d is not referenced", file));
    }
  } else if (key == 'i' || key == 'I') {
    if (show_ != kInfo) {
      show_ = kInfo;
      show_interface_ = true;
      display_block_ = kDisplayNone;
      if (key == 'I') {
        if (!ComputeYUVDistortion()) return;
      } else {
        yuv_distortion_[0] = 0.f;
        yuv_distortion_[1] = 0.f;
        yuv_distortion_[2] = 0.f;
      }
    } else {
      show_ = kDebug;
    }
    glutPostRedisplay();
  } else if (key == 't' || key == 'T') {
    Modify(enc_config_.segment_id_mode, 3u, key == 't');
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("use_segment_id_mode: %s",
                       kSegmentModes[enc_config_.segment_id_mode]));
  } else if (key == 'n' || key == 'N') {
    if (show_ != kInfo) {
      show_ = kInfo;
      show_interface_ = true;
      display_block_ = kDisplayNone;
    }
    Modify(distortion_metric_, NUM_METRIC_TYPES, key == 'n');
    if (!DecodeOutput()) return;
    glutPostRedisplay();
  } else if (key == 'h') {
    if (show_ != kHelp) {
      show_ = kHelp;
      show_interface_ = true;
      display_block_ = kDisplayNone;
    } else {
      show_ = kDebug;
    }
    glutPostRedisplay();
  } else if (key == 'H') {
    show_interface_ = !show_interface_;
    glutPostRedisplay();
  } else if (key == '1' || key == '!') {
    Modify(enc_config_.csp_type, kNumCspTypes, key == '1');
    if (!EncodeAndDecode()) return;
    SetMessage(
        SPrintf("CSP type: %s", kCSPString[(uint32_t)enc_config_.csp_type]));
  } else if (key == '2' || key == '@') {
    Modify(enc_config_.uv_mode, EncoderConfig::NumUVMode, key == '2');
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("UV Mode: %s", kUVModeString[enc_config_.uv_mode]));
  } else if (key == '3' || key == '#') {
    Modify(enc_config_.tile_shape, NUM_TILE_SHAPES, key == '3');
    if (!EncodeAndDecode()) return;
    SetMessage(
        SPrintf("Tile shape: %s", kTileShapeString[enc_config_.tile_shape]));
  } else if (key == '4' || key == '$') {
    Modify(enc_config_.partition_method, NUM_PARTITION_METHODS, key == '4');
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("Partition method: %s",
                       kPartitionMethodString[enc_config_.partition_method]));
  } else if (key == '5' || key == '%') {
    Modify(enc_config_.partition_set, NUM_PARTITION_SETS, key == '5');
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("Partition set: %s",
                       kPartitionSetString[enc_config_.partition_set]));
  } else if (key == '6' || key == '^') {
    enc_config_.partition_snapping = !enc_config_.partition_snapping;
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("Partition snapping: %s",
                       kOnOff[enc_config_.partition_snapping]));
  } else if (key == '7' || key == '&') {
    enc_config_.segments -= 1;  // 'segments' value is 1-based
    ModifyR(enc_config_.segments, (int)kMaxNumSegments, key == '7');
    enc_config_.segments += 1;
    force_segment_.clear();
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("Num segments: %d", enc_config_.segments));
  } else if (key == '8' || key == '*') {
    enc_config_.sns += (key == '8') ? 7.f : -6.f;
    enc_config_.sns = Clamp(enc_config_.sns, 0.f, 100.f);
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("SNS: %.1f", enc_config_.sns));
  } else if (key == '9' || key == '(') {
    enc_config_.error_diffusion += (key == '9') ? 7 : -6;
    enc_config_.error_diffusion = Clamp(enc_config_.error_diffusion, 0, 100);
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("Error diffusion: %d", enc_config_.error_diffusion));
  } else if (key == '0' || key == ')') {
    enc_config_.tune_perceptual = !enc_config_.tune_perceptual;
    if (!EncodeAndDecode()) return;
    SetMessage(SPrintf("Perceptual: %s", kOnOff[enc_config_.tune_perceptual]));
  } else if (key == 'k' || key == 'K') {
    keep_metadata_ = !keep_metadata_;
    if (in_.metadata_.IsEmpty()) {
      SetMessage("Metadata: none");
    } else {
      if (!EncodeAndDecode()) return;
      SetMessage(keep_metadata_ ? "Metadata: on" : "Metadata: off");
    }
  } else if (key == 'v' || key == 'V') {
    menu_selection_prev_ = menu_selection_;
    if (!SetMenuSelection(GetLeaf(menu_selection_,
                          /*leaf_offset=*/(key == 'v') ? 1 : -1))) return;
  } else if (key == '`') {
    // jump back to previous menu
    std::swap(menu_selection_prev_, menu_selection_);
    if (!SetMenuSelection(menu_selection_)) return;
  } else if (key == 'z' || key == 'Z') {
    zoom_level_ += (key == 'z') ? +1 : -1;  // Zoom in or out.
    ReshapeWindow();
  } else if (key == '?') {
    display_sparks_ = !display_sparks_;
  }
}

void HandleKey(unsigned char key, int pos_x, int pos_y) {
  global_params->HandleKey(key, pos_x, pos_y);
}

void Params::HandleKeyUp(unsigned char key, int pos_x, int pos_y) {
  (void)pos_x;
  (void)pos_y;
  const bool is_alt = (glutGetModifiers() == GLUT_ACTIVE_ALT);
  if (key == ' ' || key == '\\' || key == '|' || key == 13 || key == '\t') {
    show_ = kDebug;
    out_yuv_.Deallocate();
    glutPostRedisplay();
  } else if (key == 'w' || key == 'W') {
    show_ = kDebug;
    webp_.Deallocate();
    glutPostRedisplay();
  } else if (key == 'R' || key == 'r') {
    show_ = kDebug;
    glutPostRedisplay();
  } else if (is_alt && (key == '1' || key == '2')) {
    show_ = kDebug;
    glutPostRedisplay();
  }
}

void HandleKeyUp(unsigned char key, int pos_x, int pos_y) {
  global_params->HandleKeyUp(key, pos_x, pos_y);
}

uint32_t GetSqDist(uint32_t a, uint32_t b) {
  return (a < b) ? ((b - a) * (b - a)) : ((a - b) * (a - b));
}
uint32_t GetDist(const Rectangle& rect, BlockSize size) {
  return GetSqDist(rect.width, BlockWidthPix(size)) +
         GetSqDist(rect.height, BlockHeightPix(size));
}

void Params::HandleMouseMove(int x, int y, bool button_pressed) {
  mouse_x_ = x;
  mouse_y_ = y;
  if (!button_pressed) return;

  if (x != mouse_x_down_ || y != mouse_y_down_) {
    moved_since_last_down_ = true;
  }

  Rectangle* const rect = &forcing_block_;
  if (rect->width > 0 && rect->height > 0) {
    // Drawing a forced block.
    const uint32_t img_x = ToImageX(x), img_y = ToImageY(y);
    const uint32_t img_x_down = ToImageX(mouse_x_down_);
    const uint32_t img_y_down = ToImageY(mouse_y_down_);

    // Maximum block position and size (both relative to top-left).
    const uint32_t max_w = SizeBlocks(width_) * kMinBlockSizePix;
    const uint32_t max_h = SizeBlocks(height_) * kMinBlockSizePix;
    const uint32_t max_x = SafeSub(max_w, kMinBlockSizePix);
    const uint32_t max_y = SafeSub(max_h, kMinBlockSizePix);

    rect->x = std::min(AlignWithBlocks(std::min(img_x_down, img_x)), max_x);
    rect->y = std::min(AlignWithBlocks(std::min(img_y_down, img_y)), max_y);
    rect->width = std::max(img_x_down, img_x) - rect->x + 1;
    rect->height = std::max(img_y_down, img_y) - rect->y + 1;

    // Find the valid block matching the drawn rectangle the most.
    BlockSize closest_size = BLK_4x4;
    const PartitionSet ps = enc_config_.partition_set;
    for (uint32_t i = 0; i < GetNumBlockSizes(ps); ++i) {
      const BlockSize size = GetBlockSizes(ps)[i];
      const uint32_t block_w = BlockWidthPix(size);
      const uint32_t block_h = BlockHeightPix(size);
      uint32_t block_x = rect->x;
      if (img_x < img_x_down) {
        block_x = SafeSub(AlignWithBlocks(img_x_down), block_w);
      }
      uint32_t block_y = rect->y;
      if (img_y < img_y_down) {
        block_y = SafeSub(AlignWithBlocks(img_y_down), block_h);
      }
      const bool is_snapped =
          ((block_x % block_w) == 0 && (block_y % block_h) == 0);
      if (enc_config_.partition_snapping && !is_snapped) continue;

      if (block_x + block_w <= max_w && block_y + block_h <= max_h &&
          block_x + rect->width >= block_w &&
          block_y + rect->height >= block_h &&
          GetDist(*rect, size) < GetDist(*rect, closest_size)) {
        closest_size = size;
      }
    }
    rect->width = BlockWidthPix(closest_size);
    rect->height = BlockHeightPix(closest_size);

    // Replace the rectangle so that the first click (down) is a corner.
    if (img_x < img_x_down) {
      rect->x = SafeSub(AlignWithBlocks(img_x_down), rect->width);
    }
    if (img_y < img_y_down) {
      rect->y = SafeSub(AlignWithBlocks(img_y_down), rect->height);
    }
  } else if (view_zoom_level_ != 0 &&
             !(glutGetModifiers() & GLUT_ACTIVE_SHIFT)) {
    // Moving the view. (Shift is used for selecting areas).
    int x_change =
        (mouse_x_down_ - x) * (int)view_rect_.width / (int)viewport_width_;
    int y_change =
        (mouse_y_down_ - y) * (int)view_rect_.height / (int)viewport_height_;
    view_rect_.x =
        Clamp<int>(view_rect_down_.x + x_change, 0, width_ - view_rect_.width);
    view_rect_.y = Clamp<int>(view_rect_down_.y + y_change, 0,
                              height_ - view_rect_.height);
  }
}

void HandleMouseMove(int x, int y) {
  global_params->HandleMouseMove(x, y, /*button_pressed=*/false);
}

void HandleMouseMoveButtonPressed(int x, int y) {
  global_params->HandleMouseMove(x, y, /*button_pressed=*/true);
}

bool Params::HandleSegmentClick(int button, int state, uint32_t img_x_down,
                                uint32_t img_y_down, uint32_t img_x,
                                uint32_t img_y, int incr) {
  (void)button;
  (void)state;
  (void)img_x_down;
  (void)img_y_down;
  (void)img_x;
  (void)img_y;

#if defined(WP2_BITTRACE)
  if (!IsSegmentVisible() || button == GLUT_LEFT_BUTTON || state != GLUT_DOWN) {
    return false;
  }

  if (!dinfo_.explicit_segment_ids) {
    AddInfo("Cannot force segments at low quality (segment ids are implicit).");
    return true;
  }

  uint32_t num_segments = dinfo_.num_segments;
  // No point in forcing the segment if there's only one...
  if (num_segments == 1) return false;

  uint32_t block_x = 0, block_y = 0;
  const BlockInfo* const b = einfo_.FindBlock(img_x_down, img_y_down);
  if (b != nullptr) {
    block_x = b->rect.x;
    block_y = b->rect.y;
  }

  for (auto& forced_segment : force_segment_) {
    if (forced_segment.x == block_x && forced_segment.y == block_y) {
      if (button == GLUT_MIDDLE_BUTTON) {   // Remove.
        forced_segment = force_segment_.back();
        force_segment_.pop_back();
      } else {
        ModifyInRange(forced_segment.segment_id, num_segments, incr);
      }
      return true;
    }
  }
  if (button == GLUT_RIGHT_BUTTON) {
    EncoderInfo::ForcedSegment forced = {block_x, block_y, 0};
    force_segment_.emplace_back(forced);
    return true;
  }
#endif

  return false;
}

bool Params::HandlePredictorClick(int button, int state, uint32_t img_x_down,
                                  uint32_t img_y_down, uint32_t img_x,
                                  uint32_t img_y, int incr) {
  (void)button;
  (void)state;
  (void)img_x_down;
  (void)img_y_down;
  (void)img_x;
  (void)img_y;

#if defined(WP2_BITTRACE)
  if (!IsPredictorVisible() || button == GLUT_LEFT_BUTTON ||
      state != GLUT_DOWN) {
    return false;
  }
  Channel channel = VDChannel(dec_config_);
  if (channel == kVChannel) channel = kUChannel;

  const BlockInfo* const b = einfo_.FindBlock(img_x_down, img_y_down);
  assert(b != nullptr);

  for (auto& forced_predictor : force_predictor_) {
    if (forced_predictor.x == b->rect.x &&
        forced_predictor.y == b->rect.y &&
        forced_predictor.channel == channel) {
      if (button == GLUT_MIDDLE_BUTTON) {  // remove forced predictor
        forced_predictor = force_predictor_.back();
        force_predictor_.pop_back();
      } else {   // increment / decrement id
        const int num_predictors =
          (channel == kYChannel) ? dinfo_.y_predictors.size() :
          (channel == kAChannel) ? dinfo_.a_predictors.size() :
                                   dinfo_.uv_predictors.size();
        ModifyInRange(forced_predictor.predictor_id, num_predictors, incr);
      }
      return true;
    }
  }
  if (button == GLUT_RIGHT_BUTTON) {
    EncoderInfo::ForcedPredictor forced = {channel, b->rect.x, b->rect.y, 0};
    force_predictor_.emplace_back(forced);
    return true;
  }
#endif

  return false;
}

bool Params::HandleTransformClick(int button, int state, uint32_t img_x_down,
                                  uint32_t img_y_down, uint32_t img_x,
                                  uint32_t img_y, int incr) {
  (void)button;
  (void)state;
  (void)img_x_down;
  (void)img_y_down;
  (void)img_x;
  (void)img_y;

#if defined(WP2_BITTRACE)
  if (!IsTransformVisible() || button == GLUT_LEFT_BUTTON ||
      state != GLUT_DOWN) {
    return false;
  }

  const BlockInfo* const b = einfo_.FindBlock(img_x_down, img_y_down);
  assert(b != nullptr);

  for (auto& forced : force_transform_) {
    if (forced.x == b->rect.x && forced.y == b->rect.y) {
      if (button == GLUT_MIDDLE_BUTTON) {  // Remove.
        forced = force_transform_.back();
        force_transform_.pop_back();
      } else {
        ModifyInRange(forced.transform, kNumTransformPairs, incr);
      }
      return true;
    }
  }
  if (button == GLUT_RIGHT_BUTTON) {
    EncoderInfo::ForcedTransform forced = {b->rect.x, b->rect.y, 0};
    force_transform_.emplace_back(forced);
    return true;
  }
#endif

  return false;
}

bool Params::HandlePartitionClick(int button, int state, uint32_t img_x_down,
                                  uint32_t img_y_down, uint32_t img_x,
                                  uint32_t img_y) {
  if (button != GLUT_RIGHT_BUTTON && button != GLUT_MIDDLE_BUTTON) {
    return false;
  }

  if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
    if (IsPartitionVisible()) {
      // Start drawing a block.
      forcing_block_.x = AlignWithBlocks(img_x_down);
      forcing_block_.y = AlignWithBlocks(img_y_down);
      forcing_block_.width = kMinBlockSizePix;
      forcing_block_.height = kMinBlockSizePix;
    } else {
      forcing_block_ = {0, 0, 0, 0};
    }
    return true;
  }

  const bool delete_only = button == GLUT_MIDDLE_BUTTON;
  if ((forcing_block_.width > 0 && forcing_block_.height > 0 &&
       state == GLUT_UP) ||
      delete_only) {
    bool removed_block = false;
    if (!moved_since_last_down_) {
      // Remove the block under the mouse.
      for (uint32_t i = 0; i < force_partition_.size(); ++i) {
        if (force_partition_[i].Contains(img_x, img_y)) {
          force_partition_[i] = force_partition_.back();
          force_partition_.pop_back();
          removed_block = true;
          break;
        }
      }
    }
    if (!removed_block && !delete_only) {
      // Remove all blocks intersecting the new one.
      for (uint32_t i = 0; i < force_partition_.size();) {
        if (force_partition_[i].Intersects(forcing_block_)) {
          force_partition_[i] = force_partition_.back();
          force_partition_.pop_back();
        } else {
          ++i;
        }
      }
      force_partition_.emplace_back(forcing_block_);
    }
  }
  forcing_block_ = {0, 0, 0, 0};
  return true;
}

bool Params::HandleMenuClick(int button, int state, uint32_t img_x_down,
                             uint32_t img_y_down, uint32_t img_x,
                             uint32_t img_y) {
  if (button != GLUT_LEFT_BUTTON) {
    return false;
  }

  if (state == GLUT_DOWN) {
    einfo_.selection = dinfo_.selection = {img_x, img_y, 1, 1};
    forcing_block_ = {0, 0, 0, 0};
  } else if (state == GLUT_UP) {
    if (show_ == kMenu) {
      show_ = kDebug;
      einfo_.selection = dinfo_.selection = {0, 0, 0, 0};
      forcing_block_ = {0, 0, 0, 0};
      if (!menu_selection_.empty()) {
        // Select the first available leaf (DFS) if not already the case.
        if (!SetMenuSelection(GetLeaf(menu_selection_))) return true;
      }
    } else {
      // Handle area selection (for histogram view).
      if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
        const uint32_t min_x = std::min(img_x_down, img_x);
        const uint32_t min_y = std::min(img_y_down, img_y);
        einfo_.selection = dinfo_.selection = {
            min_x, min_y, std::max(img_x_down, img_x) - min_x + 1,
            std::max(img_y_down, img_y) - min_y + 1};
        forcing_block_ = {0, 0, 0, 0};
      }

      // Refresh vdebugs that might handle mouse 'selection'.
      if (VDMatch(enc_config_, "encoder")) {
        if (!EncodeAndDecode()) return true;
      } else if (VDMatch(dec_config_, "")) {
        if (!DecodeOutput()) return true;
      }
    }
  }
  return true;
}

void Params::HandleMouseClick(int button, int state, int x, int y) {
  mouse_x_ = x;
  mouse_y_ = y;
  if (state == GLUT_DOWN) {
    mouse_x_down_ = x;
    mouse_y_down_ = y;
    moved_since_last_down_ = false;
    view_rect_down_ = view_rect_;
  }
  const uint32_t img_x = ToImageX(x), img_y = ToImageY(y);
  const uint32_t img_x_down = ToImageX(mouse_x_down_);
  const uint32_t img_y_down = ToImageY(mouse_y_down_);

  const int incr = (button == kScrollWheelUp) ? 1 : -1;

  if (HandleMenuClick(button, state, img_x_down, img_y_down, img_x, img_y) ||
      HandleSegmentClick(button, state, img_x_down, img_y_down, img_x, img_y) ||
      HandlePredictorClick(button, state, img_x_down, img_y_down, img_x, img_y,
                           incr) ||
      HandleTransformClick(button, state, img_x_down, img_y_down, img_x, img_y,
                           incr) ||
      HandlePartitionClick(button, state, img_x_down, img_y_down, img_x,
                           img_y)) {
    return;  // Event was handled, nothing more to do.
  }

  if (state == GLUT_DOWN &&
      (button == kScrollWheelUp || button == kScrollWheelDown)) {
    UpdateViewZoomLevel(incr);
  }
}

void Params::HandleMouseWheel(bool is_up, int x, int y) {
  mouse_x_ = x;
  mouse_y_ = y;
  const uint32_t img_x = ToImageX(x), img_y = ToImageY(y);
  const uint32_t img_x_down = ToImageX(mouse_x_down_);
  const uint32_t img_y_down = ToImageY(mouse_y_down_);

  const int incr = is_up ? 1 : -1;
  if (HandlePredictorClick(GLUT_RIGHT_BUTTON, GLUT_DOWN, img_x_down, img_y_down,
                           img_x, img_y, incr) ||
      HandleSegmentClick(GLUT_RIGHT_BUTTON, GLUT_DOWN, img_x_down, img_y_down,
                         img_x, img_y, incr) ||
      HandleTransformClick(GLUT_RIGHT_BUTTON, GLUT_DOWN, img_x_down, img_y_down,
                           img_x, img_y, incr)) {
    return;  // Event was handled, nothing more to do.
  }

  // Change zoom.
  UpdateViewZoomLevel(incr);
}

void HandleMouseClick(int button, int state, int x, int y) {
  global_params->HandleMouseClick(button, state, x, y);
}

void HandleMouseWheel(int button, int dir, int x, int y) {
  // wheel down: dir=-1    up: dir=1
  global_params->HandleMouseWheel(dir > 0, x, y);
}

void Params::SetQuality(float incr) {
  if (glutGetModifiers() & GLUT_ACTIVE_ALT) incr *= 100.;  // Min or max
  const bool set_alpha =
      in_.HasTransparency() && glutGetModifiers() & GLUT_ACTIVE_CTRL;
  float* const quality = set_alpha ? &alpha_quality_ : &quality_;
  const float new_quality = WP2::Clamp(*quality + incr, 0.f, 100.f);
  if (*quality == new_quality) return;
  *quality = new_quality;
  if (!EncodeAndDecode()) return;
  SetMessage(
      SPrintf("%s: %.1f", (set_alpha ? "Alpha quality" : "Quality"), *quality));
}

void HandleSpecialKeys(int key, int pos_x, int pos_y) {
  (void)pos_x, (void)pos_y;
  if (key == GLUT_KEY_UP) return global_params->SetQuality(1);
  if (key == GLUT_KEY_DOWN) return global_params->SetQuality(-1);
  if (key == GLUT_KEY_RIGHT) return global_params->SetQuality(10);
  if (key == GLUT_KEY_LEFT) return global_params->SetQuality(-10);
}

void HandleReshape(int width, int height) {
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  global_params->viewport_width_ = (uint32_t)width;
  global_params->viewport_height_ = (uint32_t)height;
}

//------------------------------------------------------------------------------
// Display

void DisplayCheckerboard(uint32_t width, uint32_t height) {
  static constexpr uint32_t kTileSize = 8;

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0, width, height, 0, -1, 1);  // Vertical flip to start at top-left.

  for (uint32_t top = 0; top < height; top += kTileSize) {
    for (uint32_t left = 0; left < width; left += kTileSize) {
      const GLubyte color = !((top + left) & kTileSize) ? 255 : 200;
      glColor3ub(color, color, color);
      // We do not care about coordinates outside viewport, don't test it.
      glRecti(left, top, left + kTileSize, top + kTileSize);
    }
  }

  glPopMatrix();
}

void DisplayBackground(uint32_t width, uint32_t height,
                       Params::Background background) {
  if (background == Params::kCheckerboard) {
    DisplayCheckerboard(width, height);
    return;
  }

  switch (background) {
    case Params::kWhite:
      glColor3f(1.f, 1.f, 1.f);
      break;
    case Params::kBlack:
      glColor3f(0.f, 0.f, 0.f);
      break;
    case Params::kPink:
      glColor3f(1.f, 0.f, 1.f);
      break;
    default:
      assert(false);
  }
  glRectf(-1.f, -1.f, 1.f, 1.f);
}

void DisplayBuffer(const ArgbBuffer& buffer, Params::Background background) {
  if (buffer.IsEmpty()) return;
  if (buffer.HasTransparency()) {
    DisplayBackground(buffer.width(), buffer.height(), background);
  }

  // convert from src->Argb Argb format to GL's BGRA format.
  std::vector<uint8_t> rgba(4u * buffer.width() * buffer.height());
  const WP2ArgbConverterF convert =
      WP2ConversionFunction(buffer.format(), WP2_RGBA_32);
  for (uint32_t y = 0; y < buffer.height(); ++y) {
    const uint8_t* const src = buffer.GetRow8(y);
    uint8_t* const dst = &rgba[y * 4u * buffer.width()];
    convert(src, buffer.width(), dst);
  }
  glDrawPixels(buffer.width(), buffer.height(), GL_RGBA, GL_UNSIGNED_BYTE,
               reinterpret_cast<const GLvoid*>(&rgba[0]));
}

void DisplayColor(Argb32b color) {
  assert(color.a == 255u);
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0, 1, 1, 0, -1, 1);  // Vertical flip to start at top-left.
  glColor3ub(color.r, color.g, color.b);
  glRecti(0, 0, 1, 1);
  glPopMatrix();
}

void Params::HandleDisplay() {
  if (out_.IsEmpty()) return;
  glPushMatrix();
  glPixelZoom((GLfloat)(+1. / view_rect_.width * viewport_width_),
              (GLfloat)(-1. / view_rect_.height * viewport_height_));
  glRasterPos2f(-1.f, 1.f);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, view_rect_.width);

  if (show_ == kPreviewColor) {
    DisplayColor(preview_color_);
  } else {
    const ArgbBuffer& buffer = GetBuffer();
    ArgbBuffer view(buffer.format());
    const WP2Status status = view.SetView(buffer, view_rect_);
    assert(status == WP2_STATUS_OK);
    (void)status;
    DisplayBuffer(view, background_);
  }

  if (show_interface_) PrintInfo();
  glPopMatrix();
  glutSwapBuffers();
}

void HandleDisplay() { global_params->HandleDisplay(); }

void StartDisplay(const char* window_title, uint32_t width, uint32_t height) {
  const uint32_t swidth = (uint32_t)glutGet(GLUT_SCREEN_WIDTH);
  const uint32_t sheight = (uint32_t)glutGet(GLUT_SCREEN_HEIGHT);

  (void)swidth;
  (void)sheight;
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(width, height);
  glutCreateWindow(window_title);
  glutDisplayFunc(HandleDisplay);
  glutReshapeFunc(HandleReshape);
  glutIdleFunc(nullptr);
  glutKeyboardFunc(HandleKey);
  glutKeyboardUpFunc(HandleKeyUp);
  glutIgnoreKeyRepeat(GL_TRUE);
  glutPassiveMotionFunc(HandleMouseMove);
  glutMotionFunc(HandleMouseMoveButtonPressed);
  glutMouseFunc(HandleMouseClick);
#if defined(WP2_HAVE_FREEGLUT)
  glutMouseWheelFunc(HandleMouseWheel);
#else
  (void)HandleMouseWheel;
#endif
  glutSpecialFunc(HandleSpecialKeys);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);
  glClearColor(0., 0., 0., 1.);
  glClear(GL_COLOR_BUFFER_BIT);
  WP2ArgbConverterInit();
  DisplayLoop(40);
}

//------------------------------------------------------------------------------
// Main

void Help() {
  ProgramOptions opt;
  opt.Add("Visualizer for WebP2 (re-)compression, using OpenGL");
  opt.Add("");
  opt.Add("Usage:");
  opt.Add("  vwp2 in_file [options]");
  opt.Add("");
  opt.Add("Options are:");
  opt.Add("-q <float>", SPrintf("quality factor in (0..100) range, "
                                "from strongest compression to lossless "
                                "(%1.1f by default)",
                                EncoderConfig::kDefault.quality));
  opt.Add("-alpha_q <float>", SPrintf("alpha quality factor in (0..100) range, "
                                      "from strongest compression to lossless "
                                      "(%1.1f by default)",
                                      EncoderConfig::kDefault.alpha_quality));
  opt.Add("-effort <int>", SPrintf("compression effort in (0..9) range, "
                                   "from fastest to slowest/best results "
                                   "(%d by default)",
                                   EncoderConfig::kDefault.effort));
  opt.Add("-pm <int>",
          SPrintf("partition method (0..%d)", NUM_PARTITION_METHODS - 1));
  opt.Add("-ps <int>",
          SPrintf("partition set (0..%d)", NUM_PARTITION_SETS - 1));
  opt.Add("-snap", "force quadtree-like partitioning");
  opt.Add("-sns <float>", "segmentation strength (0=off..100)");
  opt.Add("-uv_mode <int>", "UV subsampling mode (0=Adapt,1=420,2=444,3=Auto)");
  opt.Add("-csp <int>", "color space (0=YCoCg, 1=YCbCr 2=Custom 3=YIQ)");
  opt.Add("-segments <int>", "number of segments ([1..8])");
  opt.Add("-segment_mode [mode]", "one of auto, explicit, implicit");
  opt.Add("-pass <int>", "number of entropy-analysis passes (1..10)");
  opt.Add("-[no_]dblk_filter", "enable/disable deblocking filter.");
  opt.Add("-[no_]drct_filter", "enable/disable directional filter.");
  opt.Add("-notransforms", "disable transforms other than DCT");
  opt.Add("-crop <x> <y> <w> <h>", "crop picture with the given rectangle");
  opt.Add("-set_block <x> <y> <w> <h>", "force one block (in px)");
  opt.Add("-set_partition <file>", "force blocks partition (x y w h per line)");
  opt.Add("-diffusion <int>", "error diffusion strength (0=off..100)");
  opt.Add("-set_dump_path <file>", "path to the dump file (for 'l'/'L' keys)");
  opt.Add("-quants <floats,...>", "force per-segment quantization factors");
  opt.Add("-nomt", "disable multi-threading");
  opt.Add("-[no]metadata", "include [or ignore] source metadata");
  opt.Add("-d", "decode-only: just show the .wp2 file");
  opt.Add("-alt <file>", "alternative image to compare with");
  opt.Add("-vbt <int>", "number of visual bittrace levels");
  opt.Add("-version", "print version number and exit");
  opt.Add("-info", "print info overlay");
  opt.Add("-title <text>", "set a custom title for the window");
  opt.Add("");
  opt.AddMemoryOptionSection();
  // System options.
  opt.AddSystemOptionSection();
  opt.Print();
  // Keyboard options.
  const std::vector<std::string> help = global_params->GetHelp();
  for (const std::string& str : help) printf("%s\n", str.c_str());
}

}  // namespace
}  // namespace WP2

//------------------------------------------------------------------------------

using WP2::ExUtilGetFloat;
using WP2::ExUtilGetFloats;
using WP2::ExUtilGetInt;
using WP2::ExUtilGetUInt;
using WP2::Help;
using WP2::Params;

int main(int argc, char* argv[]) {
  CHECK_TRUE(WP2CheckVersion(), "Error! Library version mismatch!");

  Params params;
  WP2::global_params = &params;
  bool read_initial_partition = false;
  const char* alt_name = nullptr;
  const char* window_title = "WebP2 viewer";
  for (int c = 1; c < argc; ++c) {
    bool parse_error = false;
    if (!strcmp(argv[c], "-h") || !strcmp(argv[c], "-help")) {
      Help();
      return 0;
    } else if (!strcmp(argv[c], "-info")) {
      params.show_ = Params::kInfo;
    } else if (!strcmp(argv[c], "-d")) {
      params.view_only_ = true;
    } else if (!strcmp(argv[c], "-alt") && c + 1 < argc) {
      alt_name = argv[++c];
    } else if (!strcmp(argv[c], "-pm") && c + 1 < argc) {
      params.enc_config_.partition_method =
          (WP2::PartitionMethod)ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-ps") && c + 1 < argc) {
      params.enc_config_.partition_set =
          (WP2::PartitionSet)ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-snap")) {
      params.enc_config_.partition_snapping = true;
    } else if (!strcmp(argv[c], "-sns") && c + 1 < argc) {
      params.enc_config_.sns = ExUtilGetFloat(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-uv_mode") && c + 1 < argc) {
      params.enc_config_.uv_mode =
          (WP2::EncoderConfig::UVMode)ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-csp") && c + 1 < argc) {
      params.enc_config_.csp_type =
          (WP2::Csp)ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-effort") && c + 1 < argc) {
      params.enc_config_.effort = ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-segments") && c + 1 < argc) {
      params.enc_config_.segments = ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-segment_mode") && c + 1 < argc) {
      ++c;
      if (!strcmp(argv[c], "auto")) {
        params.enc_config_.segment_id_mode =
            WP2::EncoderConfig::SEGMENT_ID_AUTO;
      } else if (!strcmp(argv[c], "explicit")) {
        params.enc_config_.segment_id_mode =
            WP2::EncoderConfig::SEGMENT_ID_EXPLICIT;
      } else if (!strcmp(argv[c], "implicit")) {
        params.enc_config_.segment_id_mode =
            WP2::EncoderConfig::SEGMENT_ID_IMPLICIT;
      } else {
        CHECK_TRUE(false, "unsupported segment mode '%s'", argv[c]);
      }
    } else if (!strcmp(argv[c], "-pass") && c + 1 < argc) {
      params.enc_config_.pass = ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-nomt")) {
      params.enc_config_.thread_level = 0;
      params.dec_config_.thread_level = 0;
    } else if (!strcmp(argv[c], "-metadata") ||
               !strcmp(argv[c], "-nometadata")) {
      params.keep_metadata_ = !strcmp(argv[c], "-metadata");
    } else if (!strcmp(argv[c], "-bt") || !strcmp(argv[c], "-BT")) {
      params.bit_trace_ = !strcmp(argv[c], "-bt") ? 1 : 2;
      if (c + 1 < argc) {
        if (isdigit(argv[c + 1][0])) {
          params.bit_trace_level_ = ExUtilGetUInt(argv[++c], &parse_error);
        } else {
          params.bit_trace_level_ = 0;
        }
      }
    } else if (!strcmp(argv[c], "-vbt") && c + 1 < argc) {
      params.visual_bit_trace_level_ = ExUtilGetUInt(argv[++c], &parse_error);
#if defined(WP2_BITTRACE)
      if (params.visual_bit_trace_level_ > WP2::kBitTraceMaxLevels) {
        printf("-vbt must be at most '%d'\n", WP2::kBitTraceMaxLevels);
        parse_error = true;
      }
#else
      printf("WP2_BITTRACE required for -vbt\n");
      parse_error = true;
#endif  // WP2_BITTRACE
    } else if (!strcmp(argv[c], "--") && c + 1 < argc) {
      params.files_.emplace_back(argv[++c]);
    } else if (!strcmp(argv[c], "-q") && c + 1 < argc) {
      params.quality_ = ExUtilGetFloat(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-quants") && c + 1 < argc) {
      ExUtilGetFloats(argv[++c], params.enc_config_.segment_factors,
                      WP2::kMaxNumSegments, &parse_error);
    } else if (!strcmp(argv[c], "-alpha_q") && c + 1 < argc) {
      params.alpha_quality_ = ExUtilGetFloat(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-dblk_filter") ||
               !strcmp(argv[c], "-no_dblk_filter")) {
      params.dec_config_.enable_deblocking_filter =
          !strcmp(argv[c], "-dblk_filter");
    } else if (!strcmp(argv[c], "-drct_filter") ||
               !strcmp(argv[c], "-no_drct_filter")) {
      params.dec_config_.enable_directional_filter =
          !strcmp(argv[c], "-drct_filter");
    } else if (!strcmp(argv[c], "-rstr_filter") ||
               !strcmp(argv[c], "-no_rstr_filter")) {
      params.dec_config_.enable_restoration_filter =
          !strcmp(argv[c], "-rstr_filter");
    } else if (!strcmp(argv[c], "-notransforms")) {
      params.einfo_.disable_transforms = true;
    } else if (!strcmp(argv[c], "-title") && c + 1 < argc) {
      window_title = argv[++c];
    } else if (!strcmp(argv[c], "-crop") && c + 4 < argc) {
      params.crop_ = true;
      params.crop_area_.x = ExUtilGetUInt(argv[++c], &parse_error);
      params.crop_area_.y = ExUtilGetUInt(argv[++c], &parse_error);
      params.crop_area_.width = ExUtilGetUInt(argv[++c], &parse_error);
      params.crop_area_.height = ExUtilGetUInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-premult")) {  // debugging option!
      params.use_premultiplied_ = true;
    } else if (!strcmp(argv[c], "-nopremult")) {  // debugging option!
      params.use_premultiplied_ = false;
    } else if (!strcmp(argv[c], "-set_block") && c + 4 < argc) {
      params.force_partition_.emplace_back(
          ExUtilGetUInt(argv[c + 1], &parse_error),
          ExUtilGetUInt(argv[c + 2], &parse_error),
          ExUtilGetUInt(argv[c + 3], &parse_error),
          ExUtilGetUInt(argv[c + 4], &parse_error));
      c += 4;
    } else if (!strcmp(argv[c], "-set_partition") && c + 1 < argc) {
      params.partition_file_path_ = argv[++c];
      read_initial_partition = true;
    } else if (!strcmp(argv[c], "-diffusion") && c + 1 < argc) {
      params.enc_config_.error_diffusion =
          ExUtilGetInt(argv[++c], &parse_error);
    } else if (!strcmp(argv[c], "-set_dump_path") && c + 1 < argc) {
      const std::string extension = WP2::GetFileExtension(argv[++c]);
      if (extension == "wp2") {
        params.dump_wp2_path_ = argv[c];
      } else if (extension == "png") {
        params.dump_png_path_ = argv[c];
      } else {
        printf("Unhandled extension for -set_dump_path '%s'\n", argv[c]);
        parse_error = true;
      }
    } else if (argv[c][0] == '-') {
      bool must_stop;
      int skip;
      if (WP2::ProgramOptions::ParseSystemOptions(argv[c], &must_stop)) {
        if (must_stop) return 0;
      } else if (WP2::ProgramOptions::ParseMemoryOptions(argv + c, argc - c,
                                                         skip)) {
        c += skip - 1;
      } else {
        printf("Unknown option '%s'\n", argv[c]);
        parse_error = true;
      }
    } else {
      params.files_.emplace_back(argv[c]);
    }

    if (parse_error) {
      Help();
      return 1;
    }
  }

  if (params.files_.empty()) {
    printf("missing input file(s)!!\n");
    Help();
    return 0;
  }

  if (!params.SetCurrentFile(0)) return 1;

  if (read_initial_partition) {
    CHECK_TRUE(params.ReadAndConvertPartition(/*read_only=*/false),
               "Error: Unable to parse partition file");
  }

  if (alt_name != nullptr) {
    CHECK_TRUE(params.SetAltFile(alt_name), "Error: Unable to read alt file");
  }

#if defined(__unix__) || defined(__CYGWIN__)
  // Work around GLUT compositor bug.
  // https://bugs.launchpad.net/ubuntu/+source/freeglut/+bug/369891
  setenv("XLIB_SKIP_ARGB_VISUALS", "1", 1);
#endif

  // Start display (and timer)
  glutInit(&argc, argv);
#if defined(WP2_HAVE_FREEGLUT)
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
#endif
  WP2::StartDisplay(window_title, params.view_rect_.width,
                    params.view_rect_.height);

  glutMainLoop();

  // Should only be reached when using FREEGLUT:
  return 0;
}

#else  // !WP2_HAVE_OPENGL || WP2_REDUCE_BINARY_SIZE

int main(int argc, const char *argv[]) {
#if defined(WP2_REDUCE_BINARY_SIZE)
  fprintf(stderr, "WARNING! Binary compiled using WP2_REDUCE_BINARY_SIZE!\n");
  fprintf(stderr, "All features are disabled in %s!\n", argv[0]);
#else
  fprintf(stderr, "OpenGL support not enabled in %s.\n", argv[0]);
#endif
  (void)argc;
  return 0;
}

#endif

//------------------------------------------------------------------------------
