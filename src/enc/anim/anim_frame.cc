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
//  Animation encoding functions.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <cassert>

#include "./anim_enc.h"
#include "src/common/constants.h"
#include "src/common/header_enc_dec.h"
#include "src/enc/wp2_enc_i.h"
#include "src/wp2/format_constants.h"

namespace WP2 {

//------------------------------------------------------------------------------

bool AllFramesAreOpaque(const Vector<Frame>& frames) {
  for (const Frame& frame : frames) {
    if (frame.ccsp_pixels.IsEmpty() ? frame.rgb_pixels.HasTransparency()
                                    : frame.ccsp_pixels.HasAlpha()) {
      return false;
    }
  }
  return true;
}

static_assert(GetMaxVarIntLength(0, kMaxFrameDurationMs) <
                  GetMaxVarIntLength(0, kMaxFrameDurationMs + 1),
              "kMaxFrameDurationMs is not optimized to fit in a varint");

WP2Status WriteANMF(uint32_t image_width, uint32_t image_height,
                    uint32_t duration, const Rectangle& window, bool dispose,
                    bool blend, bool is_last, Writer* const output) {
  assert(image_width <= kImageDimMax && image_height <= kImageDimMax);
  assert(window.width > 0 && window.x + window.width <= image_width);
  assert(window.height > 0 && window.y + window.height <= image_height);
  // If it's the last frame, it can't be a preframe.
  if (is_last) assert(duration > 0);

  uint8_t header[kANMFMaxHeaderSize];
  BitPacker enc(header, sizeof(header), "anmf_chunk");
  const uint32_t tag = dispose ? kANMFTagDispose : kANMFTagFrame;
  enc.PutBits(tag, kANMFTagNumBits, "anmf_tag");
  enc.PutBits(blend ? 1 : 0, 1, "blend");
  enc.PutBits(is_last ? 1 : 0, 1, "is_last");
  enc.PutVarUInt(duration, 0, kMaxFrameDurationMs, "duration");
  enc.PutRect(window, image_width, image_height, "window");
  enc.Align();
  assert(enc.Ok());
  assert(enc.Used() >= kANMFMinHeaderSize && enc.Used() <= kANMFMaxHeaderSize);
  WP2_CHECK_OK(output->Append(header, enc.Used()), WP2_STATUS_BAD_WRITE);
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

uint32_t Frame::GetWidth() const {
  return ccsp_pixels.IsEmpty() ? rgb_pixels.width() : ccsp_pixels.GetWidth();
}
uint32_t Frame::GetHeight() const {
  return ccsp_pixels.IsEmpty() ? rgb_pixels.height() : ccsp_pixels.GetHeight();
}

Rectangle Subframe::GetFrameRect() const {
  return {
      0, 0, ccsp_pixels.IsEmpty() ? rgb_pixels.width() : ccsp_pixels.GetWidth(),
      ccsp_pixels.IsEmpty() ? rgb_pixels.height() : ccsp_pixels.GetHeight()};
}

//------------------------------------------------------------------------------

FrameEncoder::FrameEncoder(const EncoderConfig& config,
                           Argb38b background_color,
                           const Vector<Frame>& frames,
                           const ProgressRange& progress)
    : config_(config),
      frames_(frames),
      image_has_alpha_(!AllFramesAreOpaque(frames)),
      background_color_(ToArgb32b(background_color)),
      subframe_encoder_(GetQualityHint(config.quality), image_has_alpha_,
                        FinalTileShape(config)),
      progress_(progress, 1.f) {}

WP2Status FrameEncoder::SaveSubframeToCanvas(const ArgbBuffer& subframe,
                                             const Rectangle& window) {
  // A mix of color spaces is currently unsupported.
  WP2_CHECK_OK(ccsp_canvas_.IsEmpty(), WP2_STATUS_UNSUPPORTED_FEATURE);

  if (rgb_canvas_.IsEmpty()) {
    assert(window.width == subframe.width() &&
           window.height == subframe.height());
    assert(window.x == 0 && window.y == 0);

    WP2_CHECK_STATUS(rgb_canvas_.CopyFrom(subframe));
  } else {
    assert(subframe.width() == rgb_canvas_.width());
    assert(subframe.height() == rgb_canvas_.height());

    const uint32_t pixel_depth = WP2FormatBpp(rgb_canvas_.format());
    uint8_t* canvas_row =
        rgb_canvas_.GetRow8(window.y) + window.x * pixel_depth;
    const uint8_t* frame_row =
        subframe.GetRow8(window.y) + window.x * pixel_depth;

    for (uint32_t y = 0; y < window.height; ++y) {
      memcpy(canvas_row, frame_row, window.width * pixel_depth);
      canvas_row += rgb_canvas_.stride();
      frame_row += subframe.stride();
    }
  }
  return WP2_STATUS_OK;
}

WP2Status FrameEncoder::SaveFrameToCanvas(
    const ArgbBuffer& frame, const RectangleGroup& different_pixels) {
  for (uint32_t r = 0; r < different_pixels.GetNumRectangles(); ++r) {
    WP2_CHECK_STATUS(
        SaveSubframeToCanvas(frame, different_pixels.GetRectangle(r)));
  }
  return WP2_STATUS_OK;
}

WP2Status FrameEncoder::GetPixelsDifferentFromCanvas(
    const ArgbBuffer& frame, RectangleGroup* const different_pixels) const {
  // A mix of color spaces is currently unsupported.
  WP2_CHECK_OK(ccsp_canvas_.IsEmpty(), WP2_STATUS_UNSUPPORTED_FEATURE);

  assert(frame.format() == rgb_canvas_.format());
  assert(frame.width() == rgb_canvas_.width() &&
         frame.height() == rgb_canvas_.height());

  // As quality lowers, pixels tend to be less different from previous frame.
  // TODO(yguyon): Refine 15, make a constant etc.
  const double threshold = (1. - config_.quality * 0.01) * 15.;
  const uint32_t squared_threshold =
      (uint32_t)std::lround(threshold * threshold);

  const uint32_t pixel_depth = WP2FormatBpp(rgb_canvas_.format());
  const uint8_t* canvas_row = rgb_canvas_.GetRow8(0);
  const uint8_t* frame_row = frame.GetRow8(0);

  for (uint32_t y = 0; y < frame.height(); ++y) {
    const uint8_t* canvas_pixel = canvas_row;
    const uint8_t* frame_pixel = frame_row;
    for (uint32_t x = 0; x < frame.width();
         ++x, canvas_pixel += pixel_depth, frame_pixel += pixel_depth) {
      uint32_t squared_diff = 0;
      for (uint32_t c = 0; c < pixel_depth; ++c) {
        const int32_t diff = (int32_t)canvas_pixel[c] - frame_pixel[c];
        squared_diff += (uint32_t)(diff * diff);
      }
      if (squared_diff > squared_threshold) {
        different_pixels->AddPoint(x, y);
      }
    }
    canvas_row += rgb_canvas_.stride();
    frame_row += frame.stride();
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status FrameEncoder::SaveSubframeToCanvas(const YUVPlane& subframe,
                                             const CSPMtx& ccsp_to_rgb,
                                             const Rectangle& window) {
  // A mix of color spaces is currently unsupported.
  WP2_CHECK_OK(rgb_canvas_.IsEmpty(), WP2_STATUS_UNSUPPORTED_FEATURE);

  if (ccsp_canvas_.IsEmpty()) {
    assert(window.width == subframe.GetWidth() &&
           window.height == subframe.GetHeight());
    assert(window.x == 0 && window.y == 0);

    WP2_CHECK_STATUS(ccsp_canvas_.Copy(subframe, /*resize_if_needed=*/true));
    if (ccsp_canvas_.A.IsEmpty()) {
      // Fill alpha plane as opaque in case a later (sub)frame is not.
      // TODO(yguyon): Do it lazily
      WP2_CHECK_STATUS(ccsp_canvas_.A.Resize(ccsp_canvas_.GetWidth(),
                                             ccsp_canvas_.GetHeight()));
      ccsp_canvas_.A.Fill(kAlphaMax);
    }
    ccsp_to_rgb_ = ccsp_to_rgb;
  } else {
    assert(subframe.GetWidth() == ccsp_canvas_.GetWidth());
    assert(subframe.GetHeight() == ccsp_canvas_.GetHeight());
    assert(ccsp_to_rgb == ccsp_to_rgb_);

    for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
      Plane16 from_view, to_view;
      WP2_CHECK_STATUS(
          to_view.SetView(ccsp_canvas_.GetChannel(channel), window));
      if (channel == kAChannel && subframe.GetChannel(channel).IsEmpty()) {
        to_view.Fill(kAlphaMax);
      } else {
        WP2_CHECK_STATUS(
            from_view.SetView(subframe.GetChannel(channel), window));
        WP2_CHECK_STATUS(to_view.Copy(from_view, /*resize_if_needed=*/false));
      }
    }
  }
  return WP2_STATUS_OK;
}

WP2Status FrameEncoder::SaveFrameToCanvas(
    const YUVPlane& frame, const CSPMtx& ccsp_to_rgb,
    const RectangleGroup& different_pixels) {
  for (uint32_t r = 0; r < different_pixels.GetNumRectangles(); ++r) {
    WP2_CHECK_STATUS(SaveSubframeToCanvas(frame, ccsp_to_rgb,
                                          different_pixels.GetRectangle(r)));
  }
  return WP2_STATUS_OK;
}

WP2Status FrameEncoder::GetPixelsDifferentFromCanvas(
    const YUVPlane& frame, const CSPMtx& ccsp_to_rgb,
    RectangleGroup* const different_pixels) const {
  // A mix of color spaces is currently unsupported.
  WP2_CHECK_OK(rgb_canvas_.IsEmpty(), WP2_STATUS_UNSUPPORTED_FEATURE);

  assert(frame.GetWidth() == ccsp_canvas_.GetWidth() &&
         frame.GetHeight() == ccsp_canvas_.GetHeight());
  assert(ccsp_to_rgb == ccsp_to_rgb_);

  // As quality lowers, pixels tend to be less different from previous frame.
  // TODO(yguyon): Adapt it to custom color space and precision
  const double threshold = (1. - config_.quality * 0.01) * 15.;
  const uint32_t squared_threshold =
      (uint32_t)std::lround(threshold * threshold);

  for (uint32_t y = 0; y < frame.GetHeight(); ++y) {
    for (uint32_t x = 0; x < frame.GetWidth(); ++x) {
      uint32_t squared_diff = 0;
      for (Channel channel : {kYChannel, kUChannel, kVChannel, kAChannel}) {
        const int32_t canvas_pixel = ccsp_canvas_.GetChannel(channel).At(x, y);
        const int32_t frame_pixel =
            (channel == kAChannel && frame.GetChannel(channel).IsEmpty())
                ? kAlphaMax
                : frame.GetChannel(channel).At(x, y);
        // Clamp to avoid the unlikely overflow of (max diff 16b)^2 * 4.
        const int32_t diff = ClampToSigned(canvas_pixel - frame_pixel,
                                           sizeof(squared_diff) * 8 - 3);
        squared_diff += (uint32_t)(diff * diff);
      }
      if (squared_diff > squared_threshold) different_pixels->AddPoint(x, y);
    }
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status FrameEncoder::EncodeSubframe(const Frame& frame, bool dispose,
                                       bool is_preframe,
                                       bool is_previous_preframe,
                                       const Rectangle& window,
                                       const ProgressRange& progress,
                                       Writer* const output) const {
  if (is_previous_preframe) assert(!dispose);
  EncoderConfig config = config_;
  if (frame.force_lossless) {
    config.quality = 100;
  }

  const bool is_last = (!is_preframe && (&frame == &frames_.back()));

  Subframe subframe;
  if (!frame.rgb_pixels.IsEmpty()) {
    WP2_CHECK_STATUS(subframe.rgb_pixels.SetView(frame.rgb_pixels));
  }
  if (!frame.ccsp_pixels.IsEmpty()) {
    WP2_CHECK_STATUS(subframe.ccsp_pixels.SetView(frame.ccsp_pixels));
    subframe.ccsp_to_rgb = frame.ccsp_to_rgb;
  }
  subframe.duration_ms = is_preframe ? 0 : frame.duration_ms;
  subframe.dispose = dispose;
  subframe.window = window;
  subframe.blend = false;
  WP2_CHECK_STATUS(subframe_encoder_.EncodeSubframe(config, subframe, is_last,
                                                    progress, output));
  return WP2_STATUS_OK;
}

WP2Status FrameEncoder::EncodeFrame(const Frame& frame, bool dispose,
                                    const RectangleGroup& different_pixels,
                                    const ProgressRange& progress,
                                    Writer* const output) const {
  assert(different_pixels.GetNumRectangles() >= 1);

  bool is_previous_preframe = false;
  for (uint32_t r = 0; r < different_pixels.GetNumRectangles(); ++r) {
    const ProgressRange subframe_progress(
        progress, 1. / different_pixels.GetNumRectangles());
    const bool is_preframe = (r < different_pixels.GetNumRectangles() - 1);
    const bool dispose_this_frame = (dispose && r == 0);  // Only the first one.
    WP2_CHECK_STATUS(EncodeSubframe(
        frame, dispose_this_frame, is_preframe, is_previous_preframe,
        different_pixels.GetRectangle(r), subframe_progress, output));
    is_previous_preframe = is_preframe;
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status FrameEncoder::Encode(Writer* const output) {
  const uint32_t max_num_preframes =
      DivRound<uint32_t>(kMaxNumPreframes * config_.effort, kMaxEffort);

  // One more for the regular frame following the preframes.
  RectangleGroup different_pixels(max_num_preframes + 1);
  RectangleGroup bounded_pixels(1);  // Cropped frame.
  RectangleGroup all_pixels(1);      // Entire frame.

  bool dispose = true;  // First frame is always disposing the canvas.

  uint32_t last_dispose_index = 0;
  for (uint32_t i = 0; i < frames_.size(); ++i) {
    const ProgressRange frame_progress(progress_, 1. / frames_.size());
    const Frame& frame = frames_[i];
    WP2_CHECK_STATUS(
        SetupEncoderInfo(frame.GetWidth(), frame.GetHeight(), config_));

    different_pixels.Clear();
    if (dispose) {
      different_pixels.SetRectangle(
          {0, 0, frame.GetWidth(), frame.GetHeight()});
    } else {
      if (frame.rgb_pixels.IsEmpty()) {
        WP2_CHECK_STATUS(GetPixelsDifferentFromCanvas(
            frame.ccsp_pixels, frame.ccsp_to_rgb, &different_pixels));
      } else {
        WP2_CHECK_STATUS(
            GetPixelsDifferentFromCanvas(frame.rgb_pixels, &different_pixels));
      }

      if (different_pixels.GetNumRectangles() == 0) {
        // Identical frame, store it anyway with just the top left pixel.
        // TODO(yguyon): Increase previous frame's duration (decoder setting?)
        different_pixels.SetRectangle({0, 0, 1, 1});
      } else {
        // TODO(yguyon): Merge close rectangles, to avoid slowly trying it below
        // TODO(yguyon): Check if background color is good enough for areas
        //               outside 'different_pixels' and dispose if it's the case
        // TODO(yguyon): Blend?
      }
    }

    const RectangleGroup* subframes_kept = &different_pixels;

    if (different_pixels.GetNumRectangles() == 1 &&
        different_pixels.GetRectangle(0).width == frame.GetWidth() &&
        different_pixels.GetRectangle(0).height == frame.GetHeight()) {
      // Completely different frame, store it as disposing "for free".
      dispose = true;
      // There is only one choice, encode directly to 'output'.
      WP2_CHECK_STATUS(EncodeFrame(frame, dispose, different_pixels,
                                   frame_progress, output));
    } else {
      // Try different encoding patterns and keep the smallest bitstream.
      const ProgressRange as_is_progress(frame_progress, 0.4);
      const ProgressRange merged_progress(frame_progress, 0.3);
      const ProgressRange dispose_progress(frame_progress, 0.3);

      // Keep the current settings.
      MemoryWriter writer_as_is;
      WP2_CHECK_STATUS(EncodeFrame(frame, dispose, different_pixels,
                                   as_is_progress, &writer_as_is));
      const MemoryWriter* writer_kept = &writer_as_is;

      // Merge the areas (if more than one) into their bounding rectangle.
      MemoryWriter writer_merged;
      if (different_pixels.GetNumRectangles() > 1) {
        bounded_pixels.SetRectangle(different_pixels.GetBoundingRectangle());
        WP2_CHECK_STATUS(EncodeFrame(frame, dispose, bounded_pixels,
                                     merged_progress, &writer_merged));
        if (writer_merged.size_ < writer_kept->size_) {
          writer_kept = &writer_merged;
          subframes_kept = &bounded_pixels;
        }
      } else {
        WP2_CHECK_STATUS(merged_progress.AdvanceBy(1.));
      }

      // Most of the time, in lossy, cropping a frame does not take less space
      // because it quantizes less.
      // TODO(yguyon): Adapt quality setting depending on cropped area instead
      //               or on top of doing the following:
      // Store the entire frame (and thus dispose the canvas prior to it).
      MemoryWriter writer_dispose;
      all_pixels.SetRectangle({0, 0, frame.GetWidth(), frame.GetHeight()});
      WP2_CHECK_STATUS(EncodeFrame(frame, /*dispose=*/true, all_pixels,
                                   dispose_progress, &writer_dispose));
      if (writer_dispose.size_ < writer_kept->size_) {
        writer_kept = &writer_dispose;
        subframes_kept = &all_pixels;
        dispose = true;
      }
      WP2_CHECK_OK(output->Append(writer_kept->mem_, writer_kept->size_),
                   WP2_STATUS_BAD_WRITE);
    }

    if (dispose) last_dispose_index = i;

    if (i + 1 < frames_.size()) {
      // Check if the next frame is maybe disposing to know if the current frame
      // must be saved to the canvas.
      dispose = (frames_[i + 1].force_dispose || config_.effort == 0 ||
                 i + 1 - last_dispose_index > kMaxNumDependentFrames);
      if (!dispose) {
        if (frame.rgb_pixels.IsEmpty()) {
          WP2_CHECK_STATUS(SaveFrameToCanvas(
              frame.ccsp_pixels, frame.ccsp_to_rgb, *subframes_kept));
        } else {
          WP2_CHECK_STATUS(
              SaveFrameToCanvas(frame.rgb_pixels, *subframes_kept));
        }
      }
    }
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

SubframeEncoder::SubframeEncoder(uint32_t quality_hint, bool image_has_alpha,
                                 TileShape tile_shape)
    : quality_hint_(quality_hint),
      tile_shape_(tile_shape),
      image_has_alpha_(image_has_alpha) {
  assert(tile_shape != TILE_SHAPE_AUTO);
}

WP2Status SubframeEncoder::EncodeSubframe(const EncoderConfig& config,
                                          const Subframe& subframe,
                                          bool is_last,
                                          const ProgressRange& progress,
                                          Writer* const output) const {
  WP2_CHECK_OK(subframe.GetFrameRect().Contains(subframe.window),
               WP2_STATUS_INVALID_PARAMETER);
  WP2_CHECK_OK(FinalTileShape(config) == tile_shape_,
               WP2_STATUS_INVALID_PARAMETER);
  if (!image_has_alpha_) {
    WP2_CHECK_OK(
        !(subframe.ccsp_pixels.IsEmpty() ? subframe.rgb_pixels.HasTransparency()
                                         : subframe.ccsp_pixels.HasAlpha()),
        WP2_STATUS_INVALID_PARAMETER);
  }
  WP2_CHECK_STATUS(
      WriteANMF(subframe.GetFrameRect().width, subframe.GetFrameRect().height,
                subframe.duration_ms, subframe.window, subframe.dispose,
                subframe.blend, is_last, output));
  WP2_CHECK_STATUS(progress.AdvanceBy(kOnePercent));  // ANMF

  ArgbBuffer rgb_subframe(subframe.rgb_pixels.format());

  const GlobalParams::Type type = DecideGlobalParamsType(config);
  const bool yuv_is_needed =
      (type == GlobalParams::GP_LOSSY || type == GlobalParams::GP_BOTH);
  const bool rgb_is_needed =
      (type == GlobalParams::GP_LOSSLESS || type == GlobalParams::GP_BOTH ||
       type == GlobalParams::GP_AV1 ||
       (yuv_is_needed &&
        config.csp_type == Csp::kCustom)  // For CSPTransform::Optimize()
      );

  YUVPlane yuv_subframe;
  CSPTransform csp_transform;
  if (rgb_is_needed) {
    if (subframe.rgb_pixels.IsEmpty()) {
      assert(!subframe.ccsp_pixels.IsEmpty());
      YUVPlane ccsp_subframe;
      WP2_CHECK_STATUS(
          ccsp_subframe.SetView(subframe.ccsp_pixels, subframe.window));
      WP2_CHECK_STATUS(ccsp_subframe.Export(
          subframe.ccsp_to_rgb, /*resize_if_needed=*/true, &rgb_subframe));
    } else {
      WP2_CHECK_STATUS(
          rgb_subframe.SetView(subframe.rgb_pixels, subframe.window));
    }
  }
  if (yuv_is_needed) {
    WP2_CHECK_STATUS(csp_transform.Init(config.csp_type, rgb_subframe));
    if (subframe.ccsp_pixels.IsEmpty()) {
      ArgbBuffer view(subframe.rgb_pixels.format());
      WP2_CHECK_STATUS(view.SetView(subframe.rgb_pixels, subframe.window));
      WP2_CHECK_STATUS(
          yuv_subframe.Import(view, view.HasTransparency(), csp_transform,
                              /*resize_if_needed=*/true, /*pad=*/kPredWidth));
    } else {
      YUVPlane ccsp_subframe;
      WP2_CHECK_STATUS(
          ccsp_subframe.SetView(subframe.ccsp_pixels, subframe.window));
      WP2_CHECK_STATUS(yuv_subframe.Import(
          ccsp_subframe, subframe.ccsp_to_rgb, csp_transform,
          /*resize_if_needed=*/true, /*pad=*/kPredWidth));
    }
  }

  WP2_CHECK_STATUS(EncodeTiles(subframe.window.width, subframe.window.height,
                               rgb_subframe, yuv_subframe, csp_transform,
                               config, quality_hint_, image_has_alpha_,
                               progress, output));
  return WP2_STATUS_OK;
}

}  // namespace WP2
