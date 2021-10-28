// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// -----------------------------------------------------------------------------

// Incremental decoding test function.

#include "./helpers_incr.h"

#include <cstdlib>

#include "./helpers.h"
#include "src/dec/wp2_dec_i.h"
#include "src/utils/orientation.h"
#include "src/wp2/base.h"
#include "src/wp2/decode.h"
#include "src/wp2/format_constants.h"

namespace WP2 {
namespace testutil {
namespace {

// Like assert() but is not silent if NDEBUG is defined (-c opt).
#define CHECK_OR_DIE(cond)     \
  do {                         \
    if (!(cond)) std::abort(); \
  } while (0)

//------------------------------------------------------------------------------

class ExternalCustomDecoder : public CustomDecoder {
 public:
  ExternalCustomDecoder(const DecoderConfig& config, ArgbBuffer* output_buffer)
      : CustomDecoder(config, output_buffer) {}

  void SetData(const uint8_t* data, size_t size) {
    data_ = data;
    size_ = size;
  }

 private:
  void Discard(size_t num_bytes) override { num_discarded_bytes_ += num_bytes; }

  void Reset() override {
    data_ = nullptr;
    size_ = 0;
    num_discarded_bytes_ = 0;
  }

 protected:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  size_t num_discarded_bytes_ = 0;
};

// Invalidates the buffer between CustomDecoder::Read() calls.
class UnstableCustomDecoder : public ExternalCustomDecoder {
 public:
  UnstableCustomDecoder(const DecoderConfig& config, ArgbBuffer* output_buffer)
      : ExternalCustomDecoder(config, output_buffer) {}

  void MakePreviousDataDisappear() {
    if (disappearing_input_.IsEmpty()) return;
    memset(disappearing_input_.bytes, 0, disappearing_input_.size);
    disappearing_input_.Clear();
  }

 private:
  void Fetch(size_t num_requested_bytes, const uint8_t** available_bytes,
             size_t* num_available_bytes) override {
    (void)num_requested_bytes;
    if (num_discarded_bytes_ <= size_) {
      WP2_ASSERT_STATUS(disappearing_input_.CopyFrom(
          data_ + num_discarded_bytes_, size_ - num_discarded_bytes_));
    } else {
      // All bytes were discarded (or more if chunks with corrupted sizes were
      // skipped).
      disappearing_input_.Clear();
    }
    *available_bytes = disappearing_input_.bytes;
    *num_available_bytes = disappearing_input_.size;
  }

  Data disappearing_input_;
};

// Changes the buffer between DataSource::Fetch() calls. Since it only returns
// at most 'num_requested_bytes', the buffer should change between most of the
// calls to TryGetNext() too.
class SwappingCustomDecoder : public ExternalCustomDecoder {
 public:
  SwappingCustomDecoder(const DecoderConfig& config, ArgbBuffer* output_buffer)
      : ExternalCustomDecoder(config, output_buffer) {}

 private:
  void Fetch(size_t num_requested_bytes, const uint8_t** available_bytes,
             size_t* num_available_bytes) override {
    swap(current_input_, previous_input_);
    if (!previous_input_.IsEmpty()) {
      // Make sure previous input is garbage but is not deleted yet.
      memset(previous_input_.bytes, 0, previous_input_.size);
    }

    if (num_discarded_bytes_ <= size_) {
      // Input from two Fetch() calls ago is freed and replaced.
      WP2_ASSERT_STATUS(current_input_.CopyFrom(
          data_ + num_discarded_bytes_,
          std::min(size_ - num_discarded_bytes_, num_requested_bytes)));
    } else {
      // All bytes were discarded (or more if chunks with corrupted sizes were
      // skipped).
      current_input_.Clear();
    }
    *available_bytes = current_input_.bytes;
    *num_available_bytes = current_input_.size;
  }

  Data current_input_;
  Data previous_input_;
};

//------------------------------------------------------------------------------

// Returns true if the decoding can be checked for not too many rows at a time.
bool HasTrueIncrementalOutput(DataView input) {
  ExternalDataSource data_source(input.bytes, input.size);
  BitstreamFeatures features;
  if (DecodeHeader(&data_source, &features) != WP2_STATUS_OK ||
      SkipPreview(&data_source, features) != WP2_STATUS_OK ||
      SkipICC(&data_source, features) != WP2_STATUS_OK) {
    return false;
  }

  AnimationFrame frame;
  uint32_t frame_index = 0;
  do {
    if (DecodeANMF(DecoderConfig::kDefault, &data_source, features, frame_index,
                   &frame) != WP2_STATUS_OK) {
      return false;
    }
    GlobalParams gparams;
    if (DecodeGLBL(&data_source, DecoderConfig::kDefault, features, &gparams) !=
        WP2_STATUS_OK) {
      return false;
    }
    // It would be more accurate to check each tile for GP_BOTH but false
    // positives are not an issue in this file and it would be way longer.
    // TODO(yguyon): Av1 tiles are not incremental yet.
    if (gparams.type_ == GlobalParams::GP_LOSSLESS ||
        gparams.type_ == GlobalParams::GP_BOTH ||
        gparams.type_ == GlobalParams::GP_AV1) {
      return false;
    }

    if (SkipTiles(gparams, &data_source, features, frame.window) !=
        WP2_STATUS_OK) {
      return false;
    }
    ++frame_index;
  } while (!frame.is_last);
  return true;
}

//------------------------------------------------------------------------------

class IncrementalDecodingTester {
 public:
  IncrementalDecodingTester(const DecoderConfig& config, DataView input,
                            const IncrementalDecodingTestSetup& setup)
      : config_(config),
        input_(input),
        incr_size_step_(std::max((size_t)1, setup.incr_size_step)),
        setup_(setup),
        output_(setup.output_format),
        incremental_output_(output_.format()),
        array_idec_(config, &output_),
        stream_idec_(config, &output_),
        unstable_custom_idec_(config, &output_),
        swapping_custom_idec_(config, &output_),
        has_true_incremental_output_(HasTrueIncrementalOutput(input)) {}

  WP2Status Decode(std::vector<ArgbBuffer>* const decoded_frames,
                   std::vector<uint32_t>* const decoded_durations_ms) {
    Decoder& idec = GetDecoder();
    size_t available_input_size = 0;
    uint32_t current_action_index = 0;
    bool decoded_a_full_frame = false;

    // Start, continue, rewind decoding as long as there is something to do.
    while (available_input_size < input_.size ||
           idec.GetStatus() == WP2_STATUS_NOT_ENOUGH_DATA ||
           current_action_index < setup_.actions.size()) {
      WP2_CHECK_OK(!idec.Failed(), idec.GetStatus());
      size_t previous_input_size = available_input_size;
      available_input_size =
          std::min(available_input_size + incr_size_step_, input_.size);

      // Rewind or skip frames as planned.
      ApplyActions(decoded_a_full_frame, &available_input_size,
                   &previous_input_size, &current_action_index, decoded_frames,
                   decoded_durations_ms);

      // Update the input.
      if (setup_.decoder_type == DecoderType::kArray) {
        array_idec_.SetInput(input_.bytes, available_input_size);
      } else if (setup_.decoder_type == DecoderType::kStream) {
        stream_idec_.AppendInput(input_.bytes + previous_input_size,
                                 available_input_size - previous_input_size,
                                 /*data_is_persistent=*/true);
      } else if (setup_.decoder_type == DecoderType::kUnstableCustom) {
        unstable_custom_idec_.SetData(input_.bytes, available_input_size);
      } else {
        swapping_custom_idec_.SetData(input_.bytes, available_input_size);
      }

      // Decode all available bytes until a non-skipped frame is ready.
      uint32_t duration_ms = 0;
      decoded_a_full_frame = idec.ReadFrame(&duration_ms);
      WP2_CHECK_OK(!idec.Failed(), idec.GetStatus());

      if (setup_.decoder_type == DecoderType::kUnstableCustom) {
        // Exercise with an invalid buffer until Fetch() is called (during the
        // next ReadFrame() call).
        unstable_custom_idec_.MakePreviousDataDisappear();
      }

      // Check the state of the decoder and the output.
      if (idec.TryGetDecodedFeatures() != nullptr) {
        WP2_CHECK_STATUS(CheckAnimation(decoded_a_full_frame));
        if (!idec.TryGetDecodedFeatures()->is_animation && !rewinded_) {
          CHECK_OR_DIE(idec.GetNumFrameDecodedFeatures() <= 1u);
        }
      }
      CheckDecodedArea(available_input_size, previous_input_size);
      if (idec.GetDecodedArea().GetArea() > 0) {
        // Only check new pixels to avoid timeouts when parsing a large image at
        // every byte.
        Rectangle top, bottom, left, right;
        GetDecoder().GetDecodedArea().Exclude(previous_decoded_area_, &top,
                                              &bottom, &left, &right);
        CheckIncrementalOutput(top, bottom, left, right);
        SaveIncrementalOutput(top, bottom, left, right);
      }
      previous_frame_index_ = frame_index_;
      if (setup_.compare_with_single_chunk) {
        CheckSameOutputAsSingleChunk(available_input_size, decoded_a_full_frame,
                                     duration_ms);
      }
      if (decoded_a_full_frame) {
        WP2_CHECK_STATUS(CheckFullFrame(available_input_size,
                                        previous_input_size, duration_ms));
        SaveFullFrame(duration_ms, decoded_frames, decoded_durations_ms);
        if (!idec.TryGetFrameDecodedFeatures(frame_index_)->is_last) {
          ++frame_index_;
          previous_frame_index_ = kMaxNumFrames;
          previous_decoded_area_ = Rectangle(0, 0, 0, 0);
        }
      }

      if (current_action_index >= setup_.actions.size() &&
          previous_input_size >= input_.size && !decoded_a_full_frame) {
        // Nothing will happen further so avoid an infinite loop.
        WP2_CHECK_STATUS(idec.GetStatus());
      }
    }
    return WP2_STATUS_OK;
  }

 private:
  // Rewind and/or skip frames according to the test 'setup_'.
  void ApplyActions(bool decoded_a_full_frame,
                    size_t* const available_input_size,
                    size_t* const previous_input_size,
                    uint32_t* const current_action_index,
                    std::vector<ArgbBuffer>* const decoded_frames,
                    std::vector<uint32_t>* const decoded_durations_ms) {
    while (*current_action_index < setup_.actions.size() &&
           (setup_.actions[*current_action_index].bistream_position <=
                *available_input_size ||
            *previous_input_size >= input_.size)) {
      const DecoderAction& action = setup_.actions[*current_action_index];
      if (action.type == DecoderAction::Type::kRewind ||
          action.type == DecoderAction::Type::kRewindKeepBytes) {
        const bool output_buffer_changed = (action.value > 0);
        if (output_buffer_changed) output_.Deallocate();
        output_.metadata_.Clear();  // For CheckSameOutputAsSingleChunk()
        GetDecoder().Rewind();
        *previous_input_size = 0;
        if (action.type != DecoderAction::Type::kRewindKeepBytes) {
          *available_input_size = 0;
        }
        frame_index_ = 0;
        previous_frame_index_ = kMaxNumFrames;
        previous_decoded_area_ = Rectangle(0, 0, 0, 0);
        incremental_output_.Deallocate();
        if (decoded_frames != nullptr) decoded_frames->clear();
        if (decoded_durations_ms != nullptr) decoded_durations_ms->clear();
        rewinded_ = true;
        num_skipped_frames_ = 0;
        decoded_a_full_frame = false;
      } else if (action.type == DecoderAction::Type::kSkip) {
        const uint32_t num_frames_to_skip = action.value;
        GetDecoder().SkipNumNextFrames(num_frames_to_skip);
        if (num_frames_to_skip >= 1) {
          // The first skipped frame just after a decoded frame is that same
          // frame so it does not count.
          if (!decoded_a_full_frame || num_frames_to_skip >= 2) {
            frame_index_ = std::min(frame_index_ + action.value, kMaxNumFrames);
          }
          previous_frame_index_ = kMaxNumFrames;
          previous_decoded_area_ = Rectangle(0, 0, 0, 0);
          incremental_output_.Deallocate();
          num_skipped_frames_ += num_frames_to_skip;
          decoded_a_full_frame = false;
        }
      }
      ++*current_action_index;
    }
  }

  // Contains the checks done on the features of each frame.
  static void CheckFrameFeatures(const FrameFeatures* const frame_features,
                                 uint32_t frame_index,
                                 const ArgbBuffer& output) {
    CHECK_OR_DIE(frame_features != nullptr);
    CHECK_OR_DIE(frame_features->last_dispose_frame_index <= frame_index);
    if (frame_index == 0 ||
        frame_features->last_dispose_frame_index == frame_index) {
      // Disposed frame or not an animation.
      CHECK_OR_DIE(frame_features->window.x == 0u);
      CHECK_OR_DIE(frame_features->window.y == 0u);
      if (!output.IsEmpty()) {
        CHECK_OR_DIE(frame_features->window.width == output.width());
        CHECK_OR_DIE(frame_features->window.height == output.height());
      }
    } else {
      CHECK_OR_DIE(frame_features->window.GetArea() > 0u);
      if (!output.IsEmpty()) {
        CHECK_OR_DIE(frame_features->window.x + frame_features->window.width <=
                     output.width());
        CHECK_OR_DIE(frame_features->window.y + frame_features->window.height <=
                     output.height());
      }
    }
  }

  // Contains the checks related to animated images.
  WP2Status CheckAnimation(bool decoded_a_full_frame) const {
    const Decoder& idec = GetDecoder();
    const uint32_t num_known_frames = idec.GetNumFrameDecodedFeatures();
    if (num_skipped_frames_ == 0 || decoded_a_full_frame) {
      CHECK_OR_DIE(idec.GetCurrentFrameIndex() == frame_index_);
      if (!rewinded_) {
        CHECK_OR_DIE(frame_index_ <
                     num_known_frames + (decoded_a_full_frame ? 0 : 1));
      }
    }
    for (uint32_t i = 0; i < num_known_frames; ++i) {
      CheckFrameFeatures(idec.TryGetFrameDecodedFeatures(i), i, output_);

      size_t offset;  // Offset should be known if features are.
      CHECK_OR_DIE(idec.TryGetFrameLocation(i, &offset));
      if (i < num_known_frames - 1) {
        size_t length;  // Length should be known for previous frames.
        CHECK_OR_DIE(idec.TryGetFrameLocation(i, nullptr, &length));
      }

      if (i > 0) {
        size_t previous_offset, previous_length;
        CHECK_OR_DIE(idec.TryGetFrameLocation(i - 1, &previous_offset,
                                                  &previous_length));
        CHECK_OR_DIE(offset == previous_offset + previous_length);
      }
    }

    if (idec.GetStatus() == WP2_STATUS_OK) {
      // Everything is decoded without error.
      for (uint32_t i = 0; i < num_known_frames; ++i) {
        const bool is_last = idec.TryGetFrameDecodedFeatures(i)->is_last;
        CHECK_OR_DIE(is_last == (i == num_known_frames - 1u));

        size_t offset, length;
        CHECK_OR_DIE(idec.TryGetFrameLocation(i, &offset, &length));
        WP2_CHECK_OK(
            (idec.TryGetDecodedFeatures()->has_trailing_data || !is_last)
                ? (offset + length < input_.size)
                : (offset + length == input_.size),
            WP2_STATUS_BITSTREAM_ERROR);
      }
    }
    return WP2_STATUS_OK;
  }

  // Verifies that the currently decoded area makes sense (only increasing etc.)
  void CheckDecodedArea(size_t available_input_size,
                        size_t previous_input_size) const {
    const Rectangle decoded_area = GetDecoder().GetDecodedArea();
    const BitstreamFeatures* const features =
        GetDecoder().TryGetDecodedFeatures();
    if (features != nullptr) {
      if (!output_.IsEmpty()) {
        CHECK_OR_DIE(features->width == output_.width());
        CHECK_OR_DIE(features->height == output_.height());
      }
      const Rectangle unoriented =
          OrientateRectangle(GetInverseOrientation(features->orientation),
                             features->width, features->height, decoded_area);
      CHECK_OR_DIE(unoriented.x == 0u);
      CHECK_OR_DIE(unoriented.y == 0u);
    }
    CHECK_OR_DIE(decoded_area.width >= previous_decoded_area_.width);
    CHECK_OR_DIE(decoded_area.width <= output_.width());
    CHECK_OR_DIE(decoded_area.height >= previous_decoded_area_.height);
    CHECK_OR_DIE(decoded_area.height <= output_.height());

    // Check that incremental decoding is not outputting all pixels at once.
    // Unfortunately this is hard to check for animations (border, preframes).
    if (frame_index_ == previous_frame_index_ &&
        available_input_size >= previous_input_size &&
        available_input_size - previous_input_size <= 2 &&
        features != nullptr && !features->is_animation &&
        config_.incremental_mode ==
            DecoderConfig::IncrementalMode::PARTIAL_TILE_CONTEXT) {
      const uint32_t num_lines =
          (features->raw_width != features->width)
              ? decoded_area.width - previous_decoded_area_.width
              : decoded_area.height - previous_decoded_area_.height;
      // Incremental decoding should not be able to finish more than two rows
      // of blocks in 2 bytes or fewer for lossy and a tile for lossless.
      uint32_t max_num_lines =
          std::min(features->raw_height, has_true_incremental_output_
                                             ? kMaxBlockSizePix * 2
                                             : features->tile_height);
      // Filters are applied sequentially and each needs a few rows untouched by
      // the previous one, delaying the pixel availability until the last row.
      if (config_.enable_deblocking_filter || config_.enable_alpha_filter) {
        max_num_lines += kMaxBlockSizePix;
      }
      if (config_.enable_directional_filter) {
        max_num_lines += kDrctFltSize + kDrctFltTapDist;
      }
      if (config_.enable_restoration_filter) {
        max_num_lines += kWieFltHeight;
      }
      if (num_lines > max_num_lines) {
        DumpImage(output_, "/tmp/dump.png", decoded_area);
        DumpData(input_.bytes, input_.size, "/tmp/dump.wp2");
      }
      CHECK_OR_DIE(num_lines <= max_num_lines);
    }
  }

  // Contains the checks related to partial frame decoding.
  void CheckIncrementalOutput(Rectangle top, Rectangle bottom, Rectangle left,
                              Rectangle right) const {
    CHECK_OR_DIE(GetDecoder().TryGetDecodedFeatures() != nullptr);
    // Make sure there is no transparent pixel in the incremental output if the
    // bitstream is declared as opaque.
    if (GetDecoder().TryGetDecodedFeatures()->is_opaque) {
      for (Rectangle rect : {top, bottom, left, right}) {
        if (rect.GetArea() == 0) continue;
        ArgbBuffer view(output_.format());
        CHECK_OR_DIE(view.SetView(output_, rect) == WP2_STATUS_OK);
        CHECK_OR_DIE(!view.HasTransparency());
      }
    }
  }

  // Stores the newly decoded pixels at each incremental decoding step for later
  // verification.
  void SaveIncrementalOutput(Rectangle top, Rectangle bottom, Rectangle left,
                             Rectangle right) {
    if (incremental_output_.IsEmpty()) {
      CHECK_OR_DIE(GetDecoder().TryGetDecodedFeatures() != nullptr);
      CHECK_OR_DIE(incremental_output_.Resize(
                       GetDecoder().TryGetDecodedFeatures()->width,
                       GetDecoder().TryGetDecodedFeatures()->height) ==
                   WP2_STATUS_OK);
    }

    // Copy newly decoded pixels.
    for (Rectangle rect : {top, bottom, left, right}) {
      if (rect.GetArea() == 0) continue;
      ArgbBuffer from_view(output_.format());
      ArgbBuffer to_view(incremental_output_.format());
      CHECK_OR_DIE(from_view.SetView(output_, rect) == WP2_STATUS_OK);
      CHECK_OR_DIE(to_view.SetView(incremental_output_, rect) == WP2_STATUS_OK);
      CHECK_OR_DIE(to_view.CopyFrom(from_view) == WP2_STATUS_OK);
    }
    previous_decoded_area_ = GetDecoder().GetDecodedArea();
  }

  // Verifies that the output image and features are the same whether the
  // bitstream is decoded as a single whole chunk or incrementally.
  void CheckSameOutputAsSingleChunk(size_t available_input_size,
                                    bool decoded_a_full_frame,
                                    uint32_t duration_ms) const {
    ArgbBuffer cmp_output(output_.format());
    ArrayDecoder cmp_idec(config_, &cmp_output);
    cmp_idec.SetInput(input_.bytes, available_input_size);

    // To correctly compare the bool output of ReadFrame(), the next expected
    // frame must be the same, so skip what is needed.
    cmp_idec.SkipNumNextFrames(std::max(frame_index_, num_skipped_frames_));

    uint32_t cmp_duration_ms = 0;
    bool cmp_decoded_a_full_frame = cmp_idec.ReadFrame(&cmp_duration_ms);
    CHECK_OR_DIE(!cmp_idec.Failed());

    if (cmp_decoded_a_full_frame && !decoded_a_full_frame &&
        GetDecoder().TryGetFrameDecodedFeatures(frame_index_) != nullptr &&
        GetDecoder().TryGetFrameDecodedFeatures(frame_index_)->is_last) {
      // The current frame was previously decoded and it is the last one:
      // nothing is expected.
      cmp_decoded_a_full_frame = cmp_idec.ReadFrame();  // Should be false.
    }

    CHECK_OR_DIE(
        HasSameData(cmp_output.metadata_.iccp, output_.metadata_.iccp) &&
        HasSameData(cmp_output.metadata_.xmp, output_.metadata_.xmp) &&
        HasSameData(cmp_output.metadata_.exif, output_.metadata_.exif));

    CHECK_OR_DIE(cmp_decoded_a_full_frame == decoded_a_full_frame);
    if (decoded_a_full_frame) {
      CHECK_OR_DIE(cmp_duration_ms == duration_ms);
    }

    const Rectangle decoded_area = GetDecoder().GetDecodedArea();
    CHECK_OR_DIE(cmp_idec.GetDecodedArea() == decoded_area);
    if (decoded_area.GetArea() > 0) {
      ArgbBuffer cmp_view(cmp_output.format()), view(output_.format());
      CHECK_OR_DIE(cmp_view.SetView(cmp_output, decoded_area) == WP2_STATUS_OK);
      CHECK_OR_DIE(view.SetView(output_, decoded_area) == WP2_STATUS_OK);
      CHECK_OR_DIE(Compare(cmp_view, view, "single_chunk_compare"));
    }
  }

  // Contains the checks done when a frame is completely decoded.
  WP2Status CheckFullFrame(size_t available_input_size,
                           size_t previous_input_size,
                           uint32_t duration_ms) const {
    const Rectangle decoded_area = GetDecoder().GetDecodedArea();
    CHECK_OR_DIE(decoded_area.width == output_.width());
    CHECK_OR_DIE(decoded_area.height == output_.height());
    // Compare incremental data and the current canvas.
    CHECK_OR_DIE(
        Compare(incremental_output_, output_, "incrementally decoded"));

    if (GetDecoder().TryGetDecodedFeatures()->is_animation) {
      CHECK_OR_DIE(
          GetDecoder().TryGetFrameDecodedFeatures(frame_index_)->duration_ms ==
          duration_ms);
    } else if (!GetDecoder().TryGetDecodedFeatures()->has_trailing_data &&
               !rewinded_) {
      // If a still image without trailing metadata is decoded without
      // rewinding, the remaining bytes imply an invalid bitstream.
      WP2_CHECK_OK(available_input_size == input_.size,
                   WP2_STATUS_BITSTREAM_ERROR);
    }

    // A frame was just decoded. Verify its location in the bitstream is valid.
    // If a single new byte was made available (no rewind), the frame must
    // exactly end at the current number of available bytes.
    size_t offset, length;
    CHECK_OR_DIE(
        GetDecoder().TryGetFrameLocation(frame_index_, &offset, &length));
    if (incr_size_step_ == 1 && !rewinded_) {
      CHECK_OR_DIE(available_input_size == offset + length);
    } else {
      CHECK_OR_DIE(available_input_size >= offset + length);
    }
    return WP2_STATUS_OK;
  }

  // Stores each completely decoded frame for later comparison.
  void SaveFullFrame(uint32_t duration_ms,
                     std::vector<ArgbBuffer>* const decoded_frames,
                     std::vector<uint32_t>* const decoded_durations_ms) const {
    if (decoded_frames != nullptr) {
      decoded_frames->push_back(ArgbBuffer());
      CHECK_OR_DIE(decoded_frames->back().ConvertFrom(output_) ==
                   WP2_STATUS_OK);
    }
    if (GetDecoder().TryGetDecodedFeatures()->is_animation &&
        decoded_durations_ms != nullptr) {
      decoded_durations_ms->push_back(duration_ms);
    }
  }

  // Convenient accessors
  const Decoder& GetDecoder() const {
    if (setup_.decoder_type == DecoderType::kArray) return array_idec_;
    if (setup_.decoder_type == DecoderType::kStream) return stream_idec_;
    if (setup_.decoder_type == DecoderType::kUnstableCustom) {
      return unstable_custom_idec_;
    }
    return swapping_custom_idec_;
  }
  Decoder& GetDecoder() {
    if (setup_.decoder_type == DecoderType::kArray) return array_idec_;
    if (setup_.decoder_type == DecoderType::kStream) return stream_idec_;
    if (setup_.decoder_type == DecoderType::kUnstableCustom) {
      return unstable_custom_idec_;
    }
    return swapping_custom_idec_;
  }

  const DecoderConfig& config_;
  const DataView input_;
  const size_t incr_size_step_;
  const IncrementalDecodingTestSetup& setup_;

  ArgbBuffer output_, incremental_output_;
  ArrayDecoder array_idec_;
  StreamDecoder stream_idec_;
  UnstableCustomDecoder unstable_custom_idec_;
  SwappingCustomDecoder swapping_custom_idec_;

  uint32_t frame_index_ = 0;
  uint32_t previous_frame_index_ = kMaxNumFrames;
  Rectangle previous_decoded_area_ = {0, 0, 0, 0};
  bool rewinded_ = false;
  uint32_t num_skipped_frames_ = 0;
  const bool has_true_incremental_output_;
};

}  // namespace

//------------------------------------------------------------------------------

WP2Status DecodeIncremental(const DecoderConfig& config, DataView input,
                            const IncrementalDecodingTestSetup& setup,
                            std::vector<ArgbBuffer>* const decoded_frames,
                            std::vector<uint32_t>* const decoded_durations_ms) {
  IncrementalDecodingTester tester(config, input, setup);
  return tester.Decode(decoded_frames, decoded_durations_ms);
}

//------------------------------------------------------------------------------

#undef CHECK_OR_DIE

}  // namespace testutil
}  // namespace WP2
