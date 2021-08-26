// Copyright 2021 Google LLC
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

#include "./avif.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>
#include <string>

#include "./av1_obu.h"
#include "./bit_packing.h"
#include "./container.h"

namespace container {

//------------------------------------------------------------------------------

namespace {

// ISO/IEC 23008-12 HEIC "irot" + "imir" boxes to Container::Orientation values.
// MIAF Section 7.3.6.7 says to apply rotation, then mirror.
constexpr Orientation kOrientationMapping[4][3] = {
    // No rotation.
    {Orientation::kOriginal,          // No mirroring.
     Orientation::kFlipTopToBottom,   // Mirror top to bottom.
     Orientation::kFlipLeftToRight},  // Mirror left to right.
    // 90 degrees counter-clockwise.
    {Orientation::k270Clockwise,          // No mirroring.
     Orientation::kFlipBotLeftToTopRgt,   // Mirror top to bottom.
     Orientation::kFlipTopLeftToBotRgt},  // Mirror left to right.
    // 180 degrees.
    {Orientation::k180,               // No mirroring.
     Orientation::kFlipLeftToRight,   // Mirror top to bottom.
     Orientation::kFlipTopToBottom},  // Mirror left to right.
    // 270 degrees counter-clockwise.
    {Orientation::k90Clockwise,           // No mirroring.
     Orientation::kFlipTopLeftToBotRgt,   // Mirror top to bottom.
     Orientation::kFlipBotLeftToTopRgt},  // Mirror left to right.
};

// Container::Orientation values to HEIC "irot" + "imir" boxes.
void GetIrotImir(Orientation orientation, uint8_t& irot, uint8_t& imir) {
  for (irot = 0; irot < 4; ++irot) {
    for (imir = 0; imir < 3; ++imir) {
      if (kOrientationMapping[irot][imir] == orientation) return;
    }
  }
  assert(false);
}

// Big-endian unsigned integer coding of 'value'.
// 'output' must be at least 'num_bytes'-long.
void WriteBigEndian(uint32_t value, uint8_t output[], uint32_t num_bytes = 1) {
  for (uint32_t i = 0; i < num_bytes; ++i) {
    output[i] = (value >> ((num_bytes - i - 1) * 8)) & 0xFFu;
  }
}

// 'input' must be at least 'num_bytes'-long.
uint32_t ReadBigEndian(const uint8_t input[], uint32_t num_bytes = 1) {
  uint32_t value = 0;
  for (uint32_t i = 0; i < num_bytes; ++i) {
    value = (value << 8) | input[i];
  }
  return value;
}

// Returns true if the four characters in 'fourcc' match the 'str'.
bool EqualFourCC(const uint8_t fourcc[4], const char str[]) {
  return std::equal(fourcc, fourcc + 4, str);
}

//------------------------------------------------------------------------------

// ISOBMFF box writer that uses the dtor to finalize the encoding.
class BoxWriter {
 public:
  BoxWriter(const char fourcc[4], Data& output)
      : output_(output), box_offset_(output.size()) {
    Write<uint32_t>(8);  // size placeholder, replaced in dtor
    WriteStr(fourcc);    // box tag
  }
  ~BoxWriter() {
    const size_t box_size = output_.size() - box_offset_;
    WriteBigEndian(box_size, output_.data() + box_offset_, sizeof(uint32_t));
  }

  // Appends 'value'. CodedType is used only for its number of bits and must be
  // specified explicitly.
  template <typename CodedType>
  void Write(uint32_t value) {
    output_.resize(output_.size() + sizeof(CodedType));
    WriteBigEndian(value, output_.data() + (output_.size() - sizeof(CodedType)),
                   sizeof(CodedType));
  }

  // Appends 'data_size' bytes of 'data'.
  void WriteBytes(const uint8_t data[], size_t data_size) {
    output_.insert(output_.end(), data, data + data_size);
  }

  // Appends 'str'. 'append_0' includes the null-terminating character \0.
  void WriteStr(const char str[], bool append_0 = false) {
    const size_t str_size = std::strlen(str) + (append_0 ? 1 : 0);
    output_.reserve(output_.size() + str_size);
    for (uint32_t i = 0; i < str_size; ++i) {
      output_.emplace_back(static_cast<uint8_t>(str[i]));
    }
  }

  // Appends the 32 bits of additional header needed by some ISOBMFF boxes.
  void WriteFullBoxHeader(uint8_t version = 0, uint32_t flags = 0) {
    Write<uint32_t>((version << 24) | flags);
  }

 private:
  Data& output_;
  const size_t box_offset_;  // Position of the 4 bytes for box size.
};

}  // namespace

//------------------------------------------------------------------------------

bool WrapAvif(const Image& image, const uint8_t codec_bytes[],
              size_t num_codec_bytes, Data& data) {
  // Decode the necessary information from the Sequence Header. AVIF requires
  // some of these fields to match between the MIAF container and AV1-OBU.
  SequenceHeaderObu header;
  if (!UnwrapObuSequenceHeader(codec_bytes, num_codec_bytes, header)) {
    return false;
  }

  // Position in the bitstream of the encoded value of the offset of the codec
  // payload within the bitstream.
  size_t offset_of_codec_bytes_offset_within_file;

  {  // ISO/IEC 14496-12 - 4.3.2 File Type Box Syntax
    BoxWriter ftyp("ftyp", data);   // FileTypeBox
    ftyp.WriteStr("avif");          // major_brand
    ftyp.Write<uint32_t>(0);        // minor_version
    ftyp.WriteStr("avifmif1miaf");  // compatible_brands
    // Valid AVIF files shall specify the "avif", "miaf" and "mif1" brands.
    // https://aomediacodec.github.io/av1-avif/#file-constraints requires
    // "ISO/IEC 23000-22 - 7.2.1 Box-level requirements on image items" needing
    // "ISO/IEC 23008-12 - 10.2.1.1 "mif1" structural brand file requirements".

    // Baseline Profile "MA1B" is only for Main Profile and Level at most 5.1.
    // https://aomediacodec.github.io/av1-avif/#baseline-profile
    // Advanced Profile "MA1A" is only for High Profile and Level at most 6.0.
    // https://aomediacodec.github.io/av1-avif/#advanced-profile
    // See "AV1 Bitstream & Decoding Process Specification - Annex A".
    if (header.seq_profile == 0 && header.seq_level_idx_0 <= 13) {
      ftyp.WriteStr("MA1B");  // compatible_brands
    } else if (header.seq_profile == 1 && header.seq_level_idx_0 <= 16) {
      ftyp.WriteStr("MA1A");  // compatible_brands
    }
  }

  {  // ISO/IEC 14496-12 - 8.11.1.2 Meta Box Syntax
    BoxWriter meta("meta", data);  // MetaBox, mandatory because "mif1"
    meta.WriteFullBoxHeader();

    {  // ISO/IEC 14496-12 - 8.4.3.2 Handler Reference Box Syntax
      BoxWriter hdlr("hdlr", data);  // HandlerBox, mandatory because "mif1"
      hdlr.WriteFullBoxHeader();
      hdlr.Write<uint32_t>(0);               // pre_defined
      hdlr.WriteStr("pict");                 // handler_type
      hdlr.Write<uint32_t>(0);               // reserved
      hdlr.Write<uint32_t>(0);               // reserved
      hdlr.Write<uint32_t>(0);               // reserved
      hdlr.WriteStr("", /*append_0=*/true);  // name of the library used for
                                             // encoding (unknown)
    }

    {  // ISO/IEC 14496-12 - 8.11.4.2 Primary Item Box Syntax
      // Note: According to "ISO/IEC 14496-12-8.11.1.2", the "pitm" box should
      //       be just after "hdlr", but not in "ISO/IEC 23008-12-10.2.1.1".
      BoxWriter pitm("pitm", data);  // PrimaryItemBox, mandatory because "mif1"
      pitm.WriteFullBoxHeader();
      pitm.Write<uint16_t>(1);  // item_ID
    }

    {  // ISO/IEC 14496-12 - 8.11.3.2 Item Location Box Syntax
      BoxWriter iloc("iloc", data);  // ItemLocationBox, mandatory bc "mif1"
      iloc.WriteFullBoxHeader();
      iloc.Write<uint8_t>((0x0 << 4) | 0x4);  // offset_size, length_size
      iloc.Write<uint8_t>((0x4 << 4) | 0x0);  // base_offset_size, reserved
      iloc.Write<uint16_t>(1);                // item_count
      iloc.Write<uint16_t>(1);                // item_ID
      iloc.Write<uint16_t>(0);                // data_reference_index
      offset_of_codec_bytes_offset_within_file = data.size();  // For later.
      iloc.Write<uint32_t>(0);                // base_offset (unknown yet)
      iloc.Write<uint16_t>(1);                // extent_count
      iloc.Write<uint32_t>(num_codec_bytes);  // extent_length
    }

    {  // ISO/IEC 14496-12 - 8.11.6.2 Item Information Box Syntax
      BoxWriter iinf("iinf", data);  // ItemInfoBox, mandatory because "mif1"
      iinf.WriteFullBoxHeader();
      const uint32_t entry_count = 1;
      iinf.Write<uint16_t>(entry_count);

      for (uint32_t i = 0; i < entry_count; ++i) {
        BoxWriter infe("infe", data);  // ItemInfoEntry, mandatory bc "mif1"
        infe.WriteFullBoxHeader(/*version=*/2);
        infe.Write<uint16_t>(i + 1);                // item_ID
        infe.Write<uint16_t>(0);                    // item_protection_index
        infe.WriteStr("av01");                      // item_type
        infe.WriteStr("Image", /*append_0=*/true);  // item_name
      }
    }

    {  // ISO/IEC 23008-12 - 9.3.2 Item Properties Box Syntax
      BoxWriter iprp("iprp", data);  // ItemPropertiesBox, mandatory bc "mif1"

      uint32_t num_properties = 0;  // Needed by "ipma".
      {
        BoxWriter ipco("ipco", data);  // ItemPropertyContainerBox

        {  // "av1C" is mandatory for any "infe" item of type "av01".
          // See https://aomediacodec.github.io/av1-avif/ 2.2.1
          // and https://aomediacodec.github.io/av1-isobmff/ 2.3.3
          BoxWriter ispe("av1C", data);  // AV1CodecConfigurationBox

          // The values in "av1C" should be equal to the ones defined in the OBU
          // Sequence Header.
          BitPacker ispe_fields;
          ispe_fields.EncodeUInt(1, 1);  // marker
          ispe_fields.EncodeUInt(1, 7);  // version
          ispe_fields.EncodeUInt(header.seq_profile, 3);
          ispe_fields.EncodeUInt(header.seq_level_idx_0, 5);
          ispe_fields.EncodeBool(false);  // seq_tier_0
          ispe_fields.EncodeBool(header.high_bitdepth);
          ispe_fields.EncodeBool(header.twelve_bit);
          ispe_fields.EncodeBool(header.mono_chrome);
          ispe_fields.EncodeUInt(header.subsampling_x, 1);
          ispe_fields.EncodeUInt(header.subsampling_y, 1);
          ispe_fields.EncodeUInt(header.chroma_sample_position, 2);
          ispe_fields.EncodeUInt(0, 3);  // reserved
          ispe_fields.EncodeUInt(0, 1);  // initial_presentation_delay_present
          ispe_fields.EncodeUInt(0, 4);  // reserved
          if (ispe_fields.GetNumBits() != 32) return false;
          ispe.WriteBytes(ispe_fields.GetBytes(), ispe_fields.GetNumBits() / 8);

          // "Sequence Header OBUs should not be present in the
          // AV1CodecConfigurationBox" so leave configOBUs empty.
          ++num_properties;
        }

        {
          BoxWriter ispe("ispe", data);  // ImageSpatialExtentsProperty
          ispe.WriteFullBoxHeader();
          ispe.Write<uint32_t>(image.width);
          ispe.Write<uint32_t>(image.height);
          ++num_properties;
        }

        uint8_t rotation, mirror;
        GetIrotImir(image.orientation, rotation, mirror);
        if (rotation != 0) {
          BoxWriter irot("irot", data);  // ImageRotation
          irot.Write<uint8_t>(rotation);
          ++num_properties;
        }
        if (mirror != 0) {
          BoxWriter imir("imir", data);  // ImageMirror
          imir.Write<uint8_t>(mirror - 1);
          ++num_properties;
        }

        if (image.format != Format::kUnknown) {
          BoxWriter pixi("pixi", data);  // PixelInformationProperty
          pixi.WriteFullBoxHeader();
          pixi.Write<uint8_t>(3);  // Number of channels.
          const uint8_t bit_depth = (image.format == Format::kARGB8) ? 8 : 10;
          pixi.Write<uint8_t>(bit_depth);
          pixi.Write<uint8_t>(bit_depth);
          pixi.Write<uint8_t>(bit_depth);
          ++num_properties;
        }
      }

      {
        BoxWriter ipma("ipma", data);  // ItemPropertyAssociation
        ipma.WriteFullBoxHeader();
        ipma.Write<uint32_t>(1);              // entry_count
        ipma.Write<uint16_t>(1);              // item_ID
        ipma.Write<uint8_t>(num_properties);  // association_count
        for (uint32_t i = 0; i < num_properties; ++i) {
          ipma.Write<uint8_t>(0x80 | (i + 1));  // 1b essential + 7b index
        }
      }
    }
  }

  {
    BoxWriter mdat("mdat", data);  // MediaDataBox

    // Write in the "iloc" box where the 'codec_bytes' begin in the file.
    WriteBigEndian(data.size(), &data[offset_of_codec_bytes_offset_within_file],
                   sizeof(uint32_t));

    // Append the 'codec_bytes'.
    data.insert(data.end(), codec_bytes, codec_bytes + num_codec_bytes);
  }
  return true;
}

//------------------------------------------------------------------------------

namespace {

// Minimal AVIF parsing to extract a few images features.
// Returns false in case of error.
bool ParseBoxes(const uint8_t data[], size_t data_size, Image& image,
                bool& saw_ftyp, const uint8_t*& codec_bytes,
                size_t& num_codec_bytes, uint32_t& angle, uint32_t& mirror) {
  size_t position = 0;
  while (position + 4 <= data_size) {
    const size_t size = ReadBigEndian(data + position, sizeof(uint32_t));
    if (size < 8 || position + size > data_size) return false;
    const uint8_t* const fourcc = data + position + 4;

    const bool has_fullbox_header = EqualFourCC(fourcc, "meta") ||
                                    EqualFourCC(fourcc, "ispe") ||
                                    EqualFourCC(fourcc, "pixi");
    const size_t offset = (has_fullbox_header ? 12 : 8);
    if (size < offset) return false;
    const uint8_t* content = data + position + offset;
    const size_t content_size = size - offset;

    if (EqualFourCC(fourcc, "ftyp")) {
      if (size < 12) return false;
      if (!EqualFourCC(content, "avif")) return false;
      saw_ftyp = true;
    } else if (EqualFourCC(fourcc, "meta") || EqualFourCC(fourcc, "iprp") ||
               EqualFourCC(fourcc, "ipco")) {
      if (!ParseBoxes(content, content_size, image, saw_ftyp, codec_bytes,
                      num_codec_bytes, angle, mirror)) {
        return false;
      }
    } else if (EqualFourCC(fourcc, "irot")) {
      if (content_size < 1) return false;
      angle = ReadBigEndian(content);
      if (angle > 3) return false;
    } else if (EqualFourCC(fourcc, "imir")) {
      if (content_size < 1) return false;
      mirror = 1 + ReadBigEndian(content);
      if (mirror > 2) return false;
    } else if (EqualFourCC(fourcc, "ispe")) {
      if (content_size < 4 + 4) return false;
      image.width = ReadBigEndian(content + 0, 4);
      image.height = ReadBigEndian(content + 4, 4);
    } else if (EqualFourCC(fourcc, "pixi")) {
      if (content_size < 1) return false;
      const uint8_t num_channels = ReadBigEndian(content + 0);
      if (content_size < 1 + num_channels) return false;
      for (uint32_t i = 0; i < num_channels; ++i) {
        const uint8_t bit_depth = ReadBigEndian(content + 1 + i);
        if (bit_depth == ReadBigEndian(content + 1) && bit_depth == 8) {
          image.format = Format::kARGB8;
        } else if (bit_depth == ReadBigEndian(content + 1) && bit_depth == 10) {
          image.format = Format::kAYUV10;
        } else {
          return false;
        }
      }
    } else if (EqualFourCC(fourcc, "stss")) {
      // TODO(yguyon): Read frame durations
    } else if (EqualFourCC(fourcc, "mdat")) {
      codec_bytes = content;
      num_codec_bytes = content_size;
    } else {
      // Skip any unknown box.
    }
    position += size;
  }
  return (position == data_size);  // No trailing garbage allowed.
}

}  // namespace

bool UnwrapAvif(const uint8_t data[], size_t data_size, Image& image,
                size_t& offset_till_codec_bytes, size_t& num_codec_bytes) {
  image = Image();
  bool saw_ftyp = false;
  offset_till_codec_bytes = num_codec_bytes = 0;
  uint32_t angle = 0, mirror = 0;

  const uint8_t* codec_bytes = nullptr;
  if (!ParseBoxes(data, data_size, image, saw_ftyp, codec_bytes,
                  num_codec_bytes, angle, mirror)) {
    return false;
  }
  if (!saw_ftyp || num_codec_bytes == 0) return false;
  offset_till_codec_bytes = codec_bytes - data;

  image.orientation = kOrientationMapping[angle][mirror];
  image.is_opaque = true;  // TODO(yguyon): Handle alpha
  return true;
}

//------------------------------------------------------------------------------

namespace {

// Parses the 'data' and returns a string containing all ISOBMFF box.
std::string BmffBoxesToStr(const uint8_t data[], size_t data_size,
                           uint32_t indent) {
  std::stringstream ss;
  for (size_t position = 0; position + 4 <= data_size;) {
    for (uint32_t i = 0; i < indent; ++i) ss << "  ";

    const size_t box_size = ReadBigEndian(data + position, sizeof(uint32_t));
    if (box_size < 8 || position + box_size > data_size) {
      ss << "Bad box size (" << box_size << " bytes)" << std::endl;
      break;
    }
    const uint8_t* const fourcc = data + position + 4;
    const char* const box = reinterpret_cast<const char*>(fourcc);
    ss << "Box " << box[0] << box[1] << box[2] << box[3] << " (" << box_size
       << " bytes at offset " << position << ")" << std::endl;

    if (box_size > 8) {
      if (EqualFourCC(fourcc, "meta") && box_size > 12) {
        for (uint32_t i = 0; i < indent + 1; ++i) ss << "  ";
        ss << "Version " << static_cast<uint32_t>(data[position + 8])
           << ", flags " << static_cast<uint32_t>(data[position + 9])
           << static_cast<uint32_t>(data[position + 10])
           << static_cast<uint32_t>(data[position + 11]) << std::endl;
        ss << BmffBoxesToStr(data + position + 12, box_size - 12, indent + 1);
      } else if (EqualFourCC(fourcc, "iprp") || EqualFourCC(fourcc, "ipco")) {
        ss << BmffBoxesToStr(data + position + 8, box_size - 8, indent + 1);
      } else {
        for (uint32_t i = 0; i <= indent; ++i) ss << "  ";
        // Display the first few dozens of characters.
        for (uint32_t i = 0; i < std::min(box_size - 8, (size_t)72); ++i) {
          const char c = reinterpret_cast<const char*>(data + position + 8)[i];
          // Display a printable character or an interrogation mark.
          ss << ((c >= ' ' && c <= '~') ? c : '?');
        }
        if (box_size - 8 > 72) ss << "...";  // Indicate missing characters.
        ss << std::endl;
      }
    }
    position += box_size;
  }
  return ss.str();
}

}  // namespace

std::string BmffBoxesToStr(const uint8_t data[], size_t data_size) {
  return BmffBoxesToStr(data, data_size, /*indent=*/1);
}

//------------------------------------------------------------------------------

}  // namespace container
