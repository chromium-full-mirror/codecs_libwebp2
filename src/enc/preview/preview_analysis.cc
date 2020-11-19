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
//  Preview vertices collection
//
// Author: Skal (pascal.massimino@gmail.com)

#include "./preview_enc.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#include "src/common/preview/preview.h"
#include "src/utils/random.h"

// Uncomment the following to enable visual debug functions.
// #define VISUAL_DEBUG

namespace WP2 {

//------------------------------------------------------------------------------

namespace {

// Converts pixels to opaque gray levels. Transparency gives black.
WP2Status ConvertToGrayLevels(const ArgbBuffer& src, Plane8u* const dst) {
  assert(src.width() < (1u << 16) && src.height() < (1u << 16));
  WP2_CHECK_STATUS(dst->Process(src));
  return WP2_STATUS_OK;
}

// Resizes and fills 'dst' with edge intensity. Uses a Sobel kernel.
WP2Status DetectEdges(const ArgbBuffer& canvas, Plane16* const dst) {
  Plane8u gray;
  WP2_CHECK_STATUS(ConvertToGrayLevels(canvas, &gray));

  dst->Fill(0);
  for (uint32_t y = 1; y < canvas.height() - 1; ++y) {
    const uint8_t* const src = gray.Row(y);
    const uint32_t dst_y = y * dst->h_ / canvas.height();
    auto* const sobel = dst->Row(dst_y);
    for (uint32_t x = 1; x < canvas.width() - 1; ++x) {
      const int32_t neighbors_sum = (src +          1)[x] +
                                    (src -          1)[x] +
                                    (src + gray.step_)[x] +
                                    (src - gray.step_)[x];
      const int32_t diff = std::abs(neighbors_sum - 4 * (int32_t)src[x]);
      // rescale from canvas to grid coordinate
      const uint32_t dst_x = x * dst->w_ / canvas.width();
      sobel[dst_x] = std::min(sobel[dst_x] + diff, 32767);
    }
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

constexpr float GetPotential(float square_distance) {
  return 1.f / square_distance;  // Alternative: pow(square_distance, -2.);
}

// Allocates and fills the 'potential_table' with values inversely proportional
// to the distance to the origin.
void CreatePotentialTable(Planef* const potential_table) {
  static constexpr float kMinimumDistance = 4.f;
  static constexpr float kMinimumDistancePotential =
      GetPotential(kMinimumDistance * kMinimumDistance);
  static constexpr float kMaximumDistance = 20.f;
  static constexpr float kMaximumDistancePotential = 0.f;

  for (uint32_t y = 0; y < potential_table->h_; ++y) {
    float* const dst = potential_table->Row(y);
    for (uint32_t x = 0; x < potential_table->w_; ++x) {
      const float square_distance = (float)y * y + x * x;
      float potential;
      if (x == 0 && y == 0) {
        potential = 1e20f;  // High potential on exact spot.
      } else if (square_distance < kMinimumDistance * kMinimumDistance) {
        potential = kMinimumDistancePotential;  // Clamp max.
      } else if (square_distance > kMaximumDistance * kMaximumDistance) {
        potential = kMaximumDistancePotential;  // Clamp min.
      } else {
        potential = GetPotential(square_distance);
      }
      dst[x] = potential;
    }
  }
}

// Increases each 'repulsive' cell depending on the value of the
// 'potential_table' corresponding to the distance to 'point'.
void RepulseFutureVerticesFrom(const VertexIndexedColor& point,
                               const Planef& potential_table,
                               Planef* const repulsive) {
  assert(point.x < potential_table.w_ && point.y < potential_table.h_);
  for (uint32_t y = 0; y < repulsive->h_; ++y) {
    const uint32_t distance_y = (y > point.y) ? y - point.y : point.y - y;
    float* const dst_row = repulsive->Row(y);
    const float* const src_row = potential_table.Row(distance_y);
    for (uint32_t x = 0; x < repulsive->w_; ++x) {
      const uint32_t distance_x = (x > point.x) ? x - point.x : point.x - x;
      dst_row[x] += src_row[distance_x];
    }
  }
}

//------------------------------------------------------------------------------

float GetDiff(const ArgbBuffer& reference, const ArgbBuffer& canvas,
              uint32_t pixel_x, uint32_t pixel_y,
              uint32_t grid_w, uint32_t grid_h,
              int radius = 0) {
  assert(canvas.width() == reference.width());
  assert(canvas.height() == reference.height());
  const uint32_t cx = DivRound(pixel_x * (canvas.width() - 1), grid_w - 1);
  const uint32_t cy = DivRound(pixel_y * (canvas.height() - 1), grid_h - 1);
  float diff = 0.f;
  const int y_limit = canvas.height() - 1;
  const int x_limit = canvas.width() - 1;
  for (int dy = -radius; dy <= radius; ++dy) {
    const uint32_t y = (uint32_t)Clamp((int)cy + dy, 0, y_limit);
    for (int dx = -radius; dx <= radius; ++dx) {
      const uint32_t x = (uint32_t)Clamp((int)cx + dx, 0, x_limit);
      const uint8_t* const argb_reference = reference.GetRow8(y) + 4 * x;
      const uint8_t* const argb_canvas = canvas.GetRow8(y) + 4 * x;
      for (uint32_t c = 0; c < 4; ++c) {
        diff += std::abs((float)argb_reference[c] - argb_canvas[c]);
      }
    }
  }
  const uint32_t dim = 2 * radius + 1;
  return diff / (dim * dim);
}

}  // namespace

//------------------------------------------------------------------------------

WP2Status CollectVerticesFromEdgeSelectionAndRepulsion(
    const ArgbBuffer& canvas, uint32_t max_num_vertices,
    PreviewData* const preview) {
  assert(preview != nullptr);
  Vector<VertexIndexedColor>& vertices = preview->vertices_;
  vertices.clear();

  static constexpr float kEdgeScale = 0.05f;

  Plane16 edges;
  WP2_CHECK_STATUS(edges.Resize(preview->grid_width_, preview->grid_height_));
  WP2_CHECK_STATUS(DetectEdges(canvas, &edges));

  Planef repulsive;
  WP2_CHECK_STATUS(repulsive.Resize(edges.w_, edges.h_));
  repulsive.Fill(0.f);

  Planef potential_table;
  WP2_CHECK_STATUS(potential_table.Resize(edges.w_, edges.h_));
  CreatePotentialTable(&potential_table);

  WP2_CHECK_ALLOC_OK(vertices.resize(max_num_vertices));
  uint32_t num_vertices = 0;
  for (auto& vtx : vertices) {
    float max_attraction = std::numeric_limits<float>::lowest();
    bool found = false;
    for (uint16_t y = 0; y < edges.h_; ++y) {
      const auto* const edges_row = edges.Row(y);
      const float* const repulsive_row = repulsive.Row(y);
      for (uint16_t x = 0; x < edges.w_; ++x) {
        if (preview->IsCorner(x, y)) continue;
        const float attraction = kEdgeScale * edges_row[x] - repulsive_row[x];
        if (attraction > max_attraction) {
          max_attraction = attraction;
          vtx = { x, y, 0 };
          found = true;
        }
      }
    }
    if (!found) break;
    ++num_vertices;
    RepulseFutureVerticesFrom(vtx, potential_table, &repulsive);
  }
  vertices.resize_no_check(num_vertices);
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status CollectVerticesFromColorDiffMaximization(
    const ArgbBuffer& canvas, uint32_t max_num_vertices,
    PreviewData* const preview) {
  assert(preview != nullptr);
  static constexpr uint32_t avg_radius = 3;

  auto& vertices = preview->vertices_;
  auto& palette = preview->palette_;

  WP2_CHECK_ALLOC_OK(vertices.reserve(max_num_vertices));
  WP2_CHECK_ALLOC_OK(palette.reserve(max_num_vertices));

  ArgbBuffer result;
  WP2_CHECK_STATUS(result.Resize(canvas.width(), canvas.height()));

  const uint32_t w = preview->grid_width_, h = preview->grid_height_;
  Vector_u8 used;  // Vector does not handle bool very well
  WP2_CHECK_ALLOC_OK(used.resize(w * h));
  std::fill(used.begin(), used.end(), 0u);

  palette.reset();
  WP2_CHECK_STATUS(MakePaletteValid(&palette));

  // Init corners with black color.
  for (auto& c : preview->corners_) c = 0;

  Triangulation triangulation;
  WP2_CHECK_STATUS(triangulation.Create(*preview));

  // Greedily add vertices one at a time.
  for (uint32_t i = 0; i < max_num_vertices; ++i) {
    uint32_t num_covered_pixels = 0;
    WP2_CHECK_STATUS(
        triangulation.FillTriangles(*preview, &result, &num_covered_pixels));
    assert(num_covered_pixels >= result.width() * result.height());

    float best_diff = -1.f;
    uint32_t best_offset = 0;
    VertexIndexedColor best_vtx = { 0, 0, (uint16_t)palette.size() };
    for (uint16_t y = 0; y < h; ++y) {
      for (uint16_t x = 0; x < w; ++x) {
        if (preview->IsCorner(x, y)) continue;
        const uint32_t offset = x + y * w;
        if (used[offset]) continue;
        const float diff = GetDiff(canvas, result, x, y, w, h);
        assert(diff >= 0.f);
        if (diff > best_diff) {
          best_diff = diff;
          best_vtx.x = x;
          best_vtx.y = y;
          best_offset = offset;
        }
      }
    }
    if (best_diff < 0) break;
    used[best_offset] = 1u;
    const uint32_t cx = best_vtx.x * (canvas.width() - 1) / (w - 1);
    const uint32_t cy = best_vtx.y * (canvas.height() - 1) / (h - 1);
    const AYCoCg19b col =
        ToAYCoCg19b(GetAverageColor(canvas, avg_radius, cx, cy));
    WP2_CHECK_ALLOC_OK(palette.push_back(col));
    WP2_CHECK_ALLOC_OK(vertices.push_back(best_vtx));
    WP2_CHECK_STATUS(triangulation.Insert(vertices, i));
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------

WP2Status CollectVerticesFromRandomEdgeSelection(
    const ArgbBuffer& canvas, uint32_t max_num_vertices,
    PreviewData* const preview) {
  assert(preview != nullptr);
  Vector<VertexIndexedColor>& vertices = preview->vertices_;
  vertices.clear();

  Plane16 edges;
  WP2_CHECK_STATUS(edges.Resize(preview->grid_width_, preview->grid_height_));
  WP2_CHECK_STATUS(DetectEdges(canvas, &edges));

  // Remove isolated vertices.
  for (uint16_t y = 1; y < edges.h_ - 1; ++y) {
    auto* const row = edges.Row(y);
    for (uint16_t x = 1; x < edges.w_ - 1; ++x) {
      const int nb = (int)((row[x] > row[x + edges.step_]) +
                           (row[x] > row[x + 1]) +
                           (row[x] > row[x - 1]) +
                           (row[x] > row[x - edges.step_]));
      if (nb >= 3) row[x] = 0;
    }
  }

  struct Candidate {
    uint16_t x, y, score;
    bool operator<(const Candidate& other) const {
      return (score > other.score);
    }
  };

  const size_t num_best_candidates = 2 * max_num_vertices;
  Vector<Candidate> candidates_heap;
  WP2_CHECK_ALLOC_OK(candidates_heap.reserve(num_best_candidates));

  // Discard but the best 'num_kept_vertices' candidates.
  for (uint16_t y = 0; y < edges.h_; ++y) {
    auto* const row = edges.Row(y);
    for (uint16_t x = 0; x < edges.w_; ++x) {
      if (preview->IsCorner(x, y)) continue;
      if (row[x] > 0) {
        if (candidates_heap.size() == candidates_heap.capacity()) {
          std::pop_heap(candidates_heap.begin(), candidates_heap.end());
          candidates_heap.pop_back();
        }
        WP2_CHECK_ALLOC_OK(candidates_heap.push_back(
            {x, y, (uint16_t)row[x]}, /*resize_if_needed=*/false));
        std::push_heap(candidates_heap.begin(), candidates_heap.end());
      }
    }
  }
  max_num_vertices = std::min(candidates_heap.size(), (size_t)max_num_vertices);

  // Randomly pick at most 'max_num_vertices' from the 'num_best_candidates'.
  const size_t random_seed = edges.w_ + edges.h_;
  Shuffle(candidates_heap.begin(), candidates_heap.end(), random_seed);

  WP2_CHECK_ALLOC_OK(vertices.resize(max_num_vertices));
  for (uint32_t i = 0; i < max_num_vertices; ++i) {
    const Candidate candidate = candidates_heap[i];
    vertices[i] = {candidate.x, candidate.y, 0};
  }
  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
// Voronoi-based vertices placement

namespace {

struct VoronoiCell {
  uint32_t acc   = 0;   // sum{1}
  int32_t r_acc  = 0;   // sum{rho}
  double x_acc   = 0;   // sum{x} / acc
  double y_acc   = 0;   // sum{y} / acc
  double rx_acc  = 0;   // sum{rho * x} / r_acc
  double ry_acc  = 0;   // sum{rho * y} / r_acc
  double rxx_acc = 0;   // higher moments, for determining cell's typical size
  double rxy_acc = 0;
  double ryy_acc = 0;

  void Reset() {
    acc = 0u;
    x_acc = y_acc = 0;
    r_acc = 0;
    rx_acc = ry_acc = 0;
    rxx_acc = ryy_acc = rxy_acc = 0;
  }
  void Add(double x, double y, int32_t rho) {
    acc += 1u;
    x_acc += x;
    y_acc += y;
    r_acc  += rho;
    rx_acc += rho * x;
    ry_acc += rho * y;
    rxx_acc += rho * x * x;
    rxy_acc += rho * x * y;
    ryy_acc += rho * y * y;
  }

  bool StoreAverage() {   // returns false is cell is empty
    if (acc == 0) return false;
    const double norm = 1.f / acc;
    x_acc *=  norm;
    y_acc *=  norm;
    if (r_acc > 0) {
      const double rnorm = 1.f / r_acc;
      rx_acc  *= rnorm;
      ry_acc  *= rnorm;
      rxx_acc *= rnorm;
      rxy_acc *= rnorm;
      ryy_acc *= rnorm;
    }
    return true;
  }

  void GetMainDirection(float* const dx, float* const dy,
                        float* const amp = nullptr) const {
    // cf. http://scipp.ucsc.edu/~haber/ph116A/diag2x2_11.pdf
    const double exx = rxx_acc - x_acc * x_acc;
    const double eyy = ryy_acc - y_acc * y_acc;
    const double num = rxy_acc - x_acc * y_acc;
    const double den = exx - eyy;
    double t = 0.;
    if (num * num + den * den > 0.) {
      t = 0.5 * std::atan2(2. * num, den);
    }
    // displacement of a half-radius in the main direction
    double radius = 0.5 * std::sqrt(acc / M_PI);   // a = pi.r^2
    if (amp != nullptr) {
      radius *= *amp;
      *amp = radius;
    }
    *dx = radius * std::cos(t);
    *dy = radius * std::sin(t);
  }
  bool operator<(const VoronoiCell& c) const { return (r_acc > c.r_acc); }
};

uint32_t Distance2(const VertexIndexedColor& v, int32_t x, int32_t y) {
  return (v.x - x) * (v.x - x) + (v.y - y) * (v.y - y);
}

// Collects Voronoi cell stats.
// 'view' is for debugging only. The Voronoi diagram is drawn there if non null.
static void CollectVoronoi(const Plane16& src,
                           Vector<VertexIndexedColor>& vertices,
                           Vector<VoronoiCell>& cells, Plane<int>* const view) {
  // sort y-first, x-second
  std::sort(vertices.begin(), vertices.end(),
            [](const VertexIndexedColor& a, const VertexIndexedColor& b) {
              return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); });
  // remove dupes
  uint32_t num_vertices = 0;
  for (auto& v : vertices) {
    if (num_vertices > 0 &&
        v.x == vertices[num_vertices - 1].x &&
        v.y == vertices[num_vertices - 1].y) continue;
    std::swap(vertices[num_vertices], v);
    ++num_vertices;
  }
  vertices.resize_no_check(num_vertices);

  cells.resize_no_check(num_vertices);
  for (auto& c : cells) c.Reset();

  const uint32_t w = src.w_ - 1, h = src.h_ - 1;
  int best_y = 0;
  for (uint32_t y = 0; y <= h; ++y) {
    int best = best_y;
    for (uint32_t x = 0; x <= w; ++x) {
      uint32_t best_dist = Distance2(vertices[best], x, y);
      const int32_t min_y = y - (int32_t)std::sqrt(1. * best_dist);
      for (int k = best - 1; k >= 0; --k) {
        if (vertices[k].y <= min_y) break;
        const uint32_t d = Distance2(vertices[k], x, y);
        if (d < best_dist) {
          best = k;
          best_dist = d;
        }
      }
      const int32_t max_y = y + (int32_t)std::sqrt(1. * best_dist);
      for (uint32_t k = best + 1u; k < num_vertices; ++k) {
        if (vertices[k].y >= max_y) break;
        const uint32_t d = Distance2(vertices[k], x, y);
        if (d < best_dist) {
          best = k;
          best_dist = d;
        }
      }
      cells[best].Add(x , y, src.Row(y)[x]);
      if (view != nullptr) view->Row(y)[x] = best;   // draw the cells
      if (x == 0) best_y = best;  // for next line
    }
  }
}

// Move vertices to rho-weighted cell center.
// Returns false if the overall displacement is small.
bool MoveToCentroid(Vector<VertexIndexedColor>& vertices,
                    Vector<VoronoiCell>& cells) {
  uint64_t displacement = 0;
  uint64_t count = 0;
  for (uint32_t i = 0; i < vertices.size(); ++i) {
    if (!cells[i].StoreAverage()) continue;
    const int32_t new_x = (int32_t)std::lround(cells[i].rx_acc);
    const int32_t new_y = (int32_t)std::lround(cells[i].ry_acc);
    displacement += Distance2(vertices[i], new_x, new_y);
    vertices[i].x = new_x;
    vertices[i].y = new_y;
    ++count;
  }
  return (displacement > count);  // more than 'count' point moved?
}

//------------------------------------------------------------------------------

#if defined(VISUAL_DEBUG)

void DrawDirections(const Vector<VoronoiCell>& cells, Plane8u& p) {
  for (auto& c : cells) {
    if (c.acc <= 1) continue;
    const float x = c.rx_acc, y = c.ry_acc;
    float dx, dy, amp = 1.;
    c.GetMainDirection(&dx, &dy, &amp);
    for (int32_t t = 0; t <= 32; ++t) {
      const int u = (int)std::lround(x + t * dx / 32.);
      const int v = (int)std::lround(y + t * dy / 32.);
      p.AtClamped(u, v) = 128;
    }
  }
}

void DrawDisc(Plane8u& out, float x, float y, float r) {
  for (int Y = (int)(y - r); Y <= (int)(y + r); ++Y) {
    for (int X = (int)(x - r); X <= (int)(x + r); ++X) {
      const float d2 = ((X - x) * (X - x) + (Y - y) * (Y - y)) / (r * r);
      if (d2 <= 1.) {
        const float v = out.AtClamped(X, Y) - pow(1. - d2, 0.7) * 255.;
        out.AtClamped(X, Y) = (uint8_t)(std::max(v, 0.f));
      }
    }
  }
}

WP2Status DrawAndSave(const Vector<VoronoiCell>& cells,
                      const Vector<VertexIndexedColor>& vtx,
                      const Plane16& gray, const Plane<int>& voronoi,
                      uint32_t iter, float radius) {
  const uint32_t w = voronoi.w_, h = voronoi.h_;
  Plane8u out;
  WP2_CHECK_STATUS(out.Resize(w, h));

  out.Fill(255);

  if (false) {   // draw Voronoi diagram
    for (auto it = out.begin(); it != out.end(); ++it) {
      const auto c = cells[voronoi.Row(it.y_)[it.x_]];
      *it = 255 - c.r_acc / c.acc;
    }
  }
  if (false) DrawDirections(cells, out);

  if (true) {  // draw particles
    for (auto& v : vtx) {
      DrawDisc(out, (float)v.x, (float)v.y, radius);
    }
  }
  // save as raw PPM
  char tmp[32];
  sprintf(tmp, "toto%.3d.ppm", iter++);  // NOLINT
  FILE* const f = fopen(tmp, "wb");
  fprintf(f, "P5\n%d %d\n255\n", w, h);
  for (auto c : out) fprintf(f, "%c", c);
  fclose(f);

  return WP2_STATUS_OK;
}

#endif  // VISUAL_DEBUG

//------------------------------------------------------------------------------

static int32_t ToGray(const void* src) {
  const uint8_t* const src8 = (const uint8_t*)src;
  const int32_t r = src8[1], g = src8[2], b = src8[3];
  double gray = 0.299 * r + 0.587 * g + 0.114 * b;  // rec.601
  gray = 255. * pow(gray / 255., 0.45);   // gamma-correction
  return (int32_t)(255. - gray);  // white = 0, black = 255
}

void QuantizeToGrid(Vector<VertexIndexedColor>& vertices,
                    uint32_t grid_w, uint32_t W, uint32_t grid_h, uint32_t H) {
  for (auto& v : vertices) {
    const uint32_t ix = (v.x * grid_w + W / 2) / W;
    const uint32_t iy = (v.y * grid_h + H / 2) / H;
    v.x = (ix * W + grid_w / 2) / grid_w;
    v.y = (iy * H + grid_h / 2) / grid_h;
  }
}

}  // namespace

WP2Status CollectVerticesFromVoronoiCells(const ArgbBuffer& canvas,
    uint32_t max_num_vertices, PreviewData* const preview) {
  assert(preview != nullptr);

  Vector<VertexIndexedColor>& vertices = preview->vertices_;
  vertices.clear();
  WP2_CHECK_ALLOC_OK(vertices.reserve(max_num_vertices));

  Vector<VertexIndexedColor> extra_vertices;
  WP2_CHECK_ALLOC_OK(extra_vertices.reserve(max_num_vertices));

  constexpr uint32_t upsampling = 1;  // or 2
  Plane16 gray;
  WP2_CHECK_STATUS(gray.Process(canvas, ToGray));
  if (upsampling == 2) {
    WP2_CHECK_STATUS(gray.Upsample(upsampling * canvas.width(),
                                   upsampling * canvas.height()));
  }
  WP2_CHECK_STATUS(DetectEdges(canvas, &gray));

  // determine how much density each dot represent
  double avg_rho = 0.;
  for (auto g : gray) avg_rho += g;
  avg_rho /= max_num_vertices;

  Plane<int> voronoi;
  WP2_CHECK_STATUS(voronoi.Resize(gray.w_, gray.h_));
  const uint32_t w = gray.w_ - 1, h = gray.h_ - 1;
  const uint32_t grid_w = preview->grid_width_ - 1;
  const uint32_t grid_h = preview->grid_height_ - 1;

  // initial random vertices
  uint32_t num_vertices = std::min(20u, max_num_vertices / 5u);
  WP2_CHECK_ALLOC_OK(vertices.resize(num_vertices));
  UniformIntDistribution rnd(/*seed=*/5543);
  for (auto& v : vertices) {
    v = { rnd.Get<uint16_t>(0, w), rnd.Get<uint16_t>(0, h), 0 };
  }

  Vector<VoronoiCell> cells;
  WP2_CHECK_ALLOC_OK(cells.reserve(max_num_vertices));

  const uint32_t num_iters = 40u;
  const uint32_t num_Lloyd = 4u;
  const float point_area = avg_rho;
  for (uint32_t iter = 0; iter < num_iters; ++iter) {
    // several passes of Lloyd's algorithm
    for (uint32_t n = 0; n < num_Lloyd; ++n) {
      QuantizeToGrid(vertices, grid_w, w, grid_h, h);
      CollectVoronoi(gray, vertices, cells, &voronoi);
      if (!MoveToCentroid(vertices, cells)) break;  // stabilized?
    }

    // hysteresis
    const float hysteresis = 0.20f + 0.50f * iter / num_iters;
    const float Tu = (1. + hysteresis) * point_area;
    const float Tl = (1. - hysteresis) * point_area;

    // split / keep / merge vertices
    const uint32_t last_num_vertices = vertices.size();
    vertices.clear();
    extra_vertices.clear();
    for (const auto& c : cells) {
      if (c.acc == 0) continue;   // remove empty cell
      const float rho = c.r_acc;
      if (rho < Tl) continue;       // remove the vertex (too sparse)
      const float xf = c.rx_acc;
      const float yf = c.ry_acc;
      const uint16_t xi = (uint16_t)std::lround(xf);
      const uint16_t yi = (uint16_t)std::lround(yf);
      bool stored = false;
      if (c.acc > 1 && rho > Tu) {   // if too dense, split the cell's vertex
        float dx, dy, amp = 0.5f;
        c.GetMainDirection(&dx, &dy, &amp);
        const uint16_t v1_x = (uint16_t)std::lround(xf - dx);
        const uint16_t v1_y = (uint16_t)std::lround(yf - dy);
        if (v1_x != xi || v1_y != yi) {
          if (v1_x <= w && v1_y <= h) {
            vertices.push_back_no_resize({v1_x, v1_y, 0});
            stored = true;
          }
          const uint16_t v2_x = (uint16_t)std::lround(xf + dx);
          const uint16_t v2_y = (uint16_t)std::lround(yf + dy);
          if (v2_x <= w && v2_y <= h) {
            extra_vertices.push_back_no_resize({v2_x, v2_y, 0});
          }
        }
      }
      if (!stored) vertices.push_back_no_resize({xi, yi, 0});
    }
    if (vertices.size() == last_num_vertices && extra_vertices.size() <= 5) {
      break;  // done! No further moves (or too few).
    }

    if (vertices.size() + extra_vertices.size() > max_num_vertices) {
      extra_vertices.resize_no_check(max_num_vertices - vertices.size());
    }

    for (auto& v : extra_vertices) vertices.push_back_no_resize(v);
    extra_vertices.clear();

#if defined VISUAL_DEBUG
    printf("#%d: %d\n", iter, (int)vertices.size());
    const float radius = std::sqrt(point_area / M_PI / 255.);
    DrawAndSave(cells, vertices, gray, voronoi, iter, radius);
#endif
  }

  // convert back to grid units, and clean-up
  for (auto& v : vertices) {
    v.x = v.x * grid_w / w;
    v.y = v.y * grid_h / h;
  }
  WP2_CHECK_STATUS(preview->CleanupVertices());

  return WP2_STATUS_OK;
}

//------------------------------------------------------------------------------
}  // namespace WP2
