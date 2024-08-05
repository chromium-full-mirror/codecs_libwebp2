// Copyright 2020 Google LLC
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
//  Custom memory allocators

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

// A regular ThreadLock cannot be used as its ctor would call WP2Malloc(),
// which uses this 'thread_lock', which calls WP2Malloc() in its ctor etc.
// std::mutex might be forbidden in the library, but here it's for debugging.
#include <mutex>  // NOLINT
#include <utility>
#include <vector>

#if defined(WP2_HAVE_BOOST)
#include <boost/stacktrace/stacktrace.hpp>
#include <sstream>
#elif defined(WP2_HAVE_BACKTRACE)
#include <execinfo.h>
#endif

#include "examples/example_utils.h"
#include "src/utils/utils.h"

static std::unordered_map<void*, size_t> all_blocks;  // NOLINT
static size_t mem_allocated = 0;
static size_t total_allocation = 0;
static size_t high_water_mark = 0;
static int num_malloc_calls = 0;
static int num_free_calls = 0;

static std::mutex mutex;

namespace WP2 {

namespace {
// Get the stack trace.
std::string GetTrace() {
#if defined(WP2_HAVE_BOOST)
  std::stringstream ss;
  ss << boost::stacktrace::stacktrace();
  return ss.str();
#elif defined(WP2_HAVE_BACKTRACE)
  // glibc solution with mangled names.
  static constexpr uint32_t kMaxSize = 1024;
  void* array[kMaxSize];
  const size_t size = backtrace(array, kMaxSize);
  char** strings = backtrace_symbols(array, size);
  assert(strings != NULL);
  std::string trace;
  for (size_t i = 0; i < size; i++) {
    trace += strings[i];
    trace += "\n";
  }
  free(strings);
  return trace;
#else
  return "";
#endif
}
}  // namespace

bool WP2MemoryFailNow(size_t size) {
  fprintf(stderr, "=== FAILING to allocate %u more bytes NOW ===\n",
          (uint32_t)size);
  return false;
}

MemoryTrace::~MemoryTrace() {
  if (print_info || show_traffic || memory_limit > 0 || countdown_to_fail > 0 ||
      mem_allocated > 0 || top_stacktrace_num > 0) {
    std::lock_guard<std::mutex> lock(mutex);
    fprintf(stderr, "\n=== MEMORY INFO ===\n");
    fprintf(stderr, "num calls to: malloc = %4d\n", num_malloc_calls);
    fprintf(stderr, "              free   = %4d\n", num_free_calls);
    fprintf(stderr, "final memory:     %u%s\n", (uint32_t)mem_allocated,
            (mem_allocated == 0) ? "" : "  !!! ERROR");
    fprintf(stderr, "total allocation: %u\n", (uint32_t)total_allocation);
    fprintf(stderr, "high-water mark:  %u\n", (uint32_t)high_water_mark);
    if (show_traffic && mem_allocated > 0) {
      fprintf(stderr, "Still allocated:\n");
      for (auto it : all_blocks) {
        fprintf(stderr, "ptr = %p   size = %7u\n", it.first,
                (uint32_t)it.second);
      }
    }
    all_blocks.clear();
    if (top_stacktrace_num > 0) {
      // Compute the most frequent stack traces.
      typedef std::pair<std::string, uint32_t> Pair;
      std::vector<Pair> counts;
      counts.reserve(counts_per_trace.size());
      for (const auto& iter : counts_per_trace) counts.push_back(iter);
      std::sort(
          counts.begin(), counts.end(),
          [](const Pair& p1, const Pair& p2) { return p1.second > p2.second; });
      fprintf(stderr, "\n");
      for (uint32_t i = 0;
           i < std::min(top_stacktrace_num, (uint32_t)counts.size()); ++i) {
        fprintf(stderr, "Most called %d/%d: %d %s (might be a duplicate)\n\n",
                i + 1, top_stacktrace_num, counts[i].second,
                top_alloc_size ? "bytes" : "calls");
        fprintf(stderr, "%s\n\n", counts[i].first.c_str());
      }
    }
  }
}

bool MemoryTrace::Register(void* ptr, size_t size) {
  if (ptr == nullptr) return true;
  std::lock_guard<std::mutex> lock(mutex);
  // don't call Unregister() when failing here! This would dead-lock...
  if (countdown_to_fail >= 1) {
    if (countdown_to_fail == 1) {
      fprintf(stderr,
              "WARNING! We already FAILED! Do you have any delayed "
              "malloc checks (in constructors e.g.) ??\n");
      return false;
    }
    if (--countdown_to_fail == 1) {
      return WP2MemoryFailNow(size);
    }
  }
  if (memory_limit > 0 && mem_allocated + size > memory_limit) {
    if (show_traffic) {
      fprintf(stderr, "FAILING to allocate %u bytes over the %u limit\n",
              (uint32_t)size, (uint32_t)memory_limit);
    }
    return false;
  }

  ++num_malloc_calls;
  all_blocks[ptr] = size;
  mem_allocated += size;
  total_allocation += size;
  if (mem_allocated > high_water_mark) high_water_mark = mem_allocated;
  if (top_stacktrace_num > 0) {
    counts_per_trace[GetTrace()] += top_alloc_size ? size : 1;
  }
  if (show_traffic) {
    if (countdown_to_fail > 0) {
      fprintf(stderr, "fail-count: %5d [mem=%5u \t(+%7u)  ptr=%p]\n",
              countdown_to_fail, (uint32_t)mem_allocated, (uint32_t)size, ptr);
    } else {
      fprintf(stderr, "Alloc #%5u: %u (+%u)\n", (uint32_t)num_malloc_calls,
              (uint32_t)mem_allocated, (uint32_t)size);
    }
  }
  return true;
}

void MemoryTrace::Unregister(void* ptr) {
  if (ptr == nullptr) return;
  std::lock_guard<std::mutex> lock(mutex);
  ++num_free_calls;
  if (all_blocks.find(ptr) == all_blocks.end()) {
    fprintf(stderr, "Invalid pointer free! (%p)\n", ptr);
    abort();  // Critical error. Just can't happen.
  }
  const size_t size = all_blocks[ptr];
  all_blocks.erase(ptr);
  mem_allocated -= size;
  if (show_traffic) {
    fprintf(stderr, "Free          %u (-%u)\n", (uint32_t)mem_allocated,
            (uint32_t)size);
  }
}

MemoryTrace kMemoryTrace;  // NOLINT

//------------------------------------------------------------------------------

void ProgramOptions::AddMemoryOptionSection() {
#if !defined(NDEBUG)
  // these options are always available, but are pretty arcane. We don't want to
  // confuse the regular user with these. Hence the !NDEBUG in the help message.
  Add("Memory options:");
  Add("-mem_info", "report global memory usage information");
  Add("-mem_limit <int>", "sets the maximum allowed total memory usage");
  Add("-mem_fail <int>", "sets a maximum number of allocations");
  Add("-mem_traffic", "report all malloc/free traffic (verbose!)");
  // these two tracing modes are mutually exclusive:
  Add("-mem_top <int>", "report top malloc/free stack traces");
  Add("-mem_top_alloc <int>", "report top allocation stack traces");
#endif
}

bool ProgramOptions::ParseMemoryOptions(char* arg[], int argc, int& skip) {
  bool error = false;
  if (!strcmp(arg[0], "-mem_traffic")) {
    skip = 1;
    kMemoryTrace.show_traffic = true;
  } else if (!strcmp(arg[0], "-mem_info")) {
    kMemoryTrace.print_info = true;
    skip = 1;
  } else if (!strcmp(arg[0], "-mem_limit")) {
    if (argc < 2) return false;  // not enough args
    kMemoryTrace.memory_limit = ExUtilGetUInt(arg[1], &error);
    if (error) return false;
    skip = 2;
  } else if (!strcmp(arg[0], "-mem_fail")) {
    if (argc < 2) return false;  // not enough args
    kMemoryTrace.countdown_to_fail = 1 + ExUtilGetUInt(arg[1], &error);
    if (error) return false;
    skip = 2;
#if !defined(_MSC_VER)
  } else if (!strcmp(arg[0], "-mem_top") || !strcmp(arg[0], "-mem_top_alloc")) {
    if (argc < 2) return false;  // not enough args
    kMemoryTrace.top_alloc_size = !strcmp(arg[0], "-mem_top_alloc");
    kMemoryTrace.top_stacktrace_num = ExUtilGetUInt(arg[1], &error);
    if (error) return false;
    skip = 2;
#endif
  } else {
    return false;
  }
  WP2SetMemoryHook(&kMemoryTrace);
  return true;
}

}  // namespace WP2
