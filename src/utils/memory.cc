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
// Memory allocator and containers.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <cstdlib>

#include "src/utils/utils.h"

//------------------------------------------------------------------------------
// memory hook

void WP2MemoryHook::Free(void *ptr) { free(ptr); }
void* WP2MemoryHook::Malloc(size_t size) { return malloc(size); }
void* WP2MemoryHook::Calloc(size_t count, size_t size) {
  return calloc(count, size);
}

static WP2MemoryHook kMemory;   // singleton
static WP2MemoryHook* kMemoryPtr = &kMemory;

WP2MemoryHook* WP2SetMemoryHook(WP2MemoryHook* hook) {
  if (hook == nullptr) return kMemoryPtr;
  WP2MemoryHook* const old = kMemoryPtr;
  kMemoryPtr = hook;
  return old;
}

constexpr WP2Allocable::NoThrow WP2Allocable::nothrow;

//------------------------------------------------------------------------------
// default WP2MemoryHook implementation

bool WP2MemoryHook::CheckSizeArgumentsOverflow(uint64_t nmemb, size_t size) {
  const uint64_t total_size = nmemb * size;
  if (nmemb == 0) return true;
  if ((uint64_t)size > WP2_MAX_ALLOCABLE_MEMORY / nmemb) return false;
  if (total_size != (size_t)total_size) return false;
  return true;
}

void* WP2Malloc(uint64_t nmemb, size_t size) {
  if (!kMemoryPtr->CheckSizeArgumentsOverflow(nmemb, size)) return nullptr;
  assert(nmemb * size > 0);
  void* const ptr = kMemoryPtr->Malloc((size_t)(nmemb * size));
  if (ptr == nullptr) return nullptr;
  if (!kMemoryPtr->Register(ptr, (size_t)(nmemb * size))) {
    kMemoryPtr->Free(ptr);
    return nullptr;
  }
  return ptr;
}

void* WP2Calloc(uint64_t nmemb, size_t size) {
  if (!kMemoryPtr->CheckSizeArgumentsOverflow(nmemb, size)) return nullptr;
  assert(nmemb * size > 0);
  void* const ptr = kMemoryPtr->Calloc((size_t)nmemb, size);
  if (ptr == nullptr) return nullptr;
  if (!kMemoryPtr->Register(ptr, (size_t)(nmemb * size))) {   // fake fail?
    kMemoryPtr->Free(ptr);
    return nullptr;
  }
  return ptr;
}

void WP2Free(void* ptr) {
  if (ptr != nullptr) {
    kMemoryPtr->Unregister(ptr);
    kMemoryPtr->Free(ptr);
  }
}

//------------------------------------------------------------------------------
