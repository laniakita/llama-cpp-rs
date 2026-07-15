#pragma once

#include <stdbool.h>
#include <stddef.h>

typedef enum llama_rs_status {
  LLAMA_RS_STATUS_OK = 0,
  LLAMA_RS_STATUS_INVALID_ARGUMENT = -1,
  LLAMA_RS_STATUS_ALLOCATION_FAILED = -2,
  LLAMA_RS_STATUS_EXCEPTION = -3
} llama_rs_status;

#ifdef __cplusplus

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static inline char *llama_rs_dup_string(const std::string &value) {
  char *buffer = static_cast<char *>(std::malloc(value.size() + 1));
  if (!buffer) {
    return nullptr;
  }
  std::memcpy(buffer, value.data(), value.size());
  buffer[value.size()] = '\0';
  return buffer;
}

static char **llama_rs_dup_string_vector(const std::vector<std::string> &vec) {
  if (vec.empty()) {
    return nullptr;
  }

  // Allocate array of pointers, +1 for the null terminator
  char **arr =
      static_cast<char **>(std::malloc((vec.size() + 1) * sizeof(char *)));
  if (!arr)
    return nullptr;

  for (size_t i = 0; i < vec.size(); ++i) {
    arr[i] = llama_rs_dup_string(vec[i].c_str());
  }

  // Null terminate the array so Rust knows where it ends
  arr[vec.size()] = nullptr;

  return arr;
}

#endif
