#pragma once

#include "ggml.h"

#include <cstdint>

static inline bool llama_kv_type_is_turbo(ggml_type type) {
    return type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0;
}

static inline uint32_t llama_turbo_pad_head_dim(uint32_t n) {
    return ((n + 127u) / 128u) * 128u;
}
