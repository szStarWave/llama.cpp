#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <assert.h>
#include <math.h>
#include <string.h>

static const float turbo_centroids_3[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f,
};

static const float turbo_centroids_4[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f,
};

static const float turbo_wht_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,
};

static const float turbo_wht_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,
};

void ggml_turbo_wht_128(const float * GGML_RESTRICT src, float * GGML_RESTRICT dst, int direction) {
    const float * first  = direction == 0 ? turbo_wht_s1 : turbo_wht_s2;
    const float * second = direction == 0 ? turbo_wht_s2 : turbo_wht_s1;

    for (int i = 0; i < 128; ++i) {
        dst[i] = src[i] * first[i];
    }
    for (int step = 1; step < 128; step <<= 1) {
        for (int i = 0; i < 128; i += 2 * step) {
            for (int j = i; j < i + step; ++j) {
                const float a = dst[j];
                const float b = dst[j + step];
                dst[j] = a + b;
                dst[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; ++i) {
        dst[i] *= 0.08838834764831845f * second[i];
    }
}

static int turbo_nearest_3(float value) {
    if (value < -0.154496f) return 0;
    if (value < -0.092804f) return 1;
    if (value < -0.044243f) return 2;
    if (value <  0.000000f) return 3;
    if (value <  0.044243f) return 4;
    if (value <  0.092804f) return 5;
    if (value <  0.154496f) return 6;
    return 7;
}

static int turbo_nearest_4(float value) {
    if (value < -0.212203f) return 0;
    if (value < -0.162947f) return 1;
    if (value < -0.127026f) return 2;
    if (value < -0.097164f) return 3;
    if (value < -0.070671f) return 4;
    if (value < -0.046174f) return 5;
    if (value < -0.022824f) return 6;
    if (value <  0.000000f) return 7;
    if (value <  0.022824f) return 8;
    if (value <  0.046174f) return 9;
    if (value <  0.070671f) return 10;
    if (value <  0.097164f) return 11;
    if (value <  0.127026f) return 12;
    if (value <  0.162947f) return 13;
    if (value <  0.212203f) return 14;
    return 15;
}

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);

    for (int64_t block = 0; block < k / QK_TURBO3; ++block) {
        float normalized[128];
        float rotated[128];
        float norm_sq = 0.0f;
        for (int i = 0; i < 128; ++i) {
            const float value = x[block * 128 + i];
            normalized[i] = value;
            norm_sq += value * value;
        }

        const float norm = sqrtf(norm_sq);
        const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
        for (int i = 0; i < 128; ++i) {
            normalized[i] *= inv_norm;
        }
        ggml_turbo_wht_128(normalized, rotated, 0);

        memset(y[block].qs, 0, sizeof(y[block].qs));
        memset(y[block].signs, 0, sizeof(y[block].signs));
        float reconstructed_sq = 0.0f;
        for (int i = 0; i < 128; ++i) {
            const int index = turbo_nearest_3(rotated[i]);
            y[block].qs[i / 4] |= (uint8_t) ((index & 3) << (2 * (i % 4)));
            y[block].signs[i / 8] |= (uint8_t) (((index >> 2) & 1) << (i % 8));
            reconstructed_sq += turbo_centroids_3[index] * turbo_centroids_3[index];
        }
        const float reconstructed_norm = sqrtf(reconstructed_sq);
        y[block].norm = GGML_FP32_TO_FP16(reconstructed_norm > 1e-10f ? norm / reconstructed_norm : norm);
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    for (int64_t block = 0; block < k / QK_TURBO3; ++block) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int i = 0; i < 128; ++i) {
            const int index = ((x[block].qs[i / 4] >> (2 * (i % 4))) & 3) |
                              (((x[block].signs[i / 8] >> (i % 8)) & 1) << 2);
            y[block * 128 + i] = turbo_centroids_3[index] * norm;
        }
    }
}

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);

    for (int64_t block = 0; block < k / QK_TURBO4; ++block) {
        float normalized[128];
        float rotated[128];
        float norm_sq = 0.0f;
        for (int i = 0; i < 128; ++i) {
            const float value = x[block * 128 + i];
            normalized[i] = value;
            norm_sq += value * value;
        }

        const float norm = sqrtf(norm_sq);
        const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
        for (int i = 0; i < 128; ++i) {
            normalized[i] *= inv_norm;
        }
        ggml_turbo_wht_128(normalized, rotated, 0);

        memset(y[block].qs, 0, sizeof(y[block].qs));
        float reconstructed_sq = 0.0f;
        for (int i = 0; i < 128; ++i) {
            const int index = turbo_nearest_4(rotated[i]);
            y[block].qs[i / 2] |= (uint8_t) (index << (4 * (i % 2)));
            reconstructed_sq += turbo_centroids_4[index] * turbo_centroids_4[index];
        }
        const float reconstructed_norm = sqrtf(reconstructed_sq);
        y[block].norm = GGML_FP32_TO_FP16(reconstructed_norm > 1e-10f ? norm / reconstructed_norm : norm);
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    for (int64_t block = 0; block < k / QK_TURBO4; ++block) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int i = 0; i < 128; ++i) {
            const int index = (x[block].qs[i / 2] >> (4 * (i % 2))) & 15;
            y[block * 128 + i] = turbo_centroids_4[index] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    const size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo3_0_ref(src + row * n_per_row, (block_turbo3_0 *) ((char *) dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    const size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo4_0_ref(src + row * n_per_row, (block_turbo4_0 *) ((char *) dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}
