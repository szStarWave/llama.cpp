#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern void quantize_row_turbo3_0_ref(const float * x, void * y, int64_t k);
extern void dequantize_row_turbo3_0(const void * x, float * y, int64_t k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, int64_t k);
extern void dequantize_row_turbo4_0(const void * x, float * y, int64_t k);
extern void ggml_turbo_wht_128(const float * src, float * dst, int direction);

static float cosine(const float * a, const float * b, int n) {
    float dot = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
    for (int i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / sqrtf(na * nb);
}

static float mse(const float * a, const float * b, int n) {
    float error = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        error += d * d;
    }
    return error / n;
}

static void inverse_wht(float * values) {
    float tmp[128];
    ggml_turbo_wht_128(values, tmp, 1);
    memcpy(values, tmp, sizeof(tmp));
}

int main(void) {
    char packed[128];
    float input[128];
    float output[128];

    memset(input, 0, sizeof(input));
    input[0] = 1.0f;
    quantize_row_turbo3_0_ref(input, packed, 128);
    dequantize_row_turbo3_0(packed, output, 128);
    inverse_wht(output);

    const float turbo3_mse = mse(input, output, 128);
    const float turbo3_cos = cosine(input, output, 128);
    printf("turbo3 basis: mse=%.8f cosine=%.6f\n", (double) turbo3_mse, (double) turbo3_cos);
    if (turbo3_mse > 2e-9f || turbo3_cos < 0.999999f) {
        return 1;
    }

    for (int i = 0; i < 128; ++i) {
        input[i] = cosf(i * 0.2f) * 5.0f;
    }
    quantize_row_turbo4_0_ref(input, packed, 128);
    dequantize_row_turbo4_0(packed, output, 128);
    inverse_wht(output);

    const float turbo4_mse = mse(input, output, 128);
    const float turbo4_cos = cosine(input, output, 128);
    printf("turbo4 fixed: mse=%.9g cosine=%.9g\n", (double) turbo4_mse, (double) turbo4_cos);
    if (!isfinite(turbo4_mse) || turbo4_cos < 0.9955f) {
        return 1;
    }

    return 0;
}
