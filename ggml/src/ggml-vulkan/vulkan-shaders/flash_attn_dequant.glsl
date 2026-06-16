// Asymmetric K/V flash attention: aliased SSBO views of bindings 1 (K) and 2 (V)
// covering every supported FA element type, plus an uber dequantize4() that
// switches on FaTypeK / FaTypeV. After spec-constant specialization the driver
// folds away every path except the one matching the K/V type for this pipeline.
//
// Included by flash_attn.comp and flash_attn_cm1.comp. Not included by
// flash_attn_cm2.comp, which has its own buffer_reference-based decode path.
//
// We use macros (rather than per-quant decode functions taking a struct) on
// purpose: the FA shaders don't enable GL_EXT_shader_explicit_arithmetic_types_float16
// when FLOAT16 isn't defined, which makes float16-containing struct values
// illegal to return from / pass to functions. Macros expand inline where the
// float16 stays in storage and is converted to FLOAT_TYPE at use.

// F32 is fed as a vec4 "block" (4 floats), matching what dequant_funcs_cm2.glsl
// does for F32 in the cm2 shader. FaBlockBytesK/V == 16 for F32.
layout (binding = 1) readonly buffer K_PACKED_F32  { vec4 data[]; }                k_packed_f32;
layout (binding = 2) readonly buffer V_PACKED_F32  { vec4 data[]; }                v_packed_f32;

layout (binding = 1) readonly buffer K_PACKED_Q4_0 { block_q4_0_packed16 data[]; } k_packed_q4_0;
layout (binding = 2) readonly buffer V_PACKED_Q4_0 { block_q4_0_packed16 data[]; } v_packed_q4_0;
layout (binding = 1) readonly buffer K_PACKED_Q4_1 { block_q4_1_packed16 data[]; } k_packed_q4_1;
layout (binding = 2) readonly buffer V_PACKED_Q4_1 { block_q4_1_packed16 data[]; } v_packed_q4_1;
layout (binding = 1) readonly buffer K_PACKED_Q5_0 { block_q5_0_packed16 data[]; } k_packed_q5_0;
layout (binding = 2) readonly buffer V_PACKED_Q5_0 { block_q5_0_packed16 data[]; } v_packed_q5_0;
layout (binding = 1) readonly buffer K_PACKED_Q5_1 { block_q5_1_packed16 data[]; } k_packed_q5_1;
layout (binding = 2) readonly buffer V_PACKED_Q5_1 { block_q5_1_packed16 data[]; } v_packed_q5_1;
layout (binding = 1) readonly buffer K_PACKED_Q8_0 { block_q8_0_packed16 data[]; } k_packed_q8_0;
layout (binding = 2) readonly buffer V_PACKED_Q8_0 { block_q8_0_packed16 data[]; } v_packed_q8_0;
layout (binding = 1) readonly buffer K_PACKED_TURBO3_0 { block_turbo3_0_packed16 data[]; } k_packed_turbo3_0;
layout (binding = 2) readonly buffer V_PACKED_TURBO3_0 { block_turbo3_0_packed16 data[]; } v_packed_turbo3_0;
layout (binding = 1) readonly buffer K_PACKED_TURBO4_0 { block_turbo4_0_packed16 data[]; } k_packed_turbo4_0;
layout (binding = 2) readonly buffer V_PACKED_TURBO4_0 { block_turbo4_0_packed16 data[]; } v_packed_turbo4_0;

// Q4_1 and Q5_1 packed32 views: aliased to the same memory as the packed16
// views, used by the MMQ K-side hot path for fast 4-uint loads.
layout (binding = 1) readonly buffer K_PACKED_Q4_1_P32 { block_q4_1_packed32 data[]; } k_packed_q4_1_p32;
layout (binding = 1) readonly buffer K_PACKED_Q5_1_P32 { block_q5_1_packed32 data[]; } k_packed_q5_1_p32;

// Per-quant decode bodies are expanded once for the K view set and once for
// the V view set. The macros take the buffer name as a parameter.
#define FA_DEQUANT4_F32(BUF) \
    return FLOAT_TYPEV4(BUF.data[a_offset + ib]);

#define FA_DEQUANT4_Q4_0(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles - FLOAT_TYPE(8.0f));                  \
}

#define FA_DEQUANT4_Q4_1(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * nibbles                                        \
         + FLOAT_TYPE(BUF.data[a_offset + ib].m);                                                 \
}

#define FA_DEQUANT4_Q5_0(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = uint(BUF.data[a_offset + ib].qh[0])                                                 \
            | (uint(BUF.data[a_offset + ib].qh[1]) << 16);                                        \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles + hb - FLOAT_TYPE(16.0f));            \
}

#define FA_DEQUANT4_Q5_1(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = BUF.data[a_offset + ib].qh;                                                         \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles + hb)                                 \
         + FLOAT_TYPE(BUF.data[a_offset + ib].m);                                                 \
}

#define FA_DEQUANT4_Q8_0(BUF) {                                                                   \
    const i8vec2 v0 = unpack8(int32_t(BUF.data[a_offset + ib].qs[iqs / 2    ])).xy;               \
    const i8vec2 v1 = unpack8(int32_t(BUF.data[a_offset + ib].qs[iqs / 2 + 1])).xy;               \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);          \
}

#define FA_TURBO3_CENTROID(IDX)                                                                   \
    ((IDX) == 0u ? FLOAT_TYPE(-0.157341f) :                                                       \
     (IDX) == 1u ? FLOAT_TYPE(-0.108589f) :                                                       \
     (IDX) == 2u ? FLOAT_TYPE(-0.072199f) :                                                       \
     (IDX) == 3u ? FLOAT_TYPE(-0.035979f) :                                                       \
     (IDX) == 4u ? FLOAT_TYPE( 0.035979f) :                                                       \
     (IDX) == 5u ? FLOAT_TYPE( 0.072199f) :                                                       \
     (IDX) == 6u ? FLOAT_TYPE( 0.108589f) : FLOAT_TYPE(0.157341f))

#define FA_TURBO3_BYTE(ARR, BYTE_IDX)                                                            \
    ((uint(ARR[(BYTE_IDX) / 2u]) >> (8u * ((BYTE_IDX) & 1u))) & 0xffu)

#define FA_TURBO3_IDX(BUF, IDX)                                                                   \
    (((FA_TURBO3_BYTE(BUF.data[a_offset + ib].qs, (IDX) / 4u) >> (2u * ((IDX) % 4u))) & 0x3u) |   \
     (((FA_TURBO3_BYTE(BUF.data[a_offset + ib].signs, (IDX) / 8u) >> ((IDX) % 8u)) & 0x1u) << 2u))

#define FA_DEQUANT4_TURBO3_0(BUF) {                                                               \
    const uint idx0 = iqs + 0u;                                                                   \
    const uint idx1 = iqs + 1u;                                                                   \
    const uint idx2 = iqs + 2u;                                                                   \
    const uint idx3 = iqs + 3u;                                                                   \
    return FLOAT_TYPE(BUF.data[a_offset + ib].norm) * FLOAT_TYPEV4(                               \
        FA_TURBO3_CENTROID(FA_TURBO3_IDX(BUF, idx0)),                                             \
        FA_TURBO3_CENTROID(FA_TURBO3_IDX(BUF, idx1)),                                             \
        FA_TURBO3_CENTROID(FA_TURBO3_IDX(BUF, idx2)),                                             \
        FA_TURBO3_CENTROID(FA_TURBO3_IDX(BUF, idx3)));                                            \
}

#define FA_TURBO4_CENTROID(IDX)                                                                   \
    ((IDX) ==  0u ? FLOAT_TYPE(-0.167095f) :                                                      \
     (IDX) ==  1u ? FLOAT_TYPE(-0.140354f) :                                                      \
     (IDX) ==  2u ? FLOAT_TYPE(-0.118461f) :                                                      \
     (IDX) ==  3u ? FLOAT_TYPE(-0.098574f) :                                                      \
     (IDX) ==  4u ? FLOAT_TYPE(-0.079895f) :                                                      \
     (IDX) ==  5u ? FLOAT_TYPE(-0.061921f) :                                                      \
     (IDX) ==  6u ? FLOAT_TYPE(-0.044317f) :                                                      \
     (IDX) ==  7u ? FLOAT_TYPE(-0.026836f) :                                                      \
     (IDX) ==  8u ? FLOAT_TYPE( 0.026836f) :                                                      \
     (IDX) ==  9u ? FLOAT_TYPE( 0.044317f) :                                                      \
     (IDX) == 10u ? FLOAT_TYPE( 0.061921f) :                                                      \
     (IDX) == 11u ? FLOAT_TYPE( 0.079895f) :                                                      \
     (IDX) == 12u ? FLOAT_TYPE( 0.098574f) :                                                      \
     (IDX) == 13u ? FLOAT_TYPE( 0.118461f) :                                                      \
     (IDX) == 14u ? FLOAT_TYPE( 0.140354f) : FLOAT_TYPE(0.167095f))

#define FA_TURBO4_BYTE(ARR, BYTE_IDX)                                                            \
    ((uint(ARR[(BYTE_IDX) / 2u]) >> (8u * ((BYTE_IDX) & 1u))) & 0xffu)

#define FA_TURBO4_IDX(BUF, IDX)                                                                   \
    ((FA_TURBO4_BYTE(BUF.data[a_offset + ib].qs, (IDX) / 2u) >> (4u * ((IDX) & 1u))) & 0xfu)

#define FA_DEQUANT4_TURBO4_0(BUF) {                                                               \
    const uint idx0 = iqs + 0u;                                                                   \
    const uint idx1 = iqs + 1u;                                                                   \
    const uint idx2 = iqs + 2u;                                                                   \
    const uint idx3 = iqs + 3u;                                                                   \
    return FLOAT_TYPE(BUF.data[a_offset + ib].norm) * FLOAT_TYPEV4(                               \
        FA_TURBO4_CENTROID(FA_TURBO4_IDX(BUF, idx0)),                                             \
        FA_TURBO4_CENTROID(FA_TURBO4_IDX(BUF, idx1)),                                             \
        FA_TURBO4_CENTROID(FA_TURBO4_IDX(BUF, idx2)),                                             \
        FA_TURBO4_CENTROID(FA_TURBO4_IDX(BUF, idx3)));                                            \
}

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        switch (FaTypeK) {
            case FA_TYPE_F32:  FA_DEQUANT4_F32 (k_packed_f32)
            case FA_TYPE_Q4_0: FA_DEQUANT4_Q4_0(k_packed_q4_0)
            case FA_TYPE_Q4_1: FA_DEQUANT4_Q4_1(k_packed_q4_1)
            case FA_TYPE_Q5_0: FA_DEQUANT4_Q5_0(k_packed_q5_0)
            case FA_TYPE_Q5_1: FA_DEQUANT4_Q5_1(k_packed_q5_1)
            case FA_TYPE_Q8_0: FA_DEQUANT4_Q8_0(k_packed_q8_0)
            case FA_TYPE_TURBO3_0: FA_DEQUANT4_TURBO3_0(k_packed_turbo3_0)
            case FA_TYPE_TURBO4_0: FA_DEQUANT4_TURBO4_0(k_packed_turbo4_0)
        }
    } else {
        switch (FaTypeV) {
            case FA_TYPE_F32:  FA_DEQUANT4_F32 (v_packed_f32)
            case FA_TYPE_Q4_0: FA_DEQUANT4_Q4_0(v_packed_q4_0)
            case FA_TYPE_Q4_1: FA_DEQUANT4_Q4_1(v_packed_q4_1)
            case FA_TYPE_Q5_0: FA_DEQUANT4_Q5_0(v_packed_q5_0)
            case FA_TYPE_Q5_1: FA_DEQUANT4_Q5_1(v_packed_q5_1)
            case FA_TYPE_Q8_0: FA_DEQUANT4_Q8_0(v_packed_q8_0)
            case FA_TYPE_TURBO3_0: FA_DEQUANT4_TURBO3_0(v_packed_turbo3_0)
            case FA_TYPE_TURBO4_0: FA_DEQUANT4_TURBO4_0(v_packed_turbo4_0)
        }
    }
    return FLOAT_TYPEV4(0);
}
