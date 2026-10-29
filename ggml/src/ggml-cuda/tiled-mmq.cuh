#pragma once

#include "common.cuh"

// Reads of tiled weights (see tiled.cu for the layout) for the mmq tile loaders (ggml_cuda_mmq_load_tiles_*_tiled, used
// by the mul_mat_q instances with tiled = true). The accessors return exactly the values a load from the native blocks
// would, so the rest of mmq is unchanged. Only plain 2D weights are tiled, so the absolute row and block of a load
// follow from kbx0 / stride.

// Start of the 16-row tile that holds `row`.
static __device__ __forceinline__ const uint8_t * ggml_cuda_tiled_tile(const char * x, int row, size_t row_bytes) {
    return (const uint8_t *) x + (size_t) (row / 16) * 16 * row_bytes;
}

// The 16-byte A fragment of `lane` for 32-element block (Q8_0) or block pair (Q4_K) `kb` of a tile: ints {row g, ints
// q and q + 4} x {row g + 8} with g = lane / 4, q = lane % 4, ordered (g, q), (g + 8, q), (g, q + 4), (g + 8, q + 4) for
// Q8_0 and (g, low nibbles), (g + 8, low), (g, high), (g + 8, high) for Q4_K. Consecutive lanes are contiguous.
static __device__ __forceinline__ int4 ggml_cuda_tiled_fragment(const uint8_t * tile, int kb, int lane) {
    return *(const int4 *) (tile + (kb * 32 + lane) * 16);
}

// Q8_0: native int j (0..7) of block b's qs, for row r (0..15) of the tile.
static __device__ __forceinline__ int ggml_cuda_tiled_q8_0_qs(const uint8_t * tile, int r, int b, int j) {
    const int lane = 4 * (r % 8) + j % 4;
    return *(const int *) (tile + (b * 32 + lane) * 16 + ((j / 4) * 2 + r / 8) * 4);
}

static __device__ __forceinline__ half ggml_cuda_tiled_q8_0_d(const uint8_t * tile, int k, int r, int b) {
    return *(const half *) (tile + (k / 32) * 512 + (b * 8 + r % 8) * 4 + (r / 8) * 2);
}

// Q4_K: native int j (0..31) of super-block sb's qs: low nibbles from block 2m, high nibbles from block 2m+1.
static __device__ __forceinline__ int ggml_cuda_tiled_q4_K_qs(const uint8_t * tile, int r, int sb, int j) {
    const int  kk   = 4 * (j % 8);
    const int  m    = sb * 4 + j / 8;
    const int  lane = 4 * (r % 8) + (kk % 16) / 4;
    const int  sh   = 4 * (kk / 16);
    const int * w   = (const int *) (tile + (m * 32 + lane) * 16);
    const unsigned a = ((unsigned) w[r / 8] >> sh) & 0x0F0F0F0Fu;
    const unsigned b = ((unsigned) w[2 + r / 8] >> sh) & 0x0F0F0F0Fu;
    return (int) (a | (b << 4));
}

// Q4_K: the native 16-byte header {dm, scales[12]} of super-block sb (only those fields may be read).
static __device__ __forceinline__ const block_q4_K * ggml_cuda_tiled_q4_K_header(const uint8_t * tile, int k, int r, int sb) {
    return (const block_q4_K *) (tile + (k / 32) * 256 + (sb * 16 + r) * 16);
}

// Q6_K: low 4 bits of block kb at the 4 positions of elements kk..kk+3 (kk % 4 == 0) of row r.
static __device__ __forceinline__ unsigned ggml_cuda_tiled_q6_K_lo(const uint8_t * tile, int r, int kb, int kk) {
    const int  lane = 4 * (r % 8) + (kk % 16) / 4;
    const int * w   = (const int *) (tile + ((kb / 2) * 32 + lane) * 16);
    return ((unsigned) w[(kb % 2) * 2 + r / 8] >> (4 * (kk / 16))) & 0x0F0F0F0Fu;
}

// Q6_K: high 2 bits (in bits 0..1 of each byte) of block 4u+q at elements kk..kk+3 of row r.
static __device__ __forceinline__ unsigned ggml_cuda_tiled_q6_K_hi(const uint8_t * tile, int k, int r, int u, int q, int kk) {
    const int  lane = 4 * (r % 8) + (kk % 16) / 4;
    const int * w   = (const int *) (tile + (k / 32) * 256 + (u * 32 + lane) * 16);
    return ((unsigned) w[q] >> (2 * (2 * (kk / 16) + r / 8))) & 0x03030303u;
}

// Q6_K: native int j (0..31) of super-block sb's ql (byte 4j+i: element l of block part, l+64 of block part+2).
static __device__ __forceinline__ int ggml_cuda_tiled_q6_K_ql(const uint8_t * tile, int r, int sb, int j) {
    const int n = (4 * j) / 64, part = ((4 * j) % 64) / 32, l = (4 * j) % 32;
    const int u = sb * 2 + n;
    return (int) (ggml_cuda_tiled_q6_K_lo(tile, r, 4 * u + part, l) | (ggml_cuda_tiled_q6_K_lo(tile, r, 4 * u + part + 2, l) << 4));
}

// Q6_K: native int j (0..15) of super-block sb's qh (byte 4j+i: high bits of elements q*32 + l, q = 0..3).
static __device__ __forceinline__ int ggml_cuda_tiled_q6_K_qh(const uint8_t * tile, int k, int r, int sb, int j) {
    const int n = (4 * j) / 32, l = (4 * j) % 32;
    const int u = sb * 2 + n;
    unsigned v = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        v |= ggml_cuda_tiled_q6_K_hi(tile, k, r, u, q, l) << (2 * q);
    }
    return (int) v;
}

static __device__ __forceinline__ int ggml_cuda_tiled_q6_K_scales(const uint8_t * tile, int k, int r, int sb, int j) {
    return *(const int *) (tile + (k / 32) * 384 + (sb * 16 + r) * 16 + 4 * j);
}

static __device__ __forceinline__ half ggml_cuda_tiled_q6_K_d(const uint8_t * tile, int k, int r, int sb) {
    return *(const half *) (tile + (k / 32) * 384 + (k / QK_K) * 256 + (sb * 16 + r) * 2);
}
