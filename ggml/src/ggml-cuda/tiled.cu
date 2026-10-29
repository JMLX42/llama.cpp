#include "tiled.cuh"

#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

// Layout of one tile (16 rows, k columns) of a tiled weight, using exactly the bytes of the 16 native rows
// (kb = 32-element block, sb = 256-element super-block, lane = 4*g + t):
//   Q8_0: q [kb][lane] 16 int8 | d [kb][g] {fp16 d of row g, row g+8}
//   Q4_K: q [kb/2][lane] 8 B per block (byte j = value at position j | value at position j+8 << 4)
//         | meta [sb][row] native 16-byte header {d, dmin, scales[12]}
//   Q6_K: ql [kb/2][lane] low 4 bits packed like Q4_K | qh [kb/4][lane] 4 B per block (byte b = high 2 bits of
//         positions b | b+4 << 2 | b+8 << 4 | b+12 << 6) | scales [sb][row] 16 int8 | d [sb][row] fp16
// Fragment position p of a lane: p 0..3 row g, k 4t+p; p 4..7 row g+8, k 4t+p-4; p 8..11 row g, k 16+4t+p-8;
// p 12..15 row g+8, k 16+4t+p-12 (the a0..a3 registers of mma.m16n8k32).

static constexpr int kTileRows = 16;
static constexpr int kWarps    = 4;

static bool env_flag(const char * name) {
    const char * v = getenv(name);
    return v != nullptr && atoi(v) != 0;
}

static bool tiled_enabled() {
    static const bool enabled = env_flag("GGML_CUDA_TILED");
    return enabled;
}

// A tiled weight. Entries keep their own copy of the metadata: tensors and buffers may be freed (and their memory
// reused) while the registry still refers to the memory.
struct tiled_entry {
    ggml_type   type;
    int64_t     ne0;
    int64_t     ne1;
    size_t      nbytes;
    std::string name;
};

using tiled_map = std::map<uintptr_t, tiled_entry>;

struct tiled_registry {
    std::mutex                  mutex;
    std::map<uintptr_t, size_t> uploaded;  // data address -> bytes written so far, per eligible tensor
    tiled_map                   tiled;     // data address -> tiled weight (non-overlapping)
};

static tiled_registry & registry() {
    static tiled_registry r;
    return r;
}

static bool tiled_eligible(const ggml_tensor * t) {
    if (!tiled_enabled() || t->view_src != nullptr || t->buffer == nullptr) {
        return false;
    }
    if (t->type != GGML_TYPE_Q8_0 && t->type != GGML_TYPE_Q4_K && t->type != GGML_TYPE_Q6_K) {
        return false;
    }
    if (t->ne[2] != 1 || t->ne[3] != 1 || t->ne[0] % 256 != 0 || t->ne[1] % kTileRows != 0 || !ggml_is_contiguous(t)) {
        return false;
    }
    static const bool any_buffer = env_flag("GGML_CUDA_TILED_ANY_BUFFER");
    return any_buffer || ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

// A tiled weight overlapping [ptr, ptr + size), or tiled.end(). Caller holds the registry mutex.
static tiled_map::iterator find_tiled(const void * ptr, size_t size) {
    tiled_map & tiled = registry().tiled;
    if (tiled.empty() || ptr == nullptr) {
        return tiled.end();
    }
    const uintptr_t p   = (uintptr_t) ptr;
    const uintptr_t end = p + std::max<size_t>(size, 1);
    auto it = tiled.lower_bound(end);  // first entry starting at or after the range end
    if (it == tiled.begin()) {
        return tiled.end();
    }
    --it;
    return it->first + it->second.nbytes > p ? it : tiled.end();
}

// Drops the registry entries overlapping [ptr, ptr + size). Caller holds the registry mutex.
static void forget_range(const void * ptr, size_t size) {
    tiled_registry & reg = registry();
    for (auto it = find_tiled(ptr, size); it != reg.tiled.end(); it = find_tiled(ptr, size)) {
        reg.tiled.erase(it);
    }
    const uintptr_t p = (uintptr_t) ptr;
    reg.uploaded.erase(reg.uploaded.lower_bound(p), reg.uploaded.lower_bound(p + size));
}

// ------------------------------------------------------------------------------------------------ repacking

__device__ __forceinline__ void fragment_pos(int lane, int p, int & row, int & kk) {
    const int g = lane / 4, t = lane % 4;
    row = g + ((p / 4) % 2) * 8;
    kk  = (p / 8) * 16 + 4 * t + p % 4;
}

// Raw quant value of element e of a native row: int8 for Q8_0, 0..15 for Q4_K, 0..63 for Q6_K.
template <ggml_type type>
__device__ __forceinline__ int quant_value(const char * row, int e) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        return ((const block_q8_0 *) row)[e / QK8_0].qs[e % QK8_0];
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        const block_q4_K & b = ((const block_q4_K *) row)[e / QK_K];
        const int r = e % QK_K, rr = r % 64;
        const uint8_t byte = b.qs[(r / 64) * 32 + rr % 32];
        return rr < 32 ? (byte & 0xF) : (byte >> 4);
    } else {
        const block_q6_K & b = ((const block_q6_K *) row)[e / QK_K];
        const int r = e % QK_K, n = r / 128, q = (r % 128) / 32, l = r % 32;
        const uint8_t lo_byte = b.ql[n * 64 + l + (q & 1) * 32];
        const int lo = q >= 2 ? (lo_byte >> 4) : (lo_byte & 0xF);
        return lo | (((b.qh[n * 32 + l] >> (2 * q)) & 3) << 4);
    }
}

// One thread per (tile, block, lane): writes the lane's quant bytes of one 32-element block.
template <ggml_type type>
__global__ void tiled_repack_quants(const char * __restrict__ src, char * __restrict__ dst, int64_t tiles, int k, size_t row_bytes) {
    const int     nkb   = k / 32;
    const int64_t total = tiles * nkb * 32;
    for (int64_t u = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; u < total; u += (int64_t) gridDim.x * blockDim.x) {
        const int     lane = u % 32;
        const int     kb   = (u / 32) % nkb;
        const int64_t tile = u / (32 * nkb);
        int v[16];
#pragma unroll
        for (int p = 0; p < 16; ++p) {
            int r, kk;
            fragment_pos(lane, p, r, kk);
            v[p] = quant_value<type>(src + (tile * kTileRows + r) * row_bytes, kb * 32 + kk);
        }
        char * t = dst + tile * kTileRows * row_bytes;
        if constexpr (type == GGML_TYPE_Q8_0) {
#pragma unroll
            for (int p = 0; p < 16; ++p) {
                t[(kb * 32 + lane) * 16 + p] = (char) v[p];
            }
        } else {
            char * lo = t + ((kb / 2) * 32 + lane) * 16 + (kb % 2) * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                lo[j] = (char) ((v[j] & 0xF) | ((v[j + 8] & 0xF) << 4));
            }
            if constexpr (type == GGML_TYPE_Q6_K) {
                char * hi = t + nkb * 256 + ((kb / 4) * 32 + lane) * 16 + (kb % 4) * 4;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    hi[b] = (char) ((v[b] >> 4) | ((v[4 + b] >> 4) << 2) | ((v[8 + b] >> 4) << 4) | ((v[12 + b] >> 4) << 6));
                }
            }
        }
    }
}

// One thread per (tile, block or super-block, row): writes the scales.
template <ggml_type type>
__global__ void tiled_repack_scales(const char * __restrict__ src, char * __restrict__ dst, int64_t tiles, int k, size_t row_bytes) {
    const int     nkb    = k / 32;
    const int     nsb    = k / QK_K;
    const int     per    = type == GGML_TYPE_Q8_0 ? nkb : nsb;
    const int64_t total  = tiles * per * kTileRows;
    for (int64_t u = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; u < total; u += (int64_t) gridDim.x * blockDim.x) {
        const int     r    = u % kTileRows;
        const int     b    = (u / kTileRows) % per;
        const int64_t tile = u / (kTileRows * per);
        const char * row = src + (tile * kTileRows + r) * row_bytes;
        char * t = dst + tile * kTileRows * row_bytes;
        if constexpr (type == GGML_TYPE_Q8_0) {
            const char * d = (const char *) &((const block_q8_0 *) row)[b].d;
            char * out = t + nkb * 512 + (b * 8 + r % 8) * 4 + (r / 8) * 2;
            out[0] = d[0];
            out[1] = d[1];
        } else if constexpr (type == GGML_TYPE_Q4_K) {
            const char * hdr = (const char *) &((const block_q4_K *) row)[b];  // {d, dmin, scales[12]}
            char * out = t + nkb * 256 + (b * kTileRows + r) * 16;
            for (int i = 0; i < 16; ++i) {
                out[i] = hdr[i];
            }
        } else {
            const block_q6_K & blk = ((const block_q6_K *) row)[b];
            char * sc = t + nkb * 384 + (b * kTileRows + r) * 16;
            for (int i = 0; i < 16; ++i) {
                sc[i] = (char) blk.scales[i];
            }
            const char * d = (const char *) &blk.d;
            char * out = t + nkb * 384 + nsb * 256 + (b * kTileRows + r) * 2;
            out[0] = d[0];
            out[1] = d[1];
        }
    }
}

template <ggml_type type>
void launch_repack(const char * src, char * dst, int64_t tiles, int k, size_t row_bytes, cudaStream_t stream) {
    tiled_repack_quants<type><<<1024, 256, 0, stream>>>(src, dst, tiles, k, row_bytes);
    tiled_repack_scales<type><<<256, 256, 0, stream>>>(src, dst, tiles, k, row_bytes);
}

// Inverse of the repack kernels: one thread per lane's fragment bytes (coalesced 16-byte loads). A lane's 16 positions
// are 4 runs of 4 consecutive native elements (rows g / g+8, k 4t.. / 16+4t..), so native bytes are written as words.
// A unit covers the blocks needed to complete native bytes: 1 block (Q8_0), 2 (Q4_K: a byte holds elements of
// adjacent blocks) or 4 (Q6_K: a qh byte holds elements of 4 blocks).
__device__ __forceinline__ void store_u32_2aligned(uint8_t * p, unsigned v) {
    ((unsigned short *) p)[0] = (unsigned short) (v & 0xFFFFu);
    ((unsigned short *) p)[1] = (unsigned short) (v >> 16);
}

template <ggml_type type>
__global__ void tiled_unpack_quants(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int64_t tiles, int k, size_t row_bytes) {
    constexpr int per_unit = type == GGML_TYPE_Q8_0 ? 1 : type == GGML_TYPE_Q4_K ? 2 : 4;
    constexpr int bsize    = type == GGML_TYPE_Q8_0 ? sizeof(block_q8_0) : type == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q6_K);
    const int     nkb      = k / 32;
    const int     units    = nkb / per_unit;
    const int64_t total    = tiles * units * 32;
    for (int64_t u = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; u < total; u += (int64_t) gridDim.x * blockDim.x) {
        const int       lane = u % 32;
        const int       unit = (u / 32) % units;
        const int64_t   tile = u / (32 * units);
        const uint8_t * t    = src + tile * kTileRows * row_bytes;
        const int       g = lane / 4, tt = lane % 4;
        // run i: row (i % 2 ? g + 8 : g), first element (i < 2 ? 4t : 16 + 4t)
        uint8_t * rows[2] = {dst + (tile * kTileRows + g) * row_bytes, dst + (tile * kTileRows + g + 8) * row_bytes};
        if constexpr (type == GGML_TYPE_Q8_0) {
            const int4 w = *(const int4 *) (t + (unit * 32 + lane) * 16);
            const unsigned run[4] = {(unsigned) w.x, (unsigned) w.y, (unsigned) w.z, (unsigned) w.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                store_u32_2aligned(rows[i % 2] + unit * bsize + 2 + (i / 2) * 16 + 4 * tt, run[i]);
            }
        } else if constexpr (type == GGML_TYPE_Q4_K) {
            // unit m = 64-element group: block 2m in words x, y (byte j: positions j, j+8), block 2m+1 in words z, w
            const int4 w = *(const int4 *) (t + (unit * 32 + lane) * 16);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int      sh = (i / 2) * 4;
                const unsigned a  = (((unsigned) (i % 2 ? w.y : w.x)) >> sh) & 0x0F0F0F0Fu;
                const unsigned b  = (((unsigned) (i % 2 ? w.w : w.z)) >> sh) & 0x0F0F0F0Fu;
                *(unsigned *) (rows[i % 2] + (unit / 4) * bsize + 16 + (unit % 4) * 32 + (i / 2) * 16 + 4 * tt) = a | (b << 4);
            }
        } else {
            // unit = 128-element half n of super-block sb: blocks q = 0..3 (low 4 bits in l0/l1, high 2 bits in h)
            const int4 l0 = *(const int4 *) (t + ((unit * 2) * 32 + lane) * 16);
            const int4 l1 = *(const int4 *) (t + ((unit * 2 + 1) * 32 + lane) * 16);
            const int4 h  = *(const int4 *) (t + nkb * 256 + (unit * 32 + lane) * 16);
            const unsigned lw[4][2] = {{(unsigned) l0.x, (unsigned) l0.y}, {(unsigned) l0.z, (unsigned) l0.w},
                                       {(unsigned) l1.x, (unsigned) l1.y}, {(unsigned) l1.z, (unsigned) l1.w}};
            const unsigned hw[4]    = {(unsigned) h.x, (unsigned) h.y, (unsigned) h.z, (unsigned) h.w};
            const int      sb = unit / 2, n = unit % 2;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                unsigned lo[4], hi[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    lo[q] = (lw[q][i % 2] >> ((i / 2) * 4)) & 0x0F0F0F0Fu;
                    hi[q] = (hw[q] >> (2 * i)) & 0x03030303u;
                }
                uint8_t * blk = rows[i % 2] + sb * bsize;
                const int kk  = (i / 2) * 16 + 4 * tt;
                store_u32_2aligned(blk + n * 64 + kk, lo[0] | (lo[2] << 4));       // ql: elements l, l+64
                store_u32_2aligned(blk + n * 64 + 32 + kk, lo[1] | (lo[3] << 4));  // ql: elements l+32, l+96
                store_u32_2aligned(blk + 128 + n * 32 + kk, hi[0] | (hi[1] << 2) | (hi[2] << 4) | (hi[3] << 6));
            }
        }
    }
}

// One thread per (tile, block or super-block, row): writes the native scales back.
template <ggml_type type>
__global__ void tiled_unpack_scales(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int64_t tiles, int k, size_t row_bytes) {
    constexpr int bsize = type == GGML_TYPE_Q8_0 ? sizeof(block_q8_0) : type == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q6_K);
    const int     nkb   = k / 32;
    const int     nsb   = k / QK_K;
    const int     per   = type == GGML_TYPE_Q8_0 ? nkb : nsb;
    const int64_t total = tiles * per * kTileRows;
    for (int64_t u = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; u < total; u += (int64_t) gridDim.x * blockDim.x) {
        const int       r    = u % kTileRows;
        const int       b    = (u / kTileRows) % per;
        const int64_t   tile = u / (kTileRows * per);
        const uint8_t * t    = src + tile * kTileRows * row_bytes;
        uint8_t *       blk  = dst + (tile * kTileRows + r) * row_bytes + b * bsize;
        if constexpr (type == GGML_TYPE_Q8_0) {
            *(unsigned short *) blk = *(const unsigned short *) (t + nkb * 512 + (b * 8 + r % 8) * 4 + (r / 8) * 2);
        } else if constexpr (type == GGML_TYPE_Q4_K) {  // header {d, dmin, scales[12]}; both sides 16-byte aligned
            *(int4 *) blk = *(const int4 *) (t + nkb * 256 + (b * kTileRows + r) * 16);
        } else {  // scales[16] then d, at 2-byte aligned native offsets
            const unsigned short * sc = (const unsigned short *) (t + nkb * 384 + (b * kTileRows + r) * 16);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                ((unsigned short *) (blk + 192))[i] = sc[i];
            }
            *(unsigned short *) (blk + 208) = *(const unsigned short *) (t + nkb * 384 + nsb * 256 + (b * kTileRows + r) * 2);
        }
    }
}

template <ggml_type type>
static void launch_unpack(const uint8_t * src, uint8_t * dst, int64_t tiles, int k, size_t row_bytes, cudaStream_t stream) {
    tiled_unpack_quants<type><<<1024, 256, 0, stream>>>(src, dst, tiles, k, row_bytes);
    tiled_unpack_scales<type><<<256, 256, 0, stream>>>(src, dst, tiles, k, row_bytes);
}

static void unpack_tiles(ggml_type type, const uint8_t * src, uint8_t * dst, int64_t tiles, int k, size_t row_bytes, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q8_0: launch_unpack<GGML_TYPE_Q8_0>(src, dst, tiles, k, row_bytes, stream); break;
        case GGML_TYPE_Q4_K: launch_unpack<GGML_TYPE_Q4_K>(src, dst, tiles, k, row_bytes, stream); break;
        case GGML_TYPE_Q6_K: launch_unpack<GGML_TYPE_Q6_K>(src, dst, tiles, k, row_bytes, stream); break;
        default: GGML_ABORT("unsupported tiled type");
    }
    CUDA_CHECK(cudaGetLastError());
}

static void * scratch_buffer(size_t & size) {
    static void * scratch[GGML_CUDA_MAX_DEVICES] = {};
    size = 32u << 20;
    const int device = ggml_cuda_get_device();
    if (scratch[device] == nullptr) {
        CUDA_CHECK(cudaMalloc(&scratch[device], size));
    }
    return scratch[device];
}

// Copies bytes [offset, offset + size) of tiled tensor `t`, rebuilt as native rows, to host memory `data`.
static void read_native(const uint8_t * base, const tiled_entry & t, void * data, size_t offset, size_t size, cudaStream_t stream) {
    size_t    scratch_size;
    uint8_t * scratch    = (uint8_t *) scratch_buffer(scratch_size);
    const int     k          = (int) t.ne0;
    const size_t  row_bytes  = ggml_row_size(t.type, k);
    const size_t  tile_bytes = kTileRows * row_bytes;
    const int64_t chunk      = std::max<int64_t>(1, scratch_size / tile_bytes);
    for (int64_t t0 = offset / tile_bytes; (size_t) t0 * tile_bytes < offset + size; t0 += chunk) {
        const int64_t   n   = std::min(chunk, t.ne1 / kTileRows - t0);
        const uint8_t * src = base + t0 * tile_bytes;
        unpack_tiles(t.type, src, scratch, n, k, row_bytes, stream);
        const size_t begin = std::max<size_t>(offset, t0 * tile_bytes);
        const size_t end   = std::min<size_t>(offset + size, (t0 + n) * tile_bytes);
        CUDA_CHECK(cudaMemcpyAsync((char *) data + (begin - offset), scratch + (begin - t0 * tile_bytes), end - begin,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));  // the scratch buffer is reused by the next chunk
    }
}

// Rearranges `t` in place, a chunk of tiles at a time through a scratch buffer.
static void repack_tensor(ggml_tensor * t, cudaStream_t stream) {
    size_t scratch_size;
    void * scratch = scratch_buffer(scratch_size);
    const int     k          = (int) t->ne[0];
    const size_t  row_bytes  = ggml_row_size(t->type, k);
    const size_t  tile_bytes = kTileRows * row_bytes;
    const int64_t tiles      = t->ne[1] / kTileRows;
    const int64_t chunk      = std::max<int64_t>(1, scratch_size / tile_bytes);
    GGML_ASSERT(tile_bytes <= scratch_size);

    for (int64_t t0 = 0; t0 < tiles; t0 += chunk) {
        const int64_t n   = std::min(chunk, tiles - t0);
        char *        src = (char *) t->data + t0 * tile_bytes;
        char *        dst = (char *) scratch;
        switch (t->type) {
            case GGML_TYPE_Q8_0: launch_repack<GGML_TYPE_Q8_0>(src, dst, n, k, row_bytes, stream); break;
            case GGML_TYPE_Q4_K: launch_repack<GGML_TYPE_Q4_K>(src, dst, n, k, row_bytes, stream); break;
            case GGML_TYPE_Q6_K: launch_repack<GGML_TYPE_Q6_K>(src, dst, n, k, row_bytes, stream); break;
            default: GGML_ABORT("unsupported tiled type");
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(src, dst, n * tile_bytes, cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));  // the scratch buffer is reused by the next tensor
}

// ------------------------------------------------------------------------------------------------ mma kernels

__device__ __forceinline__ unsigned word(const int4 & v, int i) {
    return (unsigned) (i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w);
}

__device__ __forceinline__ int byte_at(const int4 & v, int i) {
    return (word(v, i >> 2) >> (8 * (i & 3))) & 0xFF;
}

__device__ __forceinline__ float half_bits(unsigned bits) {
    return __half2float(__ushort_as_half((unsigned short) (bits & 0xFFFFu)));
}

__device__ __forceinline__ void mma_k32(int (&c)[4], const int (&a)[4], int b0, int b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif
}

__device__ __forceinline__ void mma_k16(int (&c)[4], int a0, int a1, int b) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b));
#else
    GGML_UNUSED_VARS(c, a0, a1, b);
    NO_DEVICE_CODE;
#endif
}

// Per-row factors of one 32-element block for C rows g and g+8: value = s * q - m (Q6_K: two 16-element halves).
struct row_scales {
    float s_g, s_g8, m_g, m_g8;
};

struct tiled_q8_0 {
    static constexpr int  blocks_per_step = 4;
    static constexpr bool k16             = false;
    static constexpr bool has_min         = false;

    struct tile_t { const int4 * q; const unsigned * d; };
    struct item_t { int4 w[4]; unsigned d[4]; };

    static __device__ tile_t tile(const char * base, int k) {
        return {(const int4 *) base, (const unsigned *) (base + (k / 32) * 512)};
    }
    static __device__ item_t load(const tile_t & tl, int step, int lane) {
        item_t it;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int kb = step * 4 + i;
            it.w[i] = __ldg(tl.q + kb * 32 + lane);
            it.d[i] = __ldg(tl.d + kb * 8 + lane / 4);
        }
        return it;
    }
    static __device__ void a_frag(const item_t & it, int j, int (&a)[4]) {
        a[0] = it.w[j].x; a[1] = it.w[j].y; a[2] = it.w[j].z; a[3] = it.w[j].w;
    }
    static __device__ row_scales scales(const item_t & it, int j) {
        return {half_bits(it.d[j]), half_bits(it.d[j] >> 16), 0.0f, 0.0f};
    }
};

struct tiled_q4_K {
    static constexpr int  blocks_per_step = 8;
    static constexpr bool k16             = false;
    static constexpr bool has_min         = true;

    struct tile_t { const int4 * q; const int4 * meta; };
    struct item_t { int4 w[4]; int4 m_g, m_g8; };

    static __device__ tile_t tile(const char * base, int k) {
        return {(const int4 *) base, (const int4 *) (base + (k / 32) * 256)};
    }
    static __device__ item_t load(const tile_t & tl, int sb, int lane) {
        item_t it;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            it.w[i] = __ldg(tl.q + (sb * 4 + i) * 32 + lane);
        }
        it.m_g  = __ldg(tl.meta + sb * kTileRows + lane / 4);
        it.m_g8 = __ldg(tl.meta + sb * kTileRows + lane / 4 + 8);
        return it;
    }
    static __device__ void a_frag(const item_t & it, int j, int (&a)[4]) {
        const int4 & w  = it.w[j / 2];
        const unsigned u0 = (unsigned) ((j & 1) ? w.z : w.x), u1 = (unsigned) ((j & 1) ? w.w : w.y);
        a[0] = (int) (u0 & 0x0F0F0F0Fu);
        a[1] = (int) (u1 & 0x0F0F0F0Fu);
        a[2] = (int) ((u0 >> 4) & 0x0F0F0F0Fu);
        a[3] = (int) ((u1 >> 4) & 0x0F0F0F0Fu);
    }
    // ggml's get_scale_min_k4 on the native header {d, dmin, scales[12]}.
    static __device__ void scale_min(const int4 & m, int j, float & s, float & mn) {
        int sc, mm;
        if (j < 4) {
            sc = byte_at(m, 4 + j) & 63;
            mm = byte_at(m, 4 + j + 4) & 63;
        } else {
            sc = (byte_at(m, 4 + j + 4) & 0xF) | ((byte_at(m, 4 + j - 4) >> 6) << 4);
            mm = (byte_at(m, 4 + j + 4) >> 4) | ((byte_at(m, 4 + j) >> 6) << 4);
        }
        s  = half_bits(word(m, 0)) * (float) sc;
        mn = half_bits(word(m, 0) >> 16) * (float) mm;
    }
    static __device__ row_scales scales(const item_t & it, int j) {
        row_scales r;
        scale_min(it.m_g, j, r.s_g, r.m_g);
        scale_min(it.m_g8, j, r.s_g8, r.m_g8);
        return r;
    }
};

struct tiled_q6_K {
    static constexpr int  blocks_per_step = 8;
    static constexpr bool k16             = true;  // 16-element scales: two k16 mmas per block

    struct tile_t { const int4 * ql; const int4 * qh; const int4 * sc; const unsigned short * d; };
    struct item_t { int4 ql[4]; int4 qh[2]; int4 sc_g, sc_g8; unsigned short d_g, d_g8; };

    static __device__ tile_t tile(const char * base, int k) {
        const int nkb = k / 32;
        return {(const int4 *) base, (const int4 *) (base + nkb * 256), (const int4 *) (base + nkb * 384),
                (const unsigned short *) (base + nkb * 384 + (k / QK_K) * 256)};
    }
    static __device__ item_t load(const tile_t & tl, int sb, int lane) {
        const int g = lane / 4;
        item_t it;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            it.ql[i] = __ldg(tl.ql + (sb * 4 + i) * 32 + lane);
        }
        it.qh[0] = __ldg(tl.qh + (sb * 2) * 32 + lane);
        it.qh[1] = __ldg(tl.qh + (sb * 2 + 1) * 32 + lane);
        it.sc_g  = __ldg(tl.sc + sb * kTileRows + g);
        it.sc_g8 = __ldg(tl.sc + sb * kTileRows + g + 8);
        it.d_g   = __ldg(tl.d + sb * kTileRows + g);
        it.d_g8  = __ldg(tl.d + sb * kTileRows + g + 8);
        return it;
    }
    static __device__ void a_frag(const item_t & it, int j, int (&a)[4]) {
        const int4 & l = it.ql[j / 2];
        const unsigned l0 = (unsigned) ((j & 1) ? l.z : l.x), l1 = (unsigned) ((j & 1) ? l.w : l.y);
        const unsigned h  = word(it.qh[j / 4], j % 4);
        a[0] = (int) __vsub4((l0 & 0x0F0F0F0Fu) | ((h & 0x03030303u) << 4), 0x20202020u);
        a[1] = (int) __vsub4((l1 & 0x0F0F0F0Fu) | (((h >> 2) & 0x03030303u) << 4), 0x20202020u);
        a[2] = (int) __vsub4(((l0 >> 4) & 0x0F0F0F0Fu) | (((h >> 4) & 0x03030303u) << 4), 0x20202020u);
        a[3] = (int) __vsub4(((l1 >> 4) & 0x0F0F0F0Fu) | (((h >> 6) & 0x03030303u) << 4), 0x20202020u);
    }
    // s = d * scale of the block's first 16 elements, m = d * scale of the second 16 (not a min).
    static __device__ row_scales scales(const item_t & it, int j) {
        const float d_g = half_bits(it.d_g), d_g8 = half_bits(it.d_g8);
        return {d_g * (float) (int8_t) byte_at(it.sc_g, 2 * j), d_g8 * (float) (int8_t) byte_at(it.sc_g8, 2 * j),
                d_g * (float) (int8_t) byte_at(it.sc_g, 2 * j + 1), d_g8 * (float) (int8_t) byte_at(it.sc_g8, 2 * j + 1)};
    }
};

// Activation of NCB 8-column blocks: xb [cb][kb][lane] {b0, b1}; xd, xds (= d * sum) [cb][kb][8].
template <typename P, int KSPLIT, int NCB>
__global__ void __launch_bounds__(kWarps * 32) tiled_mma_kernel(
        const char * __restrict__ w, int rows, int k, size_t row_bytes,
        const int2 * __restrict__ xb, const float * __restrict__ xd, const float * __restrict__ xds,
        int ncols, float * __restrict__ y) {
    const int warp  = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int tile  = blockIdx.x * (kWarps / KSPLIT) + warp / KSPLIT;
    const int part  = warp % KSPLIT;
    const bool valid = tile < rows / kTileRows;  // invalid warps still reach __syncthreads
    const int nkb   = k / 32;
    const int per   = nkb / P::blocks_per_step / KSPLIT;
    const int begin = part * per, end = begin + per;

    float acc[NCB][4] = {};
    if (valid) {
        const typename P::tile_t tl = P::tile(w + (size_t) tile * kTileRows * row_bytes, k);
        typename P::item_t cur = P::load(tl, begin, lane);
        for (int s = begin; s < end; ++s) {
            // Double buffering: step s+1's weights stream in while step s is computed.
            typename P::item_t next = cur;
            if (s + 1 < end) {
                next = P::load(tl, s + 1, lane);
            }
#pragma unroll
            for (int j = 0; j < P::blocks_per_step; ++j) {
                const int kb = s * P::blocks_per_step + j;
                int a[4];
                P::a_frag(cur, j, a);
                const row_scales sc = P::scales(cur, j);
#pragma unroll
                for (int cb = 0; cb < NCB; ++cb) {
                    const int2 b = __ldg(xb + ((size_t) cb * nkb + kb) * 32 + lane);
                    int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
                    if constexpr (P::k16) {
                        mma_k16(c0, a[0], a[1], b.x);
                        mma_k16(c1, a[2], a[3], b.y);
                    } else {
                        mma_k32(c0, a, b.x, b.y);
                    }
                    const float2 dx = __ldg((const float2 *) (xd + ((size_t) cb * nkb + kb) * 8) + (lane & 3));
                    if constexpr (P::k16) {
                        acc[cb][0] += dx.x * (sc.s_g  * (float) c0[0] + sc.m_g  * (float) c1[0]);
                        acc[cb][1] += dx.y * (sc.s_g  * (float) c0[1] + sc.m_g  * (float) c1[1]);
                        acc[cb][2] += dx.x * (sc.s_g8 * (float) c0[2] + sc.m_g8 * (float) c1[2]);
                        acc[cb][3] += dx.y * (sc.s_g8 * (float) c0[3] + sc.m_g8 * (float) c1[3]);
                    } else {
                        acc[cb][0] += sc.s_g  * (dx.x * (float) c0[0]);
                        acc[cb][1] += sc.s_g  * (dx.y * (float) c0[1]);
                        acc[cb][2] += sc.s_g8 * (dx.x * (float) c0[2]);
                        acc[cb][3] += sc.s_g8 * (dx.y * (float) c0[3]);
                        if constexpr (P::has_min) {
                            const float2 dsx = __ldg((const float2 *) (xds + ((size_t) cb * nkb + kb) * 8) + (lane & 3));
                            acc[cb][0] -= sc.m_g  * dsx.x;
                            acc[cb][1] -= sc.m_g  * dsx.y;
                            acc[cb][2] -= sc.m_g8 * dsx.x;
                            acc[cb][3] -= sc.m_g8 * dsx.y;
                        }
                    }
                }
            }
            cur = next;
        }
    }
    if constexpr (KSPLIT > 1) {
        __shared__ float partial[kWarps][32][NCB * 4];
#pragma unroll
        for (int cb = 0; cb < NCB; ++cb) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                partial[warp][lane][cb * 4 + i] = acc[cb][i];
            }
        }
        __syncthreads();
        if (part != 0) {
            return;
        }
        for (int p = 1; p < KSPLIT; ++p) {
#pragma unroll
            for (int cb = 0; cb < NCB; ++cb) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[cb][i] += partial[warp + p][lane][cb * 4 + i];
                }
            }
        }
    }
    if (!valid) {
        return;
    }
    const int row = tile * kTileRows + lane / 4;
#pragma unroll
    for (int cb = 0; cb < NCB; ++cb) {
        const int col = cb * 8 + 2 * (lane % 4);
        if (col < ncols) {
            y[(size_t) col * rows + row]     = acc[cb][0];
            y[(size_t) col * rows + row + 8] = acc[cb][2];
        }
        if (col + 1 < ncols) {
            y[(size_t) (col + 1) * rows + row]     = acc[cb][1];
            y[(size_t) (col + 1) * rows + row + 8] = acc[cb][3];
        }
    }
}

// One warp per (column, 32-element block): quantizes like ggml's quantize_q8_1 into B-fragment order. Columns in
// [ncols, ncols_padded) are written as zeros.
__global__ void tiled_quantize(const float * __restrict__ x, int k, int64_t ncols, int64_t ncols_padded,
                               char * __restrict__ xb, float * __restrict__ xd, float * __restrict__ xds) {
    const int     nkb  = k / 32;
    const int64_t blk  = blockIdx.x * (int64_t) (blockDim.x / 32) + threadIdx.x / 32;
    const int     lane = threadIdx.x % 32;
    if (blk >= ncols_padded * nkb) {
        return;
    }
    const int64_t col = blk / nkb;
    const int     kb  = blk % nkb;
    const float   v   = col < ncols ? x[col * k + kb * 32 + lane] : 0.0f;
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, o));
    }
    const float d  = amax / 127.0f;
    const int   qi = amax == 0.0f ? 0 : (int) roundf(v / d);
    int sum = qi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, o);
    }
    const int64_t cb    = col / 8;
    const int     c     = col % 8;
    const int     flane = 4 * c + (lane % 16) / 4;
    xb[((cb * nkb + kb) * 32 + flane) * 8 + (lane / 16) * 4 + lane % 4] = (char) qi;
    if (lane == 0) {
        xd[(cb * nkb + kb) * 8 + c]  = d;
        xds[(cb * nkb + kb) * 8 + c] = d * (float) sum;
    }
}

template <typename P, int KSPLIT, int NCB>
void launch_mma(const char * w, int rows, int k, size_t row_bytes, const int2 * xb, const float * xd, const float * xds,
                int ncols, float * y, cudaStream_t stream) {
    constexpr int tiles_per_block = kWarps / KSPLIT;
    const int     tiles           = rows / kTileRows;
    tiled_mma_kernel<P, KSPLIT, NCB><<<(tiles + tiles_per_block - 1) / tiles_per_block, kWarps * 32, 0, stream>>>(
        w, rows, k, row_bytes, xb, xd, xds, ncols, y);
}

template <typename P, int KSPLIT>
void launch_mma_ncb(int ncb, const char * w, int rows, int k, size_t row_bytes, const int2 * xb, const float * xd,
                    const float * xds, int ncols, float * y, cudaStream_t stream) {
    switch (ncb) {
        case 1: launch_mma<P, KSPLIT, 1>(w, rows, k, row_bytes, xb, xd, xds, ncols, y, stream); break;
        case 2: launch_mma<P, KSPLIT, 2>(w, rows, k, row_bytes, xb, xd, xds, ncols, y, stream); break;
        case 4: launch_mma<P, KSPLIT, 4>(w, rows, k, row_bytes, xb, xd, xds, ncols, y, stream); break;
        case 8: launch_mma<P, KSPLIT, 8>(w, rows, k, row_bytes, xb, xd, xds, ncols, y, stream); break;
        default: GGML_ABORT("unsupported column block count");
    }
}

template <typename P>
void launch_mma_type(int ksplit, int ncb, const char * w, int rows, int k, size_t row_bytes, const int2 * xb,
                     const float * xd, const float * xds, int ncols, float * y, cudaStream_t stream) {
    switch (ksplit) {
        case 1: launch_mma_ncb<P, 1>(ncb, w, rows, k, row_bytes, xb, xd, xds, ncols, y, stream); break;
        case 2: launch_mma_ncb<P, 2>(ncb, w, rows, k, row_bytes, xb, xd, xds, ncols, y, stream); break;
        case 4: launch_mma_ncb<P, 4>(ncb, w, rows, k, row_bytes, xb, xd, xds, ncols, y, stream); break;
        default: GGML_ABORT("unsupported k split");
    }
}

// Splits k across the warps of a block when there are too few tiles to fill the GPU (tuned on Jetson Orin Nano).
static int choose_ksplit(int tiles, int steps) {
    if (tiles >= 2048) {
        return 1;
    }
    if (steps % 4 == 0) {
        return 4;
    }
    return steps % 2 == 0 ? 2 : 1;
}

static int column_blocks_for(int64_t remaining_cb) {
    return remaining_cb >= 8 ? 8 : remaining_cb > 2 ? 4 : (int) remaining_cb;
}


void ggml_cuda_tiled_on_upload(ggml_tensor * tensor, size_t offset, size_t size, cudaStream_t stream) {
    if (!tiled_enabled()) {
        return;
    }
    tiled_registry & reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    forget_range((const char *) tensor->data + offset, size);  // (re)written with native data
    if (!tiled_eligible(tensor)) {
        return;
    }
    const size_t nbytes = ggml_nbytes(tensor);
    size_t &     done   = reg.uploaded[(uintptr_t) tensor->data];
    done += size;
    if (done < nbytes) {
        return;
    }
    reg.uploaded.erase((uintptr_t) tensor->data);
    repack_tensor(tensor, stream);
    reg.tiled[(uintptr_t) tensor->data] = {tensor->type, tensor->ne[0], tensor->ne[1], nbytes, tensor->name};
}

void ggml_cuda_tiled_forget(const void * ptr, size_t size) {
    if (!tiled_enabled()) {
        return;
    }
    tiled_registry & reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    forget_range(ptr, size);
}

bool ggml_cuda_tiled_overlaps(const void * ptr, size_t size) {
    if (!tiled_enabled()) {
        return false;
    }
    tiled_registry & reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    return find_tiled(ptr, size) != reg.tiled.end();
}

bool ggml_cuda_tiled_is_tiled(const ggml_tensor * tensor) {
    if (!tiled_enabled() || tensor == nullptr) {
        return false;
    }
    tiled_registry & reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.tiled.find((uintptr_t) tensor->data);
    return it != reg.tiled.end() && it->second.type == tensor->type && it->second.ne0 == tensor->ne[0] &&
           it->second.ne1 == tensor->ne[1] && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

void ggml_cuda_tiled_check_node(const ggml_tensor * node) {
    if (!tiled_enabled()) {
        return;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * src = node->src[i];
        if (src == nullptr) {
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && i == 0 && ggml_cuda_tiled_is_tiled(src)) {
            continue;
        }
        tiled_registry & reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto it = find_tiled(src->data, ggml_nbytes(src));
        if (it != reg.tiled.end()) {
            GGML_ABORT("%s: op %s (%s) reads tiled weight %s, which only MUL_MAT supports; unset GGML_CUDA_TILED",
                       __func__, ggml_op_desc(node), node->name, it->second.name.c_str());
        }
    }
}

bool ggml_cuda_tiled_read(const ggml_tensor * tensor, void * data, size_t offset, size_t size, cudaStream_t stream) {
    if (!tiled_enabled()) {
        return false;
    }
    tiled_registry & reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = find_tiled((const char *) tensor->data + offset, size);
    if (it == reg.tiled.end()) {
        return false;
    }
    GGML_ASSERT(it->first == (uintptr_t) tensor->data && "reading a view of a tiled weight is not supported");
    read_native((const uint8_t *) tensor->data, it->second, data, offset, size, stream);
    return true;
}

bool ggml_cuda_tiled_large_batch(const ggml_tensor * src1) {
    // The tiled kernel re-reads the weights for every 64 columns, so prompt-sized batches go to llama.cpp's kernels.
    static const int64_t max_cols = [] {
        const char * v = getenv("GGML_CUDA_TILED_MAX_COLS");
        return v != nullptr ? (int64_t) atoll(v) : (int64_t) 8;
    }();
    return ggml_nrows(src1) > max_cols;
}

bool ggml_cuda_tiled_native_copy_ok(const ggml_tensor * src0) {
    return ggml_nbytes(src0) <= (64u << 20);
}

size_t ggml_cuda_tiled_native_copy_size(const ggml_tensor * src0) {
    return ggml_nbytes(src0) + ggml_row_size(src0->type, MATRIX_ROW_PADDING);
}

void ggml_cuda_tiled_to_native(const ggml_tensor * src0, void * dst, cudaStream_t stream) {
    const int       k         = (int) src0->ne[0];
    const size_t    row_bytes = ggml_row_size(src0->type, k);
    const int64_t   tiles     = src0->ne[1] / kTileRows;
    const uint8_t * src       = (const uint8_t *) src0->data;
    unpack_tiles(src0->type, src, (uint8_t *) dst, tiles, k, row_bytes, stream);
    // native kernels may read up to MATRIX_ROW_PADDING elements past the last row
    CUDA_CHECK(cudaMemsetAsync((char *) dst + ggml_nbytes(src0), 0, ggml_row_size(src0->type, MATRIX_ROW_PADDING), stream));
}

void ggml_cuda_tiled_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src1) && ggml_is_contiguous(dst));
    GGML_ASSERT(src1->ne[0] == src0->ne[0] && dst->ne[0] == src0->ne[1]);

    const int     k         = (int) src0->ne[0];
    const int     rows      = (int) src0->ne[1];
    const size_t  row_bytes = ggml_row_size(src0->type, k);
    const int64_t ncols     = ggml_nrows(src1);  // src0 is 2D, so every src1 row is a column of the product
    const int     nkb       = k / 32;
    const int     bps       = src0->type == GGML_TYPE_Q8_0 ? tiled_q8_0::blocks_per_step : 8;
    const int     ksplit    = choose_ksplit(rows / kTileRows, nkb / bps);

    // Column blocks, rounded up to the block count of the last launch.
    int64_t ncb = 0;
    for (int64_t remaining = (ncols + 7) / 8; remaining > 0;) {
        const int n = column_blocks_for(remaining);
        ncb += n;
        remaining -= std::min<int64_t>(n, remaining);
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int2>  xb(ctx.pool(), ncb * nkb * 32);
    ggml_cuda_pool_alloc<float> xd(ctx.pool(), ncb * nkb * 8);
    ggml_cuda_pool_alloc<float> xds(ctx.pool(), ncb * nkb * 8);

    const int64_t qwarps = ncb * 8 * nkb;
    tiled_quantize<<<(qwarps + 3) / 4, 128, 0, stream>>>((const float *) src1->data, k, ncols, ncb * 8,
                                                         (char *) xb.get(), xd.get(), xds.get());

    const char * w = (const char *) src0->data;
    float *      y = (float *) dst->data;
    for (int64_t cb0 = 0; cb0 < ncb;) {
        const int n         = column_blocks_for(ncb - cb0);
        const int cols_left = (int) std::min<int64_t>(ncols - cb0 * 8, n * 8);
        const int2 *  b     = xb.get() + cb0 * nkb * 32;
        const float * d     = xd.get() + cb0 * nkb * 8;
        const float * ds    = xds.get() + cb0 * nkb * 8;
        float *       out   = y + cb0 * 8 * rows;
        switch (src0->type) {
            case GGML_TYPE_Q8_0: launch_mma_type<tiled_q8_0>(ksplit, n, w, rows, k, row_bytes, b, d, ds, cols_left, out, stream); break;
            case GGML_TYPE_Q4_K: launch_mma_type<tiled_q4_K>(ksplit, n, w, rows, k, row_bytes, b, d, ds, cols_left, out, stream); break;
            case GGML_TYPE_Q6_K: launch_mma_type<tiled_q6_K>(ksplit, n, w, rows, k, row_bytes, b, d, ds, cols_left, out, stream); break;
            default: GGML_ABORT("unsupported tiled type");
        }
        cb0 += n;
    }
    CUDA_CHECK(cudaGetLastError());
}
