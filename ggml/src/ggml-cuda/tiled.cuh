#pragma once

#include "common.cuh"

// Tiled weights (enabled with GGML_CUDA_TILED=1).
//
// Q8_0 / Q4_K / Q6_K weight matrices (2D, ne0 % 256 == 0, ne1 % 16 == 0, in a weights buffer) are rearranged in place
// once their upload completes: rows are grouped in tiles of 16 and each 32-element block of a tile is stored in the
// per-lane A-fragment order of mma.sync m16n8k32 (s8), so that a lane fetches its fragment with one 16-byte load.
// Every MUL_MAT on such a weight then runs on int8 tensor-core kernels, for any batch size.
// GGML_CUDA_TILED_ANY_BUFFER=1 also tiles tensors outside weight buffers (for test-backend-ops).

// Called after (part of) a tensor was uploaded on `stream`; tiles the tensor once all of its bytes have been written.
void ggml_cuda_tiled_on_upload(ggml_tensor * tensor, size_t offset, size_t size, cudaStream_t stream);

// Forgets tiled weights in [ptr, ptr + size) (memory freed, cleared or overwritten by another path).
void ggml_cuda_tiled_forget(const void * ptr, size_t size);

// Whether [ptr, ptr + size) overlaps a tiled weight.
bool ggml_cuda_tiled_overlaps(const void * ptr, size_t size);

// Whether `tensor` is a tiled weight (exact tensor, not a view into one).
bool ggml_cuda_tiled_is_tiled(const ggml_tensor * tensor);

// Aborts if `node` reads tiled weight memory in a way only ggml_cuda_tiled_mul_mat supports.
void ggml_cuda_tiled_check_node(const ggml_tensor * node);

// If [offset, offset + size) of `tensor` is tiled weight memory, copies it to host memory `data` rebuilt in the
// native layout (synchronously on `stream`) and returns true.
bool ggml_cuda_tiled_read(const ggml_tensor * tensor, void * data, size_t offset, size_t size, cudaStream_t stream);

// Whether a batch is large enough for llama.cpp's own kernels (mul_mat_q reads tiled weights directly) rather than
// ggml_cuda_tiled_mul_mat. GGML_CUDA_TILED_MAX_COLS sets the largest batch for the tiled kernel (default 8).
bool ggml_cuda_tiled_large_batch(const ggml_tensor * src1);

// Whether a temporary native copy of `src0` is small enough, for kernels that can't read the tiled layout.
bool ggml_cuda_tiled_native_copy_ok(const ggml_tensor * src0);

// Bytes needed for a native copy of tiled `src0`, including the row padding native kernels may read.
size_t ggml_cuda_tiled_native_copy_size(const ggml_tensor * src0);

// Writes a native copy of tiled `src0` to device memory `dst` on `stream`.
void ggml_cuda_tiled_to_native(const ggml_tensor * src0, void * dst, cudaStream_t stream);

void ggml_cuda_tiled_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
