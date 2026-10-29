# CUDA tiled weights (Jetson Orin)

Branch `jetson-tiled-weights` (based on `v0.8.0`): an opt-in rearrangement of quantized weights for faster
generation on bandwidth-bound GPUs, developed and measured on a Jetson Orin Nano Super 8 GB.

## What it does

With `GGML_CUDA_TILED=1`:

- **Load time:** every Q8_0 / Q4_K / Q6_K weight matrix (2D, `ne0 % 256 == 0`, `ne1 % 16 == 0`, in a weights
  buffer) is rearranged in place on the GPU as soon as its upload completes: rows in tiles of 16, each 32-element
  block stored in the per-lane A-fragment order of `mma.sync m16n8k32` (s8). Same bytes as the native rows (Q4_K
  keeps its packed scales), so no extra memory.
- **Batches of up to 8 tokens** (plain decoding, DFlash verification, drafter head): int8 tensor-core kernels read
  the tiles directly (`ggml/src/ggml-cuda/tiled.cu`).
- **Larger batches** (prompt processing): llama.cpp's `mul_mat_q` kernels, instantiated a second time with a
  `tiled` template parameter whose tile loaders (`mmq-load-tiles.cuh`, `*_tiled`) read the tiled layout. Each thread
  copies whole 16-byte fragments (coalesced) into the same shared-memory tile the native loaders build, so the rest
  of `mul_mat_q` is unchanged. The stock instances are compiled exactly as before (same registers / stack per
  kernel), so stock performance is unaffected.
- Kernels other than `mul_mat_q` on a large batch (not used on Orin) get a temporary native copy of the weight (up to
  64 MB; larger weights stay on the tiled kernel).
- **Safety:** fusions that would hand a tiled weight to llama.cpp's kernels are disabled for those weights; reading a
  tiled weight back (e.g. graph copies) returns the native layout; any other op reading tiled memory aborts with a
  message; buffer free / clear / memset / copies keep the registry consistent.

Without the variable the binary behaves like stock llama.cpp.

Tuning knobs: `GGML_CUDA_TILED_MAX_COLS` (largest batch for the tiled kernel, default 8),
`GGML_CUDA_TILED_ANY_BUFFER=1` (tile tensors outside weight buffers too; for `test-backend-ops`).

## Build

Nothing to configure: build the CUDA backend as usual (e.g. `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`) and run
with `GGML_CUDA_TILED=1`.

## Validation

- `test-backend-ops -b CUDA0 -o MUL_MAT`, with `GGML_CUDA_TILED=1 GGML_CUDA_TILED_ANY_BUFFER=1` and without: 2490/2490 pass
  (90 cases take the tiled path; Nsight confirms the tiled kernels run).
- Perplexity (Bolt E4B Q4_K_M, llama.cpp docs text, 20 x 512 tokens): 17.9151 stock, 17.9151 tiled.
- Greedy server outputs, stock vs tiled: same answers; some outputs diverge in wording mid-text (rounding at near-ties,
  as between stock with and without DFlash).

## Results (2026-10-29, Orin Nano Super, MAXN_SUPER, L4T 36.4.7, Nemotron 3.5 Bolt E4B Q4_K_M)

Same binary, `GGML_CUDA_TILED` unset vs `=1`; "unpatched" is the stock v0.8.0 build.

`llama-bench` (`-ngl 99 -fa 1 -lm mmap -lzm on --repack 0`):

| test | unpatched | patched, unset | tiled | tiled vs unpatched |
|---|---|---|---|---|
| tg128 (generation) | 14.54 t/s | 14.54 t/s | 23.88 t/s | 1.64x |
| pp512, `-ub 256` | 564 t/s | 564 t/s | 621 t/s | 1.10x |
| pp2048, `-ub 256` | 566 t/s | 567 t/s | 624 t/s | 1.10x |
| pp512, `-ub 512` | 608 t/s | 607 t/s | 622 t/s | 1.02x |
| pp2048, `-ub 512` | 613 t/s | 614 t/s | 623 t/s | 1.02x |

`llama-server`, 32K context, `-ub 256` (production settings), 4 prompts x greedy and sampled for generation,
1.2K / 4.5K-token prompts x 3 (prompt cache off) for prompt processing:

| | stock | tiled | |
|---|---|---|---|
| plain decoding | 14.2-14.3 | 22.4-22.8 | 1.59x |
| plain, prompt processing | 535-555 | 578-584 | 1.05-1.09x |
| DFlash, short reasoning | 51.3-54.4 | 74.6-78.5 | 1.44x |
| DFlash, JSON extraction | 39.2-44.1 | 69.7-70.5 | 1.60-1.78x |
| DFlash, code | 33.1-37.3 | 50.9-54.4 | 1.46-1.54x |
| DFlash, free-form writing | 25.1-25.9 | 38.9-41.4 | 1.55-1.60x |
| DFlash, prompt processing | 473-499 | 517-562 | 1.09-1.13x |

Model load takes ~1 s longer (one-time rearrangement).

### History of the prompt-processing path

1. Native copy of each weight per prompt batch, then stock `mul_mat_q`: pp512 467 t/s (0.82x).
2. Tiled reads inside the stock tile loaders (`if (tiled)` per read): tiled 364 t/s, and stock dropped to 302 t/s.
3. Separate tiled loaders behind a runtime pointer tag: stock 520 t/s (the extra branch changed register allocation
   of every stock kernel, with spills), tiled 506 t/s.
4. `tiled` as a template parameter (stock kernels compiled as before): stock 564 t/s, tiled 506 t/s.
5. Coalesced 16-byte fragment copies in the tiled loaders (this patch): tiled 621 t/s.
