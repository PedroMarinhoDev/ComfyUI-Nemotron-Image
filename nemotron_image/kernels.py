"""CUDA kernels for a Nemotron-Labs-Image decode step, compiled at first use with NVRTC (no CUDA toolkit or compiler
needed). Built from the Drama 3 and HunyuanImage 3 packs' decode kernels for Nemotron's geometry.

A step advances R rows (images) by one token each. Everything a step needs stays on the GPU, so a whole step can be
captured in one CUDA graph and replayed with no host sync:

- `dense`: a bf16 GEMV over R rows whose prologue applies the layer's RMS norm (or Mamba's gated group norm) to its
  input and rounds it to bf16 as the reference does, and whose epilogue writes float32, adds into the bf16 residual
  stream, or (attention) writes the key and value rows into their caches.
- `route` (MoE): the router's logits, then in the grid's last block the sigmoid / bias / top-6 / normalise / x 2.5
  routing and a table of the distinct experts the rows use ("units"; units 0 and 1 are the two halves of the shared
  expert). Its blocks also write the normed input in the ConvRot basis for the experts.
- `experts_up` / `experts_down`: every unit's rows in one launch each, for bf16, int8 ConvRot, W6A8 and W4A8 banks as
  comfy_kitchen stores them. `up` squares the ReLU and rotates the hidden row in its epilogue (a 64-row tile holds
  whole rotation groups); `down`'s last block per output slice mixes the units by their routing weights and adds the
  result into the residual stream.
- `mamba_step`: the conv window (a ring of the last 3 inputs), the SSM state update in float32, the output and its
  gate; the gated group norm runs in the next `dense`'s prologue.
- `attend_split` / `attend_combine`: GQA attention of the step's query over the key/value caches (no positional
  encoding, no mask needed for a step).
- `head`: the final norm, the head over the vision rows, temperature and Gumbel-max sampling (or a forced token from
  a schedule), then in the last block the next token's embedding row into the residual stream and the step counter.
"""
import functools

import torch

HIDDEN = 2688
EXPERT = 1856
SHARED = 3712
EXPERTS = 128
TOP_K = 6
ROUTED_SCALE = 2.5
MAMBA_HEADS = 64
MAMBA_HEAD_DIM = 64
MAMBA_STATE = 128
MAMBA_GROUPS = 8
MAMBA_INNER = 4096
CONV_DIM = 6144
PROJ = 10304
Q_HEADS = 32
KV_HEADS = 2
HEAD_DIM = 128
VISION_BASE = 131075
VISION_ROWS = 196560

DENSE_ROWS_PER_WARP = 4
DOWN_ROWS_PER_WARP = 4
ATTEND_CHUNK = 64
FORMATS = {"bf16": 0, "int8": 1, "w4": 4, "w6": 6}

SOURCE = r"""
#define WARPS 8
#define GQ (G >= 4 ? G / 4 : 1)
#define NEG_INF __int_as_float(0xff800000)
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ float bf(u32 b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ float bf_lo(u32 w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(u32 w) { return __uint_as_float(w & 0xFFFF0000u); }
// float to bfloat16, to nearest and ties to even, as a cast does
__device__ __forceinline__ u16 to_bf16(float v) {
  u32 b = __float_as_uint(v);
  return (u16)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float rbf(float v) { return bf(to_bf16(v)); }
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// float8 e4m3 to float, subnormals included
__device__ __forceinline__ float e4m3(u32 b) {
  return __int_as_float(((b & 0x80u) << 24) | ((b & 0x7Fu) << 20)) * __int_as_float(0x7b800000);
}

// The block's per-row scales for the input transform: IN 1, rsqrt(mean(x^2) + eps) of each bf16 row x [R, K];
// IN 2, rsqrt(mean + eps) of each of the 8 groups of 512 of the gated Mamba output, from per-head sums of squares
// part [R, 64] (8 heads a group).
template <int MODE>
__device__ __forceinline__ void input_scales(const void* x, int K, const float* part, float eps, float* scale) {
  __shared__ float sums[WARPS][R];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (MODE == 1) {
    for (int r = 0; r < R; r++) {
      const u32* row = (const u32*)((const u16*)x + (long long)r * K);
      float s = 0.0f;
      for (int i = threadIdx.x; i < K / 2; i += blockDim.x) {
        u32 w = row[i];
        float a = bf_lo(w), b = bf_hi(w);
        s += a * a + b * b;
      }
      s = warp_sum(s);
      if (lane == 0) sums[warp][r] = s;
    }
    __syncthreads();
    if (threadIdx.x < R) {
      float s = 0.0f;
      for (int w = 0; w < WARPS; w++) s += sums[w][threadIdx.x];
      scale[threadIdx.x] = rsqrtf(s / (float)K + eps);
    }
  } else if (MODE == 2) {
    if (threadIdx.x < R * 8) {
      int r = threadIdx.x >> 3, grp = threadIdx.x & 7;
      float s = 0.0f;
      for (int h = 0; h < 8; h++) s += part[r * 64 + grp * 8 + h];
      scale[threadIdx.x] = rsqrtf(s / 512.0f + eps);
    }
  }
  __syncthreads();
}

// 8 input values of row r from column c on: IN 0 float32 x (rounded to bf16 with ROUND), IN 1 bf16 x * scale[r] *
// g, IN 2 float32 x * scale[r][c / 512] * g; IN 1 and 2 are rounded to bf16 as the model's norms round their output.
template <int MODE, int RND>
__device__ __forceinline__ void load_x8(const void* x, int K, int r, int c, const float* g, const float* scale,
                                        float* xv) {
  if (MODE == 1) {
    uint4 q = __ldg((const uint4*)((const u16*)x + (long long)r * K + c));
    float4 ga = __ldg((const float4*)(g + c)), gb = __ldg((const float4*)(g + c) + 1);
    float s = scale[r];
    xv[0] = rbf(ga.x * (bf_lo(q.x) * s)); xv[1] = rbf(ga.y * (bf_hi(q.x) * s));
    xv[2] = rbf(ga.z * (bf_lo(q.y) * s)); xv[3] = rbf(ga.w * (bf_hi(q.y) * s));
    xv[4] = rbf(gb.x * (bf_lo(q.z) * s)); xv[5] = rbf(gb.y * (bf_hi(q.z) * s));
    xv[6] = rbf(gb.z * (bf_lo(q.w) * s)); xv[7] = rbf(gb.w * (bf_hi(q.w) * s));
  } else {
    const float4* p = (const float4*)((const float*)x + (long long)r * K + c);
    float4 a = __ldg(p), b = __ldg(p + 1);
    xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w; xv[4] = b.x; xv[5] = b.y; xv[6] = b.z; xv[7] = b.w;
    if (MODE == 2) {
      float4 ga = __ldg((const float4*)(g + c)), gb = __ldg((const float4*)(g + c) + 1);
      float s = scale[r * 8 + c / 512];
      xv[0] = rbf(xv[0] * s * ga.x); xv[1] = rbf(xv[1] * s * ga.y); xv[2] = rbf(xv[2] * s * ga.z);
      xv[3] = rbf(xv[3] * s * ga.w); xv[4] = rbf(xv[4] * s * gb.x); xv[5] = rbf(xv[5] * s * gb.y);
      xv[6] = rbf(xv[6] * s * gb.z); xv[7] = rbf(xv[7] * s * gb.w);
    } else if (RND) {
      for (int i = 0; i < 8; i++) xv[i] = rbf(xv[i]);
    }
  }
}

__device__ __forceinline__ float dot8(uint4 w, const float* x) {
  return bf_lo(w.x) * x[0] + bf_hi(w.x) * x[1] + bf_lo(w.y) * x[2] + bf_hi(w.y) * x[3] + bf_lo(w.z) * x[4] +
         bf_hi(w.z) * x[5] + bf_lo(w.w) * x[6] + bf_hi(w.w) * x[7];
}

// acc[m][r] = bf16 weight row rows[m] . input row r (see load_x8), for the warp's NR rows.
template <int MODE, int RND, int NR>
__device__ __forceinline__ void dense_dot(const void* x, int K, const float* g, const float* scale,
                                          const u16* const* rows, float (*acc)[R]) {
  int lane = threadIdx.x & 31;
#pragma unroll 2
  for (int c = lane * 8; c < K; c += 256) {
    uint4 wv[NR];
#pragma unroll
    for (int m = 0; m < NR; m++) wv[m] = __ldcs((const uint4*)(rows[m] + c));
    if (R == 1) {
      float xv[8];
      load_x8<MODE, RND>(x, K, 0, c, g, scale, xv);
#pragma unroll
      for (int m = 0; m < NR; m++) acc[m][0] += dot8(wv[m], xv);
    } else {
      // several rows: each weight vector unpacked once, then met by every row
      float wf[NR][8];
#pragma unroll
      for (int m = 0; m < NR; m++) {
        u32 words[4] = {wv[m].x, wv[m].y, wv[m].z, wv[m].w};
#pragma unroll
        for (int j = 0; j < 4; j++) { wf[m][2 * j] = bf_lo(words[j]); wf[m][2 * j + 1] = bf_hi(words[j]); }
      }
#pragma unroll
      for (int r = 0; r < R; r++) {
        float xv[8];
        load_x8<MODE, RND>(x, K, r, c, g, scale, xv);
#pragma unroll
        for (int m = 0; m < NR; m++)
#pragma unroll
          for (int j = 0; j < 8; j++) acc[m][r] = fmaf(wf[m][j], xv[j], acc[m][r]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < NR; m++)
#pragma unroll
    for (int r = 0; r < R; r++) acc[m][r] = warp_sum(acc[m][r]);
}

// out = W x over R rows for a bf16 weight given as up to three row segments (w0 rows [0, n1), w1 [n1, n2), w2
// [n2, N)), with the input transform IN (see load_x8). OUT 0: out [R, N] float32. OUT 1: resid [R, N] (bf16) +=.
// OUT 2 (attention q/k/v): rows [0, n1) to out [R, n1] float32, rows [n1, n2) and [n2, N) are the key and value rows
// of 2 heads, written as bf16 to kc / vc [R, 2, cap, 128] at row pos[r]. Grid (ceil(N / (8 RW))), block 256.
extern "C" __global__ void __launch_bounds__(256) dense(const void* x, const float* g, const float* part, double eps,
                                                        int K, const u16* w0, const u16* w1, const u16* w2, int n1,
                                                        int n2, int N, float* out, u16* resid, u16* kc, u16* vc,
                                                        const int* pos, int cap) {
  __shared__ float scale[R * 8];
  input_scales<IN>(x, K, part, (float)eps, scale);
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int first = (blockIdx.x * WARPS + warp) * RW;
  const u16* rows[RW];
#pragma unroll
  for (int m = 0; m < RW; m++) {
    int n = min(first + m, N - 1);
    rows[m] = n < n1 ? w0 + (long long)n * K : n < n2 ? w1 + (long long)(n - n1) * K : w2 + (long long)(n - n2) * K;
  }
  float acc[RW][R];
#pragma unroll
  for (int m = 0; m < RW; m++)
#pragma unroll
    for (int r = 0; r < R; r++) acc[m][r] = 0.0f;
  dense_dot<IN, ROUND, RW>(x, K, g, scale, rows, acc);
  if (lane != 0) return;
#pragma unroll
  for (int m = 0; m < RW; m++) {
    int n = first + m;
    if (n >= N) break;
#pragma unroll
    for (int r = 0; r < R; r++) {
      float v = acc[m][r];
      if (OUT == 0) {
        out[(long long)r * N + n] = v;
      } else if (OUT == 1) {
        u16* p = resid + (long long)r * N + n;
        *p = to_bf16(bf(*p) + v);
      } else {
        if (n < n1) {
          out[(long long)r * n1 + n] = v;
        } else {
          int j = n < n2 ? n - n1 : n - n2;
          u16* cache = n < n2 ? kc : vc;
          cache[(((long long)r * 2 + j / 128) * cap + pos[r]) * 128 + (j & 127)] = to_bf16(v);
        }
      }
    }
  }
}

// A uniform 32-bit hash of (key, n) (splitmix64)
__device__ __forceinline__ u64 mix(u64 z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
// A standard Gumbel sample from 32 random bits, accurate in both tails.
__device__ __forceinline__ float gumbel(u32 h) {
  float e = h < 0x80000000u ? -logf(((float)h + 0.5f) * 2.3283064365386963e-10f)
                            : -log1pf(-((float)(0xFFFFFFFFu - h) + 0.5f) * 2.3283064365386963e-10f);
  return -logf(e);
}

// The step's last layer: the final norm, then the head's vision rows w [N, K] (bf16) against each row, and the next
// token: force[r * force_stride + pos[r] + 1] when it is >= 0, else vision_base + the sample of softmax(logits /
// temperature) by Gumbel-max (argmax with `greedy`; the noise is keyed by seed, row0 + r and the position). With
// `keep`, the logits go to logits [R, N] as well, at every step, forced or not. With `external` the step ends there:
// `accept` takes the token sampled elsewhere from the logits. Blocks write
// their best (value, index) per row to best_v / best_i [R, gridDim.x]; the last block picks each row's token, writes
// it to tokens[r * force_stride + pos[r] + 1], copies its embedding row (emb, [vocab, K] bf16, device or host memory)
// into resid [R, K] and advances pos[r]. With every row forced the head is not evaluated. Grid ceil(N / (8 RW)).
extern "C" __global__ void __launch_bounds__(256) head(const u16* x, const float* g, double eps, int K, const u16* w,
                                                       int N, double inv_temp, int greedy, int seed, int row0,
                                                       const int* force, int force_stride, int vision_base, int keep,
                                                       int external, float* logits, float* best_v, int* best_i,
                                                       int* counter, int* tokens, const u16* emb, u16* resid, int* pos) {
  __shared__ float scale[R * 8];
  __shared__ float top_v[WARPS][R];
  __shared__ int top_i[WARPS][R];
  __shared__ int last;
  __shared__ int all_forced;
  if (threadIdx.x == 0) {
    int forced = 1;
    for (int r = 0; r < R; r++) forced &= force[r * force_stride + pos[r] + 1] >= 0;
    all_forced = forced && !keep;
  }
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (!all_forced) {
    input_scales<1>(x, K, 0, (float)eps, scale);
    int first = (blockIdx.x * WARPS + warp) * RW;
    const u16* rows[RW];
#pragma unroll
    for (int m = 0; m < RW; m++) rows[m] = w + (long long)min(first + m, N - 1) * K;
    float acc[RW][R];
#pragma unroll
    for (int m = 0; m < RW; m++)
#pragma unroll
      for (int r = 0; r < R; r++) acc[m][r] = 0.0f;
    dense_dot<1, 1, RW>(x, K, g, scale, rows, acc);
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < R; r++) {
        u64 key = mix((u64)(u32)seed ^ mix(((u64)(row0 + r) << 32) | (u32)(pos[r] + 1)));
        float best = NEG_INF;
        int index = 0x7fffffff;
#pragma unroll
        for (int m = 0; m < RW; m++) {
          int n = first + m;
          if (n >= N) break;
          if (keep || external) logits[(long long)r * N + n] = acc[m][r];
          float v = greedy ? acc[m][r] : acc[m][r] * (float)inv_temp + gumbel((u32)mix(key + (u64)n));
          if (v > best) { best = v; index = n; }
        }
        top_v[warp][r] = best;
        top_i[warp][r] = index;
      }
    }
    __syncthreads();
    if (threadIdx.x < R) {
      int r = threadIdx.x;
      float best = NEG_INF;
      int index = 0x7fffffff;
      for (int i = 0; i < WARPS; i++)
        if (top_v[i][r] > best || (top_v[i][r] == best && top_i[i][r] < index)) { best = top_v[i][r]; index = top_i[i][r]; }
      best_v[r * gridDim.x + blockIdx.x] = best;
      best_i[r * gridDim.x + blockIdx.x] = index;
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(counter, 1) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  if (external) {
    if (threadIdx.x == 0) *counter = 0;
    return;
  }
  __threadfence();
  // the last block: each warp picks a row's token
  for (int r = warp; r < R; r += WARPS) {
    int t = pos[r] + 1, token = force[r * force_stride + t];
    if (token < 0) {
      float best = NEG_INF;
      int index = 0x7fffffff;
      for (int b = lane; b < gridDim.x; b += 32) {
        float v = __ldcg(best_v + r * gridDim.x + b);
        int i = __ldcg(best_i + r * gridDim.x + b);
        if (v > best || (v == best && i < index)) { best = v; index = i; }
      }
      for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, index, o);
        if (ov > best || (ov == best && oi < index)) { best = ov; index = oi; }
      }
      token = vision_base + index;
    }
    if (lane == 0) tokens[r * force_stride + t] = token;
    const uint4* src = (const uint4*)(emb + (long long)token * K);
    uint4* dst = (uint4*)(resid + (long long)r * K);
    for (int i = lane; i < K / 8; i += 32) dst[i] = src[i];
  }
  __syncthreads();
  if (threadIdx.x < R) pos[threadIdx.x] += 1;
  if (threadIdx.x == 0) *counter = 0;
}

// The next token of each row when it was sampled outside (top-k / top-p): force[r * force_stride + pos[r] + 1] when
// >= 0, else vision_base + sampled[r]; written to tokens, its embedding row (emb, device or host memory) into resid
// [R, K], pos[r] advanced. One block.
extern "C" __global__ void __launch_bounds__(256) accept(const long long* sampled, const int* force, int force_stride,
                                                         int vision_base, int* tokens, const u16* emb, u16* resid,
                                                         int K, int* pos) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int r = warp; r < R; r += WARPS) {
    int t = pos[r] + 1, token = force[r * force_stride + t];
    if (token < 0) token = vision_base + (int)sampled[r];
    if (lane == 0) tokens[r * force_stride + t] = token;
    const uint4* src = (const uint4*)(emb + (long long)token * K);
    uint4* dst = (uint4*)(resid + (long long)r * K);
    for (int i = lane; i < K / 8; i += 32) dst[i] = src[i];
  }
  __syncthreads();
  if (threadIdx.x < R) pos[threadIdx.x] += 1;
}

// out[i] = table[ids[i]] for rows of K bf16 values: an embedding lookup from a table in pinned host memory (or on the
// card). Grid (rows), block 128.
extern "C" __global__ void __launch_bounds__(128) gather_rows(const long long* ids, const u16* table, int K, u16* out) {
  const uint4* src = (const uint4*)(table + ids[blockIdx.x] * (long long)K);
  uint4* dst = (uint4*)(out + (long long)blockIdx.x * K);
  for (int i = threadIdx.x; i < K / 8; i += blockDim.x) dst[i] = src[i];
}

// In-place regular Hadamard (h4 kron ... kron h4) of size G on `n` floats of shared memory, in groups of G, by the
// threads of the block; v is scaled by 1 / sqrt(G). G = 1 leaves it unchanged.
__device__ __forceinline__ void hadamard_shared(float* v, int n) {
  for (int stride = 1; stride < G; stride *= 4) {
    for (int t = threadIdx.x; t < n / 4; t += blockDim.x) {
      int within = t % GQ, low = within % stride, high = within / stride;
      int i = (t / GQ) * G + high * stride * 4 + low;
      float a = v[i], b = v[i + stride], c = v[i + 2 * stride], d = v[i + 3 * stride];
      v[i] = a + b + c - d;
      v[i + stride] = a + b - c + d;
      v[i + 2 * stride] = a - b + c + d;
      v[i + 3 * stride] = -a + b + c + d;
    }
    __syncthreads();
  }
}

// 16 floats as int16 with one float scale (amax / 32767): the input the quantized experts' dp2a dot products read.
__device__ __forceinline__ void quantize16(const float* v, float factor, short* q, float* scale) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(v[i] * factor));
  float inv = amax > 0.0f ? 32767.0f / amax : 0.0f;
  u32 packed[8];
#pragma unroll
  for (int i = 0; i < 8; i++)
    packed[i] = ((u32)__float2int_rn(v[2 * i] * factor * inv) & 0xFFFFu) | ((u32)__float2int_rn(v[2 * i + 1] * factor * inv) << 16);
  ((uint4*)q)[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  ((uint4*)q)[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
  *scale = amax / 32767.0f;
}

// MoE routing for R rows of the residual stream x [R, K] (bf16) through the layer's norm (g, eps): logits [R, 128] =
// router w [128, K] (float32, as the file keeps it) . normed row (bf16-rounded, as float32); the blocks also write xr [R, K], the normed rows
// in the experts' ConvRot basis (rotation groups of G; G = 1 for bf16 experts), float32 and as int16 with a scale per
// 16 (xq, xsc). The last block routes each row
// (sigmoid scores, + bias for the choice only, top 6, weights normalised to sum 1 then x 2.5) into ids / weights
// [R, 6] and builds the unit table: unit 0 and 1 the shared expert's halves (every row, weight 1), then each distinct
// routed expert in order of first use: unit_id [U], unit_n [U] rows, unit_row / unit_w [U, R]; n_units[0] = U.
// Grid 16, block 256: a router row per warp.
extern "C" __global__ void __launch_bounds__(256) route(const u16* x, const float* g, double eps, int K, const float* w,
                                                        const float* bias, float* logits, float* xr, short* xq,
                                                        float* xsc, int* ids,
                                                        float* weights, int* n_units, int* unit_id, int* unit_n,
                                                        int* unit_row, float* unit_w, int* counter, int layer, int slots,
                                                        int* table, int* owner, int* stamps, int* clock, int* stats,
                                                        int* unit_slot, int* unit_fill) {
  __shared__ float scale[R * 8];
  __shared__ float chunk[WARPS][64];
  __shared__ int last;
  input_scales<1>(x, K, 0, (float)eps, scale);
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // the normed rows in the rotated basis: chunks of 64 (whole rotation groups), a chunk per warp at a time
  int chunks = K / 64;
  for (int c = blockIdx.x * WARPS + warp; c < R * chunks; c += gridDim.x * WARPS) {
    int r = c / chunks, k0 = (c % chunks) * 64;
    for (int i = lane; i < 64; i += 32)
      chunk[warp][i] = rbf(g[k0 + i] * (bf(x[(long long)r * K + k0 + i]) * scale[r]));
    __syncwarp();
    for (int stride = 1; stride < G; stride *= 4) {
      if (lane < 16) {
        int within = lane % GQ, low = within % stride, high = within / stride;
        int i = (lane / GQ) * G + high * stride * 4 + low;
        float a = chunk[warp][i], b = chunk[warp][i + stride], cc = chunk[warp][i + 2 * stride],
              d = chunk[warp][i + 3 * stride];
        chunk[warp][i] = a + b + cc - d;
        chunk[warp][i + stride] = a + b - cc + d;
        chunk[warp][i + 2 * stride] = a - b + cc + d;
        chunk[warp][i + 3 * stride] = -a + b + cc + d;
      }
      __syncwarp();
    }
    float norm = rsqrtf((float)G);
    for (int i = lane; i < 64; i += 32) xr[(long long)r * K + k0 + i] = chunk[warp][i] * norm;
    if (lane < 4) quantize16(chunk[warp] + 16 * lane, norm, xq + (long long)r * K + k0 + 16 * lane, xsc + (long long)r * (K / 16) + k0 / 16 + lane);
    __syncwarp();
  }
  int e = blockIdx.x * WARPS + warp;
  const float* row = w + (long long)e * K;
  float acc[R];
  for (int r = 0; r < R; r++) acc[r] = 0.0f;
  for (int c = lane * 8; c < K; c += 256) {
    float4 wa = __ldg((const float4*)(row + c)), wb = __ldg((const float4*)(row + c) + 1);
    for (int r = 0; r < R; r++) {
      float xv[8];
      load_x8<1, 1>(x, K, r, c, g, scale, xv);
      acc[r] += wa.x * xv[0] + wa.y * xv[1] + wa.z * xv[2] + wa.w * xv[3] + wb.x * xv[4] + wb.y * xv[5] + wb.z * xv[6] +
                wb.w * xv[7];
    }
  }
  for (int r = 0; r < R; r++) acc[r] = warp_sum(acc[r]);
  if (lane == 0)
    for (int r = 0; r < R; r++) logits[r * 128 + e] = acc[r];
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(counter, 1) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  __shared__ float top_w[R][6];
  __shared__ int top_e[R][6];
  for (int r = warp; r < R; r += WARPS) {
    float score[4], choice[4];
    for (int j = 0; j < 4; j++) {
      float l = __ldcg(logits + r * 128 + lane + 32 * j);
      score[j] = 1.0f / (1.0f + expf(-l));
      choice[j] = score[j] + bias[lane + 32 * j];
    }
    float total = 0.0f;
    for (int k = 0; k < 6; k++) {
      float best = NEG_INF;
      int index = 128;
      for (int j = 0; j < 4; j++)
        if (choice[j] > best) { best = choice[j]; index = lane + 32 * j; }
      for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, index, o);
        if (ov > best || (ov == best && oi < index)) { best = ov; index = oi; }
      }
      float s = __shfl_sync(0xffffffffu, score[index >> 5], index & 31);
      if ((index & 31) == lane) choice[index >> 5] = NEG_INF;
      total += s;
      if (lane == 0) { top_e[r][k] = index; top_w[r][k] = s; }
    }
    __syncwarp();
    if (lane < 6) {
      float wgt = top_w[r][lane] / (total + 1e-20f) * 2.5f;
      top_w[r][lane] = wgt;
      ids[r * 6 + lane] = top_e[r][lane];
      weights[r * 6 + lane] = wgt;
    }
  }
  __syncthreads();
  // the unit table: units 0 and 1 the shared expert's halves, then the experts the rows chose in ascending order; a
  // row's slot in a unit is the number of earlier rows that chose the same expert
  __shared__ int used[128], start[128], warp_total[4];
  int t = threadIdx.x;
  if (t < 128) {
    int count = 0;
    for (int r = 0; r < R; r++)
      for (int k = 0; k < 6; k++) count += top_e[r][k] == t;
    used[t] = count;
    int flag = count > 0, upto = flag;
    for (int o = 1; o < 32; o <<= 1) {
      int other = __shfl_up_sync(0xffffffffu, upto, o);
      if (lane >= o) upto += other;
    }
    start[t] = upto - flag;
    if (lane == 31) warp_total[warp] = upto;
  }
  __syncthreads();
  if (t < 128) {
    for (int w = 0; w < warp; w++) start[t] += warp_total[w];
    if (used[t]) {
      unit_id[2 + start[t]] = t;
      unit_n[2 + start[t]] = used[t];
    }
  }
  __syncthreads();
  if (t < R * 6) {
    int r = t / 6, e = top_e[r][t % 6], u = 2 + start[e], slot = 0;
    for (int o = 0; o < r; o++)
      for (int k = 0; k < 6; k++) slot += top_e[o][k] == e;
    unit_row[u * R + slot] = r;
    unit_w[u * R + slot] = top_w[r][t % 6];
  }
  if (t < 2 * R) {
    unit_row[t] = t % R;
    unit_w[t] = 1.0f;
  }
  if (t < 2) {
    unit_id[t] = 128 + t;
    unit_n[t] = R;
  }
  if (t == 0) {
    n_units[0] = 2 + warp_total[0] + warp_total[1] + warp_total[2] + warp_total[3];
    *counter = 0;
  }
  // Where each routed unit's expert is read: a resident bank (slots 0) at the expert's own index; else the expert
  // cache's slot. A hit stamps its slot with the clock (once per MoE layer per step); a miss takes the least recently
  // stamped slot that this layer has not stamped (one LRU over every layer), evicts its owner, and is marked fill:
  // the expert kernels read it from host memory and copy it into the slot on the way.
  __shared__ int now, missed[128], best_v[WARPS], best_s[WARPS];
  if (slots == 0) {
    if (t < 128 && used[t]) {
      unit_slot[2 + start[t]] = t;
      unit_fill[2 + start[t]] = 0;
    }
    return;
  }
  if (t == 0) now = ++clock[0];
  __syncthreads();
  if (t < 128) {
    missed[t] = 0;
    if (used[t]) {
      int u = 2 + start[t], slot = table[layer * 128 + t];
      if (slot >= 0) {
        stamps[slot] = now;
        unit_slot[u] = slot;
        unit_fill[u] = 0;
      } else {
        missed[t] = 1;
      }
    }
  }
  __syncthreads();
  int misses = 0;
  for (int e = 0; e < 128; e++) {
    if (!missed[e]) continue;
    misses++;
    int v = 0x7fffffff, at = 0x7fffffff;
    for (int i = t; i < slots; i += blockDim.x) {
      int stamp = stamps[i];
      if (stamp != now && (stamp < v || (stamp == v && i < at))) { v = stamp; at = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
      int ov = __shfl_xor_sync(0xffffffffu, v, o), oa = __shfl_xor_sync(0xffffffffu, at, o);
      if (ov < v || (ov == v && oa < at)) { v = ov; at = oa; }
    }
    if (lane == 0) { best_v[warp] = v; best_s[warp] = at; }
    __syncthreads();
    if (t == 0) {
      for (int w = 1; w < WARPS; w++)
        if (best_v[w] < v || (best_v[w] == v && best_s[w] < at)) { v = best_v[w]; at = best_s[w]; }
      int key = layer * 128 + e, old = owner[at];
      if (old >= 0) table[old] = -1;
      owner[at] = key;
      table[key] = at;
      stamps[at] = now;
      unit_slot[2 + start[e]] = at;
      unit_fill[2 + start[e]] = 1;
    }
    __syncthreads();
  }
  if (t == 0) {
    stats[0] += warp_total[0] + warp_total[1] + warp_total[2] + warp_total[3] - misses;
    stats[1] += misses;
  }
}

// An expert weight row's 16 columns from column c on, as FMT stores them: bf16, or on the int8 grid: int8 ConvRot as
// stored; W4A8 / W6A8 codes (a nibble plane, at 6 bits plus a plane of each code's top 2 bits after the K/2 nibble
// bytes) through the 16-entry codebook (4 bits) or code - 32 (6 bits), times the group's e4m3 scale (16 columns),
// rounded (ties to even, after a float32 product, as torch rounds) and clamped to +-127 as comfy_kitchen decodes
// them. `row` is the row's start, `rel` its group scales, `width` its full width; `peak` the largest |value| before
// the scale, so a group whose scale cannot reach 127.5 skips the clamp. All in full-rate float arithmetic: integers
// pass through the low bits of 1.5 * 2^23 or 2^23 instead of int/float conversions.
#define MAGIC 12582912.0f
#if FMT == 0
struct WV { uint4 a, b; };
#else
typedef int4 WV;
#endif

__device__ __forceinline__ void unpack16(WV q, float* w) {
#if FMT == 0
  u32 words[8] = {q.a.x, q.a.y, q.a.z, q.a.w, q.b.x, q.b.y, q.b.z, q.b.w};
#pragma unroll
  for (int j = 0; j < 8; j++) { w[2 * j] = bf_lo(words[j]); w[2 * j + 1] = bf_hi(words[j]); }
#else
  // a signed byte b as float: 2^23 + (b + 128) - (2^23 + 128)
  u32 words[4] = {(u32)q.x ^ 0x80808080u, (u32)q.y ^ 0x80808080u, (u32)q.z ^ 0x80808080u, (u32)q.w ^ 0x80808080u};
#pragma unroll
  for (int j = 0; j < 16; j++) w[j] = __uint_as_float(__byte_perm(words[j >> 2], 0x4B000000u, 0x7650u + (j & 3))) - 8388736.0f;
#endif
}

__device__ __forceinline__ WV load_raw(const unsigned char* row, int c) {
#if FMT == 0
  const uint4* p = (const uint4*)(row + 2 * c);
  WV v;
  v.a = __ldcs(p);
  v.b = __ldcs(p + 1);
  return v;
#else
  return __ldcs((const int4*)(row + c));
#endif
}

// A vector's raw bytes as the bank stores them, fetched ahead of decoding so a warp's loads for all its rows are in
// flight together.
#if FMT <= 1
typedef WV Raw;
#else
struct Raw { uint2 low; u32 high; u32 scale; };
#endif

__device__ __forceinline__ Raw fetch(const unsigned char* row, const unsigned char* rel, int width, int c) {
#if FMT <= 1
  return load_raw(row, c);
#else
  Raw r;
  r.low = __ldcs((const uint2*)(row + c / 2));
  r.high = FMT == 6 ? __ldcs((const u32*)(row + width / 2 + c / 4)) : 0u;
  r.scale = __ldcs(rel + c / 16);
  return r;
#endif
}

// Packed half2 arithmetic (fp16 pairs in a u32), rounded to nearest and never fused.
__device__ __forceinline__ u32 hsub2(u32 a, u32 b) { u32 r; asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 hmul2(u32 a, u32 b) { u32 r; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 hadd2(u32 a, u32 b) { u32 r; asm("add.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 hclamp127(u32 a) {
  u32 r;
  asm("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(0xD7F0D7F0u));  // -127
  asm("min.f16x2 %0, %1, %2;" : "=r"(r) : "r"(r), "r"(0x57F057F0u));  // 127
  return r;
}
__device__ __forceinline__ u32 half2_of(float v) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
  return (u32)h * 0x10001u;
}

// 4 codes (the low 16 bits of c4, a nibble each) through a 16-entry int8 table t: byte_perm picks each code's entry
// among bytes 0-7 and 8-15 by its low 3 bits, then between the two by its top bit.
__device__ __forceinline__ u32 lut4(u32 c4, uint4 t) {
  u32 idx = c4 & 0x7777u;
  u32 lo = __byte_perm(t.x, t.y, idx), hi = __byte_perm(t.z, t.w, idx);
  return __byte_perm(lo, hi, ((c4 >> 1) & 0x4444u) | 0x3210u);
}

// The 16 weights as int8 bytes on the grid. W4A8: through the block's table of every e4m3 scale's 16 rounded codebook
// values; W6A8: (code - 32) * scale in float, rounded (ties to even, after a float32 product, as torch rounds) and
// clamped to +-127, integers passing through the low bits of 1.5 * 2^23 instead of int/float conversions.
__device__ __forceinline__ int4 to_int8(Raw raw, const uint4* table, float peak) {
#if FMT == 0
  return make_int4(0, 0, 0, 0);
#elif FMT == 1
  return raw;
#elif FMT == 4
  uint4 t = table[raw.scale];
  return make_int4((int)lut4(raw.low.x, t), (int)lut4(raw.low.x >> 16, t), (int)lut4(raw.low.y, t),
                   (int)lut4(raw.low.y >> 16, t));
#else
  // W6A8, two codes at a time in half precision: (code - 32) * s is exact in fp16 (6 by 4 significant bits), and
  // 1536 + v rounds it to an integer (ties to even) whose two's complement is the low byte of the half's bits
  float s = e4m3(raw.scale);
  u32 s2 = half2_of(s), clamp = s * peak >= 127.0f;
  u32 f[8];
#pragma unroll
  for (int b = 0; b < 8; b++) {
    u32 byte = __byte_perm(b < 4 ? raw.low.x : raw.low.y, 0u, 0x4440u | (b & 3));
    u32 nibbles = (byte * 0x1001u) & 0x000F000Fu;                   // codes 2b and 2b+1, low 4 bits, at bits 0 and 16
    u32 high = (((raw.high >> (4 * b)) & 0xFu) * 0x40010u) & 0x00300030u;  // their top 2 bits at bits 4 and 20
    u32 v = hmul2(hsub2(nibbles | high | 0x64006400u, 0x64206420u), s2);   // (1024 + code) - 1056 = code - 32
    if (clamp) v = hclamp127(v);
    f[b] = hadd2(v, 0x66006600u);                                    // + 1536
  }
  return make_int4((int)__byte_perm(f[0], f[1], 0x6420u), (int)__byte_perm(f[2], f[3], 0x6420u),
                   (int)__byte_perm(f[4], f[5], 0x6420u), (int)__byte_perm(f[6], f[7], 0x6420u));
#endif
}

// W4A8: table[s] = the 16 codebook values times e4m3 scale s on the int8 grid, as comfy_kitchen rounds and clamps them.
__device__ __forceinline__ void build_table(uint4* table, const float* lut) {
  for (int t = threadIdx.x; t < 256; t += blockDim.x) {
    float s = e4m3(t);
    u32 words[4] = {0u, 0u, 0u, 0u};
    for (int c = 0; c < 16; c++) {
      float f = fminf(fmaxf(__fadd_rn(__fmul_rn(lut[c], s), MAGIC), MAGIC - 127.0f), MAGIC + 127.0f);
      words[c >> 2] |= (__float_as_uint(f) & 0xFFu) << (8 * (c & 3));
    }
    table[t] = make_uint4(words[0], words[1], words[2], words[3]);
  }
}

__device__ __forceinline__ void load_x16(const float* x, float* xv) {
  const float4* p = (const float4*)x;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float4 v = __ldg(p + i);
    xv[4 * i] = v.x; xv[4 * i + 1] = v.y; xv[4 * i + 2] = v.z; xv[4 * i + 3] = v.w;
  }
}

__device__ __forceinline__ float dot16(const float* w, const float* x) {
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; j++) acc = fmaf(w[j], x[j], acc);
  return acc;
}

// 16 int8 weights . 16 int16 inputs (xa, xb), exactly, in 8 dp2a.
__device__ __forceinline__ int dot16_int(int4 w, int4 xa, int4 xb) {
  int a = __dp2a_lo(xa.x, w.x, 0);
  a = __dp2a_hi(xa.y, w.x, a);
  a = __dp2a_lo(xa.z, w.y, a);
  a = __dp2a_hi(xa.w, w.y, a);
  a = __dp2a_lo(xb.x, w.z, a);
  a = __dp2a_hi(xb.y, w.z, a);
  a = __dp2a_lo(xb.z, w.w, a);
  return __dp2a_hi(xb.w, w.w, a);
}

// acc[m][s] = the warp's NR weight rows . input row s (s < count) over K columns from the rows' column col. bf16
// weights meet float32 inputs xf[s]; quantized weights, decoded to int8 once per vector, meet int16 inputs xq[s] with
// a float scale per 16 (xsc[s]) in exact integer dot products.
template <int NR>
__device__ __forceinline__ void expert_dot(const unsigned char* const* rows, const unsigned char* const* rels, int width,
                                           int col, int K, const uint4* table, float peak, const float* const* xf,
                                           const short* const* xq, const float* const* xsc, int count, float (*acc)[R]) {
  int lane = threadIdx.x & 31;
  for (int c = lane * 16; c < K; c += 512) {
    Raw raw[NR];
#pragma unroll
    for (int m = 0; m < NR; m++) raw[m] = fetch(rows[m], rels[m], width, col + c);
#if FMT == 0
#pragma unroll
    for (int s = 0; s < R; s++) {
      if (s >= count) break;
      float xv[16];
      load_x16(xf[s] + c, xv);
#pragma unroll
      for (int m = 0; m < NR; m++) {
        float w[16];
        unpack16(raw[m], w);
        acc[m][s] += dot16(w, xv);
      }
    }
#else
    int4 wv[NR];
#pragma unroll
    for (int m = 0; m < NR; m++) wv[m] = to_int8(raw[m], table, peak);
#pragma unroll
    for (int s = 0; s < R; s++) {
      if (s >= count) break;
      const int4* xp = (const int4*)(xq[s] + c);
      int4 xa = __ldg(xp), xb = __ldg(xp + 1);
      float sc = __ldg(xsc[s] + c / 16);
#pragma unroll
      for (int m = 0; m < NR; m++) acc[m][s] += (float)dot16_int(wv[m], xa, xb) * sc;
    }
#endif
  }
}

// A bank's fields: w the rows, rel the e4m3 group scales [rows, width / 16] (W4A8 / W6A8), s the row scales [rows]
// (int8: scale; W4A8 / W6A8: s_channel), lut the 4-bit codebook.
#define BANK_ARGS const unsigned char *rw, const unsigned char *rrel, const float *rs, const float *rlut, \
  const unsigned char *sw, const unsigned char *srel, const float *ss, const float *slut, const unsigned char *hw, \
  const unsigned char *hrel, const float *hs, const int *unit_slot, const int *unit_fill
struct Bank {
  const unsigned char* w;
  const unsigned char* rel;
  const float* s;
  const float* lut;
};

__device__ __forceinline__ long long row_bytes(int width) {
  return FMT == 0 ? 2LL * width : FMT == 1 ? (long long)width : (long long)width * FMT / 8;
}

// Weight row `n` of a unit's matrix: a routed expert (`id` < 128) is rows [slot * N, (slot + 1) * N) of the routed
// bank [* N, width] (the resident bank, where slot is the expert's index, or the expert cache's slots); the shared
// expert's half h = id - 128 is, for up, rows [h * N, (h + 1) * N) of its [2N, width] and, for down, columns
// [h * K, (h + 1) * K) of its [N, 2K]. Gives the row's start, its group scales' start, its scale, its full width and
// the unit's first column in it.
__device__ __forceinline__ void unit_row(int id, int slot, int n, int N, int K, int down, const Bank& routed,
                                         const Bank& shared, const unsigned char** row, const unsigned char** rel,
                                         float* s, int* width, int* col) {
  const Bank& bank = id < 128 ? routed : shared;
  int h = id - 128;
  long long r = id < 128 ? (long long)slot * N + n : down ? n : (long long)h * N + n;
  *width = id < 128 || !down ? K : 2 * K;
  *col = id < 128 || !down ? 0 : h * K;
  *row = bank.w + r * row_bytes(*width);
  *rel = bank.rel + r * (*width / 16);
  *s = FMT == 0 ? 1.0f : bank.s[r];
}

// A cache miss: the warp copies routed expert `id`'s rows n0.. (NR of them, `width` wide) from the host bank into the
// slot rows the unit reads, and returns their scales.
template <int NR>
__device__ __forceinline__ void fill_rows(int id, int n0, int N, int width, const Bank& host, const unsigned char* const* rows,
                                          const unsigned char* const* rels, const float* slot_s, int slot, float* scales) {
  int lane = threadIdx.x & 31;
  long long bytes = row_bytes(width);
  for (int m = 0; m < NR; m++) {
    long long r = (long long)id * N + n0 + m;
    const u32* src = (const u32*)(host.w + r * bytes);
    u32* dst = (u32*)rows[m];
    for (int i = lane; i < bytes / 4; i += 32) dst[i] = src[i];
    if (FMT >= 4) {
      const u32* rs = (const u32*)(host.rel + r * (width / 16));
      u32* rd = (u32*)rels[m];
      for (int i = lane; i < width / 64; i += 32) rd[i] = rs[i];
    }
    float sc = FMT == 0 ? 1.0f : host.s[r];
    if (FMT != 0 && lane == 0) ((float*)slot_s)[(long long)slot * N + n0 + m] = sc;
    scales[m] = sc;
  }
  __threadfence_block();
  __syncwarp();
}

// The block's decode state for a unit of bank FMT: W4A8's per-scale table of its codebook (routed or shared bank), and
// the largest |value| before the scale (W6A8: 32), past which a group's scale may need the clamp.
__device__ __forceinline__ float prepare_bank(int id, const Bank& routed, const Bank& shared, uint4* table) {
  if (FMT == 4) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = (id < 128 ? routed.lut : shared.lut)[threadIdx.x];
    __syncthreads();
    build_table(table, lut);
    __syncthreads();
  }
  return 32.0f;
}

// Tile of 64 rows of the up projection of units u0 + blockIdx.y against its input rows (the normed rows in the ConvRot basis: xr
// float32 for bf16 banks, xq / xsc int16 for quantized ones), squared ReLU, then (quantized formats) the hidden row
// rotated into the down projection's ConvRot basis and stored as int16 with a scale per 16: hidq [U, R, N], hids
// [U, R, N / 16] (bf16: hid [U, R, N] float32). Grid (N / 64, units), block 256: 8 rows a warp. A shared expert stored
// in another format than the routed banks runs as its own launch (units 0 and 1, u0 = 0) with that format.
extern "C" __global__ void __launch_bounds__(256) experts_up(const float* xr, const short* xq, const float* xsc, int K,
                                                             int N, const int* n_units, const int* unit_id,
                                                             const int* unit_n, const int* unit_row_of, BANK_ARGS,
                                                             float* hid, short* hidq, float* hids, int u0) {
  Bank routed = {rw, rrel, rs, rlut}, shared = {sw, srel, ss, slut}, host = {hw, hrel, hs, rlut};
  __shared__ float tile[R][64];
  __shared__ uint4 table[FMT == 4 ? 256 : 1];
  int u = u0 + blockIdx.y;
  if (u >= n_units[0]) return;
  int id = unit_id[u], count = unit_n[u];
  float peak = prepare_bank(id, routed, shared, table);
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int first = blockIdx.x * 64 + warp * 8;
  const unsigned char* rows[8];
  const unsigned char* rels[8];
  float scales[8];
  int width = K, col = 0;
  int slot = id < 128 ? unit_slot[u] : 0;
#pragma unroll
  for (int m = 0; m < 8; m++) unit_row(id, slot, first + m, N, K, 0, routed, shared, &rows[m], &rels[m], &scales[m], &width, &col);
  if (id < 128 && unit_fill[u]) fill_rows<8>(id, first, N, width, host, rows, rels, rs, slot, scales);
  const float* xf[R];
  const short* xqs[R];
  const float* xss[R];
#pragma unroll
  for (int s = 0; s < R; s++) {
    long long r = s < count ? unit_row_of[u * R + s] : 0;
    xf[s] = xr + r * K;
    xqs[s] = xq + r * K;
    xss[s] = xsc + r * (K / 16);
  }
  float acc[8][R];
#pragma unroll
  for (int m = 0; m < 8; m++)
#pragma unroll
    for (int s = 0; s < R; s++) acc[m][s] = 0.0f;
  expert_dot<8>(rows, rels, width, col, K, table, peak, xf, xqs, xss, count, acc);
#pragma unroll
  for (int m = 0; m < 8; m++)
#pragma unroll
    for (int s = 0; s < R; s++) {
      float v = warp_sum(acc[m][s]) * scales[m];
      v = fmaxf(v, 0.0f);
      if (lane == 0) tile[s][warp * 8 + m] = v * v;
    }
  __syncthreads();
  for (int s = 0; s < count; s++) hadamard_shared(tile[s], 64);
  float norm = rsqrtf((float)G);
  if (FMT == 0) {
    for (int i = threadIdx.x; i < count * 64; i += blockDim.x)
      hid[((long long)u * R + i / 64) * N + blockIdx.x * 64 + (i & 63)] = tile[i / 64][i & 63];
  } else if (threadIdx.x < count * 4) {
    int s = threadIdx.x / 4, part = threadIdx.x % 4;
    long long at = ((long long)u * R + s) * N + blockIdx.x * 64 + 16 * part;
    quantize16(tile[s] + 16 * part, norm, hidq + at, hids + at / 16);
  }
}

// The down projection of units u0 + blockIdx.y, of their hidden rows (hid / hidq, hids as experts_up writes them, K
// wide) into part [U, R, N]. With `combine`, the last block of each slice of 8 DRW output rows then adds, for each row
// r, the sum over all its units of weight x part into resid [R, N] (bf16), in unit order; a shared expert in its own
// format runs first as a launch without `combine`. Grid (N / (8 DRW), units), block 256.
extern "C" __global__ void __launch_bounds__(256) experts_down(const float* hid, const short* hidq, const float* hids,
                                                               int K, int N, const int* n_units, const int* unit_id,
                                                               const int* unit_n, const int* unit_row_of,
                                                               const float* unit_w, BANK_ARGS, float* part,
                                                               int* counter, u16* resid, int u0, int combine) {
  Bank routed = {rw, rrel, rs, rlut}, shared = {sw, srel, ss, slut}, host = {hw, hrel, hs, rlut};
  __shared__ uint4 table[FMT == 4 ? 256 : 1];
  __shared__ int last;
  int u = u0 + blockIdx.y, units = n_units[0];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (u < units) {
    int id = unit_id[u], count = unit_n[u];
    float peak = prepare_bank(id, routed, shared, table);
    int first = (blockIdx.x * WARPS + warp) * DRW;
    const unsigned char* rows[DRW];
    const unsigned char* rels[DRW];
    float scales[DRW];
    int width = K, col = 0;
    int slot = id < 128 ? unit_slot[u] : 0;
#pragma unroll
    for (int m = 0; m < DRW; m++) unit_row(id, slot, first + m, N, K, 1, routed, shared, &rows[m], &rels[m], &scales[m], &width, &col);
    if (id < 128 && unit_fill[u]) fill_rows<DRW>(id, first, N, width, host, rows, rels, rs, slot, scales);
    const float* xf[R];
    const short* xqs[R];
    const float* xss[R];
#pragma unroll
    for (int s = 0; s < R; s++) {
      long long at = ((long long)u * R + s) * K;
      xf[s] = hid + at;
      xqs[s] = hidq + at;
      xss[s] = hids + at / 16;
    }
    float acc[DRW][R];
#pragma unroll
    for (int m = 0; m < DRW; m++)
#pragma unroll
      for (int s = 0; s < R; s++) acc[m][s] = 0.0f;
    expert_dot<DRW>(rows, rels, width, col, K, table, peak, xf, xqs, xss, count, acc);
#pragma unroll
    for (int m = 0; m < DRW; m++)
#pragma unroll
      for (int s = 0; s < R; s++) {
        float v = warp_sum(acc[m][s]) * scales[m];
        if (lane == 0 && s < count) part[((long long)u * R + s) * N + first + m] = v;
      }
  }
  if (!combine) return;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(counter + blockIdx.x, 1) == gridDim.y - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  int rows_per_slice = WARPS * DRW;
  for (int t = threadIdx.x; t < R * rows_per_slice; t += blockDim.x) {
    int r = t / rows_per_slice, n = blockIdx.x * rows_per_slice + t % rows_per_slice;
    float acc = 0.0f;
    for (int v = 0; v < units; v++) {
      int count = unit_n[v];
      for (int s = 0; s < count; s++)
        if (unit_row_of[v * R + s] == r) acc += unit_w[v * R + s] * __ldcg(part + ((long long)v * R + s) * N + n);
    }
    u16* p = resid + (long long)r * N + n;
    *p = to_bf16(bf(*p) + acc);
  }
  if (threadIdx.x == 0) counter[blockIdx.x] = 0;
}

// The input of an int8 ConvRot projection, once per layer: row r of x through the transform IN (see load_x8: the
// layer's RMS norm, Mamba's gated group norm, or a plain bf16 rounding), rotated by the regular Hadamard of G per group
// into the weight's basis, stored as int16 with a scale per 16 (xq [R, K], xsc [R, K / 16]). Grid (R), block 256;
// K <= 4096.
extern "C" __global__ void __launch_bounds__(256) prep_rows(const void* x, const float* g, const float* part, double eps,
                                                            int K, short* xq, float* xsc) {
  __shared__ float scale[R * 8];
  __shared__ float row[4096];
  input_scales<IN>(x, K, part, (float)eps, scale);
  int r = blockIdx.x;
  for (int c = threadIdx.x * 8; c < K; c += blockDim.x * 8) {
    float v[8];
    load_x8<IN, 1>(x, K, r, c, g, scale, v);
    for (int i = 0; i < 8; i++) row[c + i] = v[i];
  }
  __syncthreads();
  hadamard_shared(row, K);
  float norm = rsqrtf((float)G);
  for (int c = threadIdx.x * 16; c < K; c += blockDim.x * 16)
    quantize16(row + c, norm, xq + (long long)r * K + c, xsc + (long long)r * (K / 16) + c / 16);
}

// dense for int8 ConvRot weights: up to three row segments (w0 rows [0, n1) with row scales s0, w1 [n1, n2) / s1,
// w2 [n2, N) / s2), against prep_rows' xq / xsc, exact integer dot products (dp2a). OUT as dense. Grid
// (ceil(N / (8 RW))), block 256.
extern "C" __global__ void __launch_bounds__(256) dense_q8(const short* xq, const float* xsc, int K, const signed char* w0,
                                                           const float* s0, const signed char* w1, const float* s1,
                                                           const signed char* w2, const float* s2, int n1, int n2, int N,
                                                           float* out, u16* resid, u16* kc, u16* vc, const int* pos,
                                                           int cap) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int first = (blockIdx.x * WARPS + warp) * RW;
  const signed char* rows[RW];
  float scales[RW];
#pragma unroll
  for (int m = 0; m < RW; m++) {
    int n = min(first + m, N - 1);
    rows[m] = n < n1 ? w0 + (long long)n * K : n < n2 ? w1 + (long long)(n - n1) * K : w2 + (long long)(n - n2) * K;
    scales[m] = n < n1 ? s0[n] : n < n2 ? s1[n - n1] : s2[n - n2];
  }
  float acc[RW][R];
#pragma unroll
  for (int m = 0; m < RW; m++)
#pragma unroll
    for (int r = 0; r < R; r++) acc[m][r] = 0.0f;
  for (int c = lane * 16; c < K; c += 512) {
    int4 wv[RW];
#pragma unroll
    for (int m = 0; m < RW; m++) wv[m] = __ldcs((const int4*)(rows[m] + c));
#pragma unroll
    for (int r = 0; r < R; r++) {
      const int4* xp = (const int4*)(xq + (long long)r * K + c);
      int4 xa = __ldg(xp), xb = __ldg(xp + 1);
      float sc = __ldg(xsc + (long long)r * (K / 16) + c / 16);
#pragma unroll
      for (int m = 0; m < RW; m++) acc[m][r] += (float)dot16_int(wv[m], xa, xb) * sc;
    }
  }
#pragma unroll
  for (int m = 0; m < RW; m++)
#pragma unroll
    for (int r = 0; r < R; r++) acc[m][r] = warp_sum(acc[m][r]) * scales[m];
  if (lane != 0) return;
#pragma unroll
  for (int m = 0; m < RW; m++) {
    int n = first + m;
    if (n >= N) break;
#pragma unroll
    for (int r = 0; r < R; r++) {
      float v = acc[m][r];
      if (OUT == 0) {
        out[(long long)r * N + n] = v;
      } else if (OUT == 1) {
        u16* p = resid + (long long)r * N + n;
        *p = to_bf16(bf(*p) + v);
      } else {
        if (n < n1) {
          out[(long long)r * n1 + n] = v;
        } else {
          int j = n < n2 ? n - n1 : n - n2;
          u16* cache = n < n2 ? kc : vc;
          cache[(((long long)r * 2 + j / 128) * cap + pos[r]) * 128 + (j & 127)] = to_bf16(v);
        }
      }
    }
  }
}

__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.0f ? v : log1pf(expf(v)); }

// A Mamba2 step for head h of row r. proj [R, 10304] (float32) is in_proj's output: gate z [4096] | conv input
// [6144] (x 4096, B 1024, C 1024) | dt [64]. ring [R, 4, 6144] holds the last conv inputs (slot t & 3 for step t =
// pos[r]); conv_w [6144, 4], conv_b [6144]; the SSM state [R, 64, 64, 128] float32; A = -exp(A_log), dt_bias, D [64].
// Writes v [R, 4096] = y * SiLU(z) and part [R, 64], each head's sum of v^2 (the gated norm's statistics). The conv
// input of step t is written to its ring slot by the head owning the channel (B and C: a group's first head).
// Grid (64, R), block 256.
extern "C" __global__ void __launch_bounds__(256) mamba_step(const float* proj, float* ring, const float* conv_w,
                                                             const float* conv_b, float* state, const float* A,
                                                             const float* dt_bias, const float* D, const int* pos,
                                                             float* v, float* part) {
  __shared__ float xs[64], Bs[128], Cs[128], ys[64], sums[2];
  int h = blockIdx.x, r = blockIdx.y, grp = h / 8;
  int t = pos[r];
  const float* in = proj + (long long)r * 10304 + 4096;
  float* ring_r = ring + (long long)r * 4 * 6144;
  for (int i = threadIdx.x; i < 320; i += blockDim.x) {
    int c = i < 64 ? h * 64 + i : i < 192 ? 4096 + grp * 128 + (i - 64) : 5120 + grp * 128 + (i - 192);
    float now = in[c];
    float4 k = *(const float4*)(conv_w + (long long)c * 4);
    float acc = conv_b[c] + k.w * now + k.z * ring_r[((t - 1) & 3) * 6144 + c] + k.y * ring_r[((t - 2) & 3) * 6144 + c] +
                k.x * ring_r[((t - 3) & 3) * 6144 + c];
    acc = silu(acc);
    if (i < 64) xs[i] = acc;
    else if (i < 192) Bs[i - 64] = acc;
    else Cs[i - 192] = acc;
    if (i < 64 || h % 8 == 0) ring_r[(t & 3) * 6144 + c] = now;
  }
  __syncthreads();
  float dt = softplus(proj[(long long)r * 10304 + 10240 + h] + dt_bias[h]);
  float dA = expf(dt * A[h]);
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float* s = state + ((long long)r * 64 + h) * 64 * 128;
  for (int d = warp; d < 64; d += WARPS) {
    float xd = xs[d] * dt, y = 0.0f;
    float4* row = (float4*)(s + d * 128) + lane;
    float4 sv = *row;
    sv.x = sv.x * dA + xd * Bs[lane * 4];
    sv.y = sv.y * dA + xd * Bs[lane * 4 + 1];
    sv.z = sv.z * dA + xd * Bs[lane * 4 + 2];
    sv.w = sv.w * dA + xd * Bs[lane * 4 + 3];
    *row = sv;
    y = sv.x * Cs[lane * 4] + sv.y * Cs[lane * 4 + 1] + sv.z * Cs[lane * 4 + 2] + sv.w * Cs[lane * 4 + 3];
    y = warp_sum(y);
    if (lane == 0) ys[d] = y + D[h] * xs[d];
  }
  __syncthreads();
  float sq = 0.0f;
  if (threadIdx.x < 64) {
    int c = h * 64 + threadIdx.x;
    float out = ys[threadIdx.x] * silu(proj[(long long)r * 10304 + c]);
    v[(long long)r * 4096 + c] = out;
    sq = out * out;
  }
  if (threadIdx.x < 64) {
    sq = warp_sum(sq);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = sq;
  }
  __syncthreads();
  if (threadIdx.x == 0) part[r * 64 + h] = sums[0] + sums[1];
}

// Decode attention split over key rows (flash-decoding): q [R, 32, 128] float32 (16 query heads per key/value head),
// keys and values [R, 2, cap, 128] bf16 with rows [0, pos[r] + 1) written. Block (kv head, half, split; z = row) runs
// rows [split * chunk, (split + 1) * chunk) for 8 query heads with an online softmax into part_m, part_l [R, 32,
// splits] and part_acc [R, 32, splits, 128]. Each warp takes every fourth key row, each lane 4 of the 128 columns.
// Grid (4, splits, R), block 128.
extern "C" __global__ void __launch_bounds__(128) attend_split(const float* q, const u16* keys, const u16* values,
                                                               const int* pos, int cap, int chunk, double scale,
                                                               float* part_m, float* part_l, float* part_acc) {
  __shared__ float shared_m[4][8], shared_l[4][8], shared_acc[4][8][128];
  int kv = blockIdx.x >> 1, half = blockIdx.x & 1, split = blockIdx.y, splits = gridDim.y, r = blockIdx.z;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int length = pos[r] + 1, first = split * chunk, last = min(first + chunk, length);
  int head0 = kv * 16 + half * 8;
  float qv[8][4], m[8], l[8], acc[8][4];
  for (int g = 0; g < 8; g++) {
    float4 p = *(const float4*)(q + ((long long)r * 32 + head0 + g) * 128 + lane * 4);
    qv[g][0] = p.x * (float)scale; qv[g][1] = p.y * (float)scale; qv[g][2] = p.z * (float)scale; qv[g][3] = p.w * (float)scale;
    m[g] = NEG_INF; l[g] = 0.0f;
    for (int i = 0; i < 4; i++) acc[g][i] = 0.0f;
  }
  const u16* k_head = keys + ((long long)r * 2 + kv) * cap * 128 + lane * 4;
  const u16* v_head = values + ((long long)r * 2 + kv) * cap * 128 + lane * 4;
  for (int t = first + warp; t < last; t += 4) {
    uint2 kp = *(const uint2*)(k_head + (long long)t * 128), vp = *(const uint2*)(v_head + (long long)t * 128);
    float k0 = bf_lo(kp.x), k1 = bf_hi(kp.x), k2 = bf_lo(kp.y), k3 = bf_hi(kp.y);
    float v0 = bf_lo(vp.x), v1 = bf_hi(vp.x), v2 = bf_lo(vp.y), v3 = bf_hi(vp.y);
#pragma unroll
    for (int g = 0; g < 8; g++) {
      float d = warp_sum(qv[g][0] * k0 + qv[g][1] * k1 + qv[g][2] * k2 + qv[g][3] * k3);
      float top = fmaxf(m[g], d), keep = __expf(m[g] - top), weight = __expf(d - top);
      l[g] = l[g] * keep + weight;
      acc[g][0] = acc[g][0] * keep + weight * v0; acc[g][1] = acc[g][1] * keep + weight * v1;
      acc[g][2] = acc[g][2] * keep + weight * v2; acc[g][3] = acc[g][3] * keep + weight * v3;
      m[g] = top;
    }
  }
  for (int g = 0; g < 8; g++) {
    if (lane == 0) { shared_m[warp][g] = m[g]; shared_l[warp][g] = l[g]; }
    for (int i = 0; i < 4; i++) shared_acc[warp][g][lane * 4 + i] = acc[g][i];
  }
  __syncthreads();
  int column = threadIdx.x;
  for (int g = 0; g < 8; g++) {
    float top = NEG_INF;
    for (int w = 0; w < 4; w++) top = fmaxf(top, shared_m[w][g]);
    float total = 0.0f, value = 0.0f;
    if (top > NEG_INF) {
      for (int w = 0; w < 4; w++) {
        if (shared_m[w][g] == NEG_INF) continue;
        float f = __expf(shared_m[w][g] - top);
        total += shared_l[w][g] * f;
        value += shared_acc[w][g][column] * f;
      }
    }
    long long at = ((long long)r * 32 + head0 + g) * splits + split;
    part_acc[at * 128 + column] = value;
    if (column == 0) { part_m[at] = top; part_l[at] = total; }
  }
}

// The splits of attend_split merged: out [R, 32 * 128] float32. Grid (32, R), block 128.
extern "C" __global__ void __launch_bounds__(128) attend_combine(const float* part_m, const float* part_l,
                                                                 const float* part_acc, int splits, float* out) {
  int head = blockIdx.x, r = blockIdx.y, column = threadIdx.x;
  long long base = ((long long)r * 32 + head) * splits;
  float top = NEG_INF;
  for (int s = 0; s < splits; s++) top = fmaxf(top, part_m[base + s]);
  float total = 0.0f, value = 0.0f;
  for (int s = 0; s < splits; s++) {
    float ms = part_m[base + s];
    if (ms == NEG_INF) continue;
    float f = __expf(ms - top);
    total += part_l[base + s] * f;
    value += part_acc[(base + s) * 128 + column] * f;
  }
  out[((long long)r * 32 + head) * 128 + column] = value / total;
}
"""


@functools.cache
def _kernel(name, device_index, defines):
    options = [f"-D{k}={v}" for k, v in defines] + ["-std=c++17"]
    with torch.cuda.device(device_index):
        kernel = torch.cuda._compile_kernel(SOURCE, name, nvcc_options=options)
    return kernel


def _defines(**defines):
    values = {"R": 1, "RW": DENSE_ROWS_PER_WARP, "DRW": DOWN_ROWS_PER_WARP, "IN": 0, "ROUND": 0, "OUT": 0, "FMT": 0, "G": 1}
    values.update(defines)
    return tuple(sorted(values.items()))


def _index(device):
    return device.index if device.index is not None else torch.cuda.current_device()


def _launch(name, device, grid, block, args, shared_mem=0, **defines):
    _kernel(name, _index(device), _defines(**defines))(grid=grid, block=block, args=args, shared_mem=shared_mem,
                                                       stream=torch.cuda.current_stream(device))


@functools.cache
def available(device):
    """Whether the kernels compile for `device`: a CUDA device and NVRTC."""
    if device.type != "cuda":
        return False
    try:
        _kernel("dense", _index(device), _defines())
    except (AttributeError, RuntimeError, OSError):  # PyTorch without the NVRTC helper, or NVRTC missing
        return False
    return True


class Bank:
    """An expert bank's fields as the expert kernels read them: format ("bf16", "int8", "w4", "w6"), weight rows
    (bf16 [rows, width], int8 [rows, width], packed codes [rows, width * bits / 8]), e4m3 group scales (W4A8/W6A8,
    [rows, width / 16]), row scales (int8 scale, W4A8/W6A8 s_channel, float32 [rows]) and the 4-bit codebook [16]."""

    def __init__(self, fmt, weight, rel=None, scale=None, codebook=None):
        self.fmt, self.weight, self.rel, self.scale, self.codebook = fmt, weight, rel, scale, codebook

    def args(self):
        # the launcher passes pointers; a field the format does not read points at the weight
        return (self.weight, self.weight if self.rel is None else self.rel, self.weight if self.scale is None else self.scale,
                self.weight if self.codebook is None else self.codebook)


def dense(x, w, out=None, resid=None, norm=None, eps=1e-5, gated_part=None, round_input=False, segments=None,
          kv=None, pos=None):
    """R rows of x [R, K] through a bf16 weight (one [N, K] tensor, or up to three row segments `segments`).
    norm: the input's RMS norm weight [K] (float32; x bf16), or with gated_part [R, 64] Mamba's gated group norm over
    x float32. Writes out [R, N] float32 (or, attention, q [R, n1] with kv = (keys, values, cap) caches at pos), or
    adds into resid [R, N] bf16."""
    rows, k = x.shape
    ws = segments if segments is not None else [w]
    n_seg = [t.shape[0] for t in ws]
    n = sum(n_seg)
    w0, w1, w2 = (ws + [ws[-1]] * 3)[:3]
    n1 = n_seg[0] if len(ws) > 1 else n
    n2 = n1 + n_seg[1] if len(ws) > 2 else n
    mode_in = 0 if norm is None else 2 if gated_part is not None else 1
    mode_out = 2 if kv is not None else 1 if resid is not None else 0
    dummy = x
    keys, values, cap = kv if kv is not None else (dummy, dummy, 0)
    args = [x, norm if norm is not None else dummy, gated_part if gated_part is not None else dummy, float(eps), k,
            w0, w1, w2, n1, n2, n, out if out is not None else dummy, resid if resid is not None else dummy, keys,
            values, pos if pos is not None else dummy, cap]
    per_block = 8 * DENSE_ROWS_PER_WARP
    _launch("dense", x.device, (-(-n // per_block), 1, 1), (256, 1, 1), args, R=rows, IN=mode_in,
            ROUND=int(round_input), OUT=mode_out)


def dense_q8(x, segments, group, scratch, out=None, resid=None, norm=None, eps=1e-5, gated_part=None, kv=None, pos=None):
    """dense for int8 ConvRot weights: segments [(int8 [n, K], row scales [n] float32)] (up to three), ConvRot group
    `group`; the input transform as dense (norm, gated_part; without norm the input is rounded to bf16). scratch is
    (int16 [R, >= K], float32 [R, >= K / 16]) for the prepared input."""
    rows, k = x.shape
    xq, xsc = scratch
    mode_in = 0 if norm is None else 2 if gated_part is not None else 1
    dummy = x
    _launch("prep_rows", x.device, (rows, 1, 1), (256, 1, 1),
            [x, norm if norm is not None else dummy, gated_part if gated_part is not None else dummy, float(eps), k, xq, xsc],
            R=rows, IN=mode_in, G=group)
    n_seg = [w.shape[0] for w, _ in segments]
    n = sum(n_seg)
    (w0, s0), (w1, s1), (w2, s2) = (segments + [segments[-1]] * 3)[:3]
    n1 = n_seg[0] if len(segments) > 1 else n
    n2 = n1 + n_seg[1] if len(segments) > 2 else n
    mode_out = 2 if kv is not None else 1 if resid is not None else 0
    keys, values, cap = kv if kv is not None else (dummy, dummy, 0)
    args = [xq, xsc, k, w0, s0, w1, s1, w2, s2, n1, n2, n, out if out is not None else dummy,
            resid if resid is not None else dummy, keys, values, pos if pos is not None else dummy, cap]
    _launch("dense_q8", x.device, (-(-n // (8 * DENSE_ROWS_PER_WARP)), 1, 1), (256, 1, 1), args, R=rows, OUT=mode_out)


def head_blocks(n):
    return -(-n // (8 * DENSE_ROWS_PER_WARP))


def head(x, norm, w, eps, temperature, seed, force, best_v, best_i, counter, tokens, emb, pos, logits=None,
         vision_base=VISION_BASE, row0=0, keep=False, external=False):
    """The step's last kernel (see the CUDA source): samples or forces each row's next token into tokens, writes its
    embedding row into x [R, K] (the residual stream) and advances pos. force / tokens [R, T] int32. With `keep`, logits
    [R, N] are written at every step; with `external`, the token is left to `accept`."""
    rows, k = x.shape
    n = w.shape[0]
    greedy = temperature <= 0
    seed = (int(seed) ^ (int(seed) >> 32)) & 0xFFFFFFFF  # ComfyUI seeds are 64-bit; the launcher passes a C int
    args = [x, norm, float(eps), k, w, n, 0.0 if greedy else 1.0 / temperature, int(greedy), seed - (seed >> 31 << 32), int(row0), force,
            force.shape[1], vision_base, int(keep), int(external), logits if logits is not None else best_v, best_v,
            best_i, counter, tokens, emb, x, pos]
    _launch("head", x.device, (head_blocks(n), 1, 1), (256, 1, 1), args, R=rows)


def accept(sampled, force, tokens, emb, x, pos, vision_base=VISION_BASE):
    """Each row's next token from `sampled` [R] (int64, a vision code) unless the schedule forces one (see accept)."""
    rows, k = x.shape
    _launch("accept", x.device, (1, 1, 1), (256, 1, 1),
            [sampled, force, force.shape[1], vision_base, tokens, emb, x, k, pos], R=rows)


def gather_rows(ids, table, out):
    """out [n, K] (bf16) = rows `ids` [n] (int64, on the card) of table [V, K] (bf16, pinned host memory or the card)."""
    _launch("gather_rows", out.device, (ids.numel(), 1, 1), (128, 1, 1), [ids, table, table.shape[1], out])


def max_units(rows):
    return 2 + min(EXPERTS, TOP_K * rows)


class MoEBuffers:
    """The fixed buffers one MoE layer's kernels use for R rows (reused across layers and steps)."""

    def __init__(self, rows, device):
        units = max_units(rows)
        f32, i32 = dict(dtype=torch.float32, device=device), dict(dtype=torch.int32, device=device)
        with torch.inference_mode(False):
            self.logits = torch.zeros((rows, EXPERTS), **f32)
            self.xr = torch.zeros((rows, HIDDEN), **f32)
            self.xq = torch.zeros((rows, HIDDEN), dtype=torch.int16, device=device)
            self.xsc = torch.zeros((rows, HIDDEN // 16), **f32)
            self.ids = torch.zeros((rows, TOP_K), **i32)
            self.weights = torch.zeros((rows, TOP_K), **f32)
            self.n_units = torch.zeros((1,), **i32)
            self.unit_id = torch.zeros((units,), **i32)
            self.unit_n = torch.zeros((units,), **i32)
            self.unit_row = torch.zeros((units, rows), **i32)
            self.unit_w = torch.zeros((units, rows), **f32)
            self.hid = torch.zeros((units, rows, EXPERT), **f32)
            self.hidq = torch.zeros((units, rows, EXPERT), dtype=torch.int16, device=device)
            self.hids = torch.zeros((units, rows, EXPERT // 16), **f32)
            self.part = torch.zeros((units, rows, HIDDEN), **f32)
            self.route_counter = torch.zeros((1,), **i32)
            self.unit_slot = torch.zeros((units,), **i32)
            self.unit_fill = torch.zeros((units,), **i32)
            self.down_counter = torch.zeros((HIDDEN // (8 * DOWN_ROWS_PER_WARP),), **i32)
        self.rows, self.units = rows, units


class ExpertCache:
    """Expert slots on the card for banks kept in host memory: `slots` experts, each its up rows [EXPERT, HIDDEN wide]
    and down rows [HIDDEN, EXPERT wide] in the bank's stored layout (format `fmt`, the banks' 4-bit codebook), one LRU
    over every MoE layer (`layers` of them). The table, owners, stamps and clock live on the card; `stats` counts hits
    and misses."""

    def __init__(self, fmt, slots, layers, codebook, device):
        def fields(rows, width):
            row = {"bf16": 2 * width, "int8": width, "w4": width // 2, "w6": width * 3 // 4}[fmt]
            w = torch.empty((slots * rows, row), dtype=torch.uint8, device=device)
            rel = torch.empty((slots * rows, width // 16), dtype=torch.uint8, device=device) if fmt in ("w4", "w6") else None
            scale = torch.empty((slots * rows,), dtype=torch.float32, device=device) if fmt != "bf16" else None
            return Bank(fmt, w, rel=rel, scale=scale, codebook=codebook)

        i32 = dict(dtype=torch.int32, device=device)
        with torch.inference_mode(False):
            self.up, self.down = fields(EXPERT, HIDDEN), fields(HIDDEN, EXPERT)
            self.table = torch.full((layers * EXPERTS,), -1, **i32)
            self.owner = torch.full((slots,), -1, **i32)
            self.last = torch.full((slots,), -1, **i32)
            self.clock = torch.zeros((1,), **i32)
            self.stats = torch.zeros((2,), **i32)
        self.slots = slots

    @staticmethod
    def slot_bytes(fmt):
        per_weight = {"bf16": 2, "int8": 1, "w6": 0.75 + 1 / 16, "w4": 0.5 + 1 / 16}[fmt]
        return int(2 * EXPERT * HIDDEN * per_weight) + (0 if fmt == "bf16" else (EXPERT + HIDDEN) * 4)


def route(x, norm, eps, router, bias, buffers, group, layer=0, cache=None):
    """MoE routing for x [R, H] (bf16 residual stream): fills buffers (logits, xr in the ConvRot basis of rotation
    group `group`, ids, weights, the unit table) and each routed unit's slot: its expert's index in a resident bank, or
    with `cache` (an ExpertCache) its slot for MoE layer `layer`."""
    b = buffers
    c = cache
    cache_args = [layer, 0] + [b.unit_slot] * 5 if c is None else [layer, c.slots, c.table, c.owner, c.last, c.clock, c.stats]
    args = [x, norm, float(eps), x.shape[1], router, bias, b.logits, b.xr, b.xq, b.xsc, b.ids, b.weights, b.n_units, b.unit_id,
            b.unit_n, b.unit_row, b.unit_w, b.route_counter, *cache_args, b.unit_slot, b.unit_fill]
    _launch("route", x.device, (EXPERTS // 8, 1, 1), (256, 1, 1), args, R=x.shape[0], G=group)


def experts(x, buffers, up, down, shared_up, shared_down, group, host_up=None, host_down=None):
    """The routed and shared experts of the rows `route` filled buffers for, added into x [R, H] (bf16). up / down are
    the routed banks on the card (resident, or an ExpertCache's slots, then host_up / host_down are the banks in
    pinned host memory that misses are read from). The shared expert may be stored in another format than the routed
    banks (its own launches then)."""
    b, device = buffers, x.device
    host_up, host_down = host_up or up, host_down or down
    split = shared_up.fmt != up.fmt
    slices = HIDDEN // (8 * DOWN_ROWS_PER_WARP)
    launches = [(shared_up, shared_down, 0, 2, 0), (up, down, 2, b.units - 2, 1)] if split else [(up, down, 0, b.units, 1)]
    for bank_up, _, u0, count, _ in launches:
        args = [b.xr, b.xq, b.xsc, HIDDEN, EXPERT, b.n_units, b.unit_id, b.unit_n, b.unit_row, *up.args(),
                *shared_up.args(), *host_up.args()[:3], b.unit_slot, b.unit_fill, b.hid, b.hidq, b.hids, u0]
        _launch("experts_up", device, (EXPERT // 64, count, 1), (256, 1, 1), args, R=b.rows, FMT=FORMATS[bank_up.fmt], G=group)
    for _, bank_down, u0, count, combine in launches:
        args = [b.hid, b.hidq, b.hids, EXPERT, HIDDEN, b.n_units, b.unit_id, b.unit_n, b.unit_row, b.unit_w, *down.args(),
                *shared_down.args(), *host_down.args()[:3], b.unit_slot, b.unit_fill, b.part, b.down_counter, x, u0, combine]
        _launch("experts_down", device, (slices, count, 1), (256, 1, 1), args, R=b.rows, FMT=FORMATS[bank_down.fmt], G=group)


def experts_down_only(x, buffers, down, shared_down, group):
    """experts_down alone over buffers' unit table and hidden rows (checks)."""
    b, fmt = buffers, FORMATS[down.fmt]
    args = [b.hid, b.hidq, b.hids, EXPERT, HIDDEN, b.n_units, b.unit_id, b.unit_n, b.unit_row, b.unit_w, *down.args(),
            *shared_down.args(), *down.args()[:3], b.unit_slot, b.unit_fill, b.part, b.down_counter, x, 0, 1]
    _launch("experts_down", x.device, (HIDDEN // (8 * DOWN_ROWS_PER_WARP), b.units, 1), (256, 1, 1), args, R=b.rows,
            FMT=fmt, G=group)


def mamba_step(proj, ring, conv_w, conv_b, state, A, dt_bias, D, pos, v, part):
    """A Mamba2 step for R rows (see the CUDA source)."""
    rows = proj.shape[0]
    _launch("mamba_step", proj.device, (MAMBA_HEADS, rows, 1), (256, 1, 1),
            [proj, ring, conv_w, conv_b, state, A, dt_bias, D, pos, v, part])


def attend(q, keys, values, pos, part_m, part_l, part_acc, out):
    """Attention of q [R, 32 * 128] (float32) over the caches [R, 2, cap, 128] (bf16) up to pos[r]: out [R, 4096]
    float32. part_* are the split buffers [R, 32, splits] and [R, 32, splits, 128]."""
    rows, cap, splits = q.shape[0], keys.shape[2], part_m.shape[-1]
    _launch("attend_split", q.device, (4, splits, rows), (128, 1, 1),
            [q, keys, values, pos, cap, ATTEND_CHUNK, HEAD_DIM ** -0.5, part_m, part_l, part_acc])
    _launch("attend_combine", q.device, (Q_HEADS, rows, 1), (128, 1, 1), [part_m, part_l, part_acc, splits, out])
