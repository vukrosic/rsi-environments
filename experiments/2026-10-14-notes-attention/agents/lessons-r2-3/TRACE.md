# Agent `lessons-r2-3`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what is still true from the notes you were given. At most 300 words, in words, no code.

14 min, 98,617 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 7.595 ms | 8.88x |
| run2 | 9.149 ms | 7.8x |
| run3 | 11.472 ms | 6.14x |
| run4 | 5.204 ms | 12.78x |
| run5 | 7.868 ms | 8.94x |
| run6 | 5.57 ms | 12.25x |
| run7 | 5.541 ms | 12.1x |
| run8 | 4.131 ms | 17.55x |
| run9 | 5.412 ms | 12.88x |
| run10 | 4.097 ms | 17.36x |
| final | 3.741 ms | 18.25x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the task files and earlier lessons, then use correctness checks and the ten-run benchmark budget to tune the kernel.

### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -1,10 +1,107 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
 import torch
+import torch.nn.functional as F
+
+
+_BQ = 64
+_BK = 64
+_SOURCE = r"""
+#include <metal_stdlib>
+#include <metal_simdgroup_matrix>
+using namespace metal;
+constant uint BQ = 64;
+constant uint BK = 64;
+kernel void attention(
+    device const float *q [[buffer(0)]],
+    device const float *k [[buffer(1)]],
+    device const float *v [[buffer(2)]],
+    device float *out [[buffer(3)]],
+    uint2 group [[threadgroup_position_in_grid]],
+    uint tid [[thread_index_in_threadgroup]],
+    uint sg [[simdgroup_index_in_threadgroup]],
+    uint lane [[thread_index_in_simdgroup]]) {
+    threadgroup half stage[(BQ > BK ? BQ : BK) * 64];
+    threadgroup float scores[BQ * BK];
+    threadgroup half probs[BQ * BK];
+    const uint base = group.y * 1024 * 64;
+    const uint query = group.x * BQ;
+    for (uint i = tid; i < BQ * 64; i += 256)
+        stage[i] = half(q[base + query * 64 + i]);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    simdgroup_half8x8 qm[8];
+    simdgroup_float8x8 acc[8];
+    for (uint d = 0; d < 8; ++d) {
+        simdgroup_load(qm[d], stage + sg * 8 * 64 + d * 8, 64);
+        acc[d] = simdgroup_float8x8(0.0f);
+    }
+    float maxima[8], denom[8];
+    for (uint r = 0; r < 8; ++r) { maxima[r] = -INFINITY; denom[r] = 0.0f; }
+    const uint matrix_row = ((lane & 7) >> 1) + ((lane >> 4) << 2);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    for (uint key = 0; key < query + BQ; key += BK) {
+        for (uint i = tid; i < BK * 64; i += 256)
+            stage[i] = half(k[base + key * 64 + i]);
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for (uint c = 0; c < BK / 8; ++c) {
+            simdgroup_float8x8 s(0.0f);
+            for (uint d = 0; d < 8; ++d) {
+                simdgroup_half8x8 km;
+                simdgroup_load(km, stage + c * 8 * 64 + d * 8, 64, ulong2(0), true);
+                simdgroup_multiply_accumulate(s, qm[d], km, s);
+            }
+            simdgroup_store(s, scores + sg * 8 * BK + c * 8, BK);
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for (uint i = tid; i < BK * 64; i += 256)
+            stage[i] = half(v[base + key * 64 + i]);
+        float alpha[8];
+        for (uint r = 0; r < 8; ++r) {
+            const uint row = sg * 8 + r;
+            float vals[BK / 32];
+            float mx = -INFINITY;
+            for (uint j = 0; j < BK / 32; ++j) {
+                uint c = lane + j * 32;
+                float s = scores[row * BK + c] * 0.125f;
+                s = key + c <= query + row ? s : -INFINITY;
+                vals[j] = s;
+                mx = max(mx, s);
+            }
+            mx = max(maxima[r], simd_max(mx));
+            alpha[r] = fast::exp(maxima[r] - mx);
+            float sum = 0.0f;
+            for (uint j = 0; j < BK / 32; ++j) {
+                float p = fast::exp(vals[j] - mx);
+                probs[row * BK + lane + j * 32] = half(p);
+                sum += p;
+            }
+            denom[r] = denom[r] * alpha[r] + simd_sum(sum);
+            maxima[r] = mx;
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for (uint d = 0; d < 8; ++d) {
+            acc[d].thread_elements()[0] *= alpha[matrix_row];
+            acc[d].thread_elements()[1] *= alpha[matrix_row];
+            for (uint c = 0; c < BK / 8; ++c) {
+                simdgroup_half8x8 pm, vm;
+                simdgroup_load(pm, probs + sg * 8 * BK + c * 8, BK);
+                simdgroup_load(vm, stage + c * 8 * 64 + d * 8, 64);
+                simdgroup_multiply_accumulate(acc[d], pm, vm, acc[d]);
+            }
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+    }
+    for (uint d = 0; d < 8; ++d) {
+        acc[d].thread_elements()[0] /= denom[matrix_row];
+        acc[d].thread_elements()[1] /= denom[matrix_row];
+        simdgroup_store(acc[d], out + base + (query + sg * 8) * 64 + d * 8, 64);
+    }
+}
+"""
+_LIB = torch.mps.compile_shader(_SOURCE)
 
 
 def kernel(q, k, v):
-    T = q.size(-2)
-    s = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
-    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
-    e = (s - s.max(-1, keepdim=True).values).exp()
-    return (e / e.sum(-1, keepdim=True)) @ v
+    if q.shape != (8, 8, 1024, 64) or not all(x.is_contiguous() for x in (q, k, v)):
+        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
+    out = torch.empty_like(q)
+    _LIB.attention(q, k, v, out, threads=(1024 // _BQ * 256, 64, 1), group_size=(256, 1, 1))
+    return out

```

</details>

`pwd && rg --files -g 'AGENTS.md' -g 'kernel.py' -g 'reference.py' -g 'bench.py' -g 'check.py' -g 'LESSONS.md`  
→ check.py  
→ kernel.py

### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -20,8 +20,6 @@
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
     threadgroup half stage[(BQ > BK ? BQ : BK) * 64];
-    threadgroup float scores[BQ * BK];
-    threadgroup half probs[BQ * BK];
     const uint base = group.y * 1024 * 64;
     const uint query = group.x * BQ;
     for (uint i = tid; i < BQ * 64; i += 256)
@@ -33,65 +31,68 @@
         simdgroup_load(qm[d], stage + sg * 8 * 64 + d * 8, 64);
         acc[d] = simdgroup_float8x8(0.0f);
     }
-    float maxima[8], denom[8];
-    for (uint r = 0; r < 8; ++r) { maxima[r] = -INFINITY; denom[r] = 0.0f; }
+    float maxima = -INFINITY, denom = 0.0f;
     const uint matrix_row = ((lane & 7) >> 1) + ((lane >> 4) << 2);
+    const uint matrix_col = ((lane & 8) >> 1) + ((lane & 1) << 1);
+    const uint row = query + sg * 8 + matrix_row;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for (uint key = 0; key < query + BQ; key += BK) {
         for (uint i = tid; i < BK * 64; i += 256)
             stage[i] = half(k[base + key * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
+        simdgroup_float8x8 scores[BK / 8];
         for (uint c = 0; c < BK / 8; ++c) {
-            simdgroup_float8x8 s(0.0f);
+            scores[c] = simdgroup_float8x8(0.0f);
             for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km, stage + c * 8 * 64 + d * 8, 64, ulong2(0), true);
-                simdgroup_multiply_accumulate(s, qm[d], km, s);
+                simdgroup_multiply_accumulate(scores[c], qm[d], km, scores[c]);
             }
-            simdgroup_store(s, scores + sg * 8 * BK + c * 8, BK);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for (uint i = tid; i < BK * 64; i += 256)
             stage[i] = half(v[base + key * 64 + i]);
-        float alpha[8];
-        for (uint r = 0; r < 8; ++r) {
-            const uint row = sg * 8 + r;
-            float vals[BK / 32];
-            float mx = -INFINITY;
-            for (uint j = 0; j < BK / 32; ++j) {
-                uint c = lane + j * 32;
-                float s = scores[row * BK + c] * 0.125f;
-                s = key + c <= query + row ? s : -INFINITY;
-                vals[j] = s;
+        float mx = -INFINITY;
+        for (uint c = 0; c < BK / 8; ++c) {
+            for (uint i = 0; i < 2; ++i) {
+                float s = scores[c].thread_elements()[i] * 0.125f;
+                s = key + c * 8 + matrix_col + i <= row ? s : -INFINITY;
+                scores[c].thread_elements()[i] = s;
                 mx = max(mx, s);
             }
-            mx = max(maxima[r], simd_max(mx));
-            alpha[r] = fast::exp(maxima[r] - mx);
-            float sum = 0.0f;
-            for (uint j = 0; j < BK / 32; ++j) {
-                float p = fast::exp(vals[j] - mx);
-                probs[row * BK + lane + j * 32] = half(p);
+        }
+        mx = max(mx, simd_shuffle_xor(mx, 1));
+        mx = max(mx, simd_shuffle_xor(mx, 8));
+        mx = max(mx, maxima);
+        float alpha = fast::exp(maxima - mx);
+        simdgroup_half8x8 pm[BK / 8];
+        float sum = 0.0f;
+        for (uint c = 0; c < BK / 8; ++c) {
+            for (uint i = 0; i < 2; ++i) {
+                float p = fast::exp(scores[c].thread_elements()[i] - mx);
+                pm[c].thread_elements()[i] = half(p);
                 sum += p;
             }
-            denom[r] = denom[r] * alpha[r] + simd_sum(sum);
-            maxima[r] = mx;
         }
+        sum += simd_shuffle_xor(sum, 1);
+        sum += simd_shuffle_xor(sum, 8);
+        denom = denom * alpha + sum;
+        maxima = mx;
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for (uint d = 0; d < 8; ++d) {
-            acc[d].thread_elements()[0] *= alpha[matrix_row];
-            acc[d].thread_elements()[1] *= alpha[matrix_row];
+            acc[d].thread_elements()[0] *= alpha;
+            acc[d].thread_elements()[1] *= alpha;
             for (uint c = 0; c < BK / 8; ++c) {
-                simdgroup_half8x8 pm, vm;
-                simdgroup_load(pm, probs + sg * 8 * BK + c * 8, BK);
+                simdgroup_half8x8 vm;
                 simdgroup_load(vm, stage + c * 8 * 64 + d * 8, 64);
-                simdgroup_multiply_accumulate(acc[d], pm, vm, acc[d]);
+                simdgroup_multiply_accumulate(acc[d], pm[c], vm, acc[d]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
     for (uint d = 0; d < 8; ++d) {
-        acc[d].thread_elements()[0] /= denom[matrix_row];
-        acc[d].thread_elements()[1] /= denom[matrix_row];
+        acc[d].thread_elements()[0] /= denom;
+        acc[d].thread_elements()[1] /= denom;
         simdgroup_store(acc[d], out + base + (query + sg * 8) * 64 + d * 8, 64);
     }
 }

```

</details>

`cat kernel.py reference.py bench.py check.py LESSONS.md`  
→ Fully fused online attention was slower: query/key tile sizes 32/32, 64/32, and 64/64 took 10.45, 10.40, and 11.25 milliseconds. Reusing shared memory and retaining probabilities in registers with 128/64 tiles took 10.31. One trial failed compilation because Metal max is not a constant expression for array sizes; a ternary expression works.  
→ All timed successful variants passed correctness. GPU contention changes reference times, so compare absolute kernel times too.

- `python -c 'import torch; print(torch.__version__); help(torch.mps.compile_shader)`
> The earlier notes point to a fused Metal kernel as the fastest approach. I’ll start with half precision matrix operands, float32 accumulation, and online softmax, then tune the tile sizes.

- `python - <<'PY` (exited 1 in 0ms:)
- `python -c 'import torch`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.46e-04 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -2,13 +2,13 @@
 import torch.nn.functional as F
 
 
-_BQ = 64
+_BQ = 32
 _BK = 64
 _SOURCE = r"""
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-constant uint BQ = 64;
+constant uint BQ = 32;
 constant uint BK = 64;
 kernel void attention(
     device const float *q [[buffer(0)]],
@@ -22,7 +22,7 @@
     threadgroup half stage[(BQ > BK ? BQ : BK) * 64];
     const uint base = group.y * 1024 * 64;
     const uint query = group.x * BQ;
-    for (uint i = tid; i < BQ * 64; i += 256)
+    for (uint i = tid; i < BQ * 64; i += 128)
         stage[i] = half(q[base + query * 64 + i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
@@ -37,7 +37,7 @@
     const uint row = query + sg * 8 + matrix_row;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for (uint key = 0; key < query + BQ; key += BK) {
-        for (uint i = tid; i < BK * 64; i += 256)
+        for (uint i = tid; i < BK * 64; i += 128)
             stage[i] = half(k[base + key * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 scores[BK / 8];
@@ -50,7 +50,7 @@
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < BK * 64; i += 256)
+        for (uint i = tid; i < BK * 64; i += 128)
             stage[i] = half(v[base + key * 64 + i]);
         float mx = -INFINITY;
         for (uint c = 0; c < BK / 8; ++c) {
@@ -104,5 +104,5 @@
     if q.shape != (8, 8, 1024, 64) or not all(x.is_contiguous() for x in (q, k, v)):
         return F.scaled_dot_product_attention(q, k, v, is_causal=True)
     out = torch.empty_like(q)
-    _LIB.attention(q, k, v, out, threads=(1024 // _BQ * 256, 64, 1), group_size=(256, 1, 1))
+    _LIB.attention(q, k, v, out, threads=(1024 // _BQ * 128, 64, 1), group_size=(128, 1, 1))
     return out

```

</details>

`python bench.py`  
→ run 1 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 67.48 ms, kernel 7.59 ms, speedup 8.88x

- `cat kernel.py`
> The first fused kernel passed and took 7.59 ms, an 8.88× speedup. I’ll now try keeping scores and probabilities in registers to reduce shared-memory traffic and softmax work.

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.70e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -2,13 +2,13 @@
 import torch.nn.functional as F
 
 
-_BQ = 32
+_BQ = 64
 _BK = 64
 _SOURCE = r"""
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-constant uint BQ = 32;
+constant uint BQ = 64;
 constant uint BK = 64;
 kernel void attention(
     device const float *q [[buffer(0)]],
@@ -22,11 +22,12 @@
     threadgroup half stage[(BQ > BK ? BQ : BK) * 64];
     const uint base = group.y * 1024 * 64;
     const uint query = group.x * BQ;
-    for (uint i = tid; i < BQ * 64; i += 128)
+    for (uint i = tid; i < BQ * 64; i += 256)
         stage[i] = half(q[base + query * 64 + i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 acc[8];
+    #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d) {
         simdgroup_load(qm[d], stage + sg * 8 * 64 + d * 8, 64);
         acc[d] = simdgroup_float8x8(0.0f);
@@ -37,23 +38,27 @@
     const uint row = query + sg * 8 + matrix_row;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for (uint key = 0; key < query + BQ; key += BK) {
-        for (uint i = tid; i < BK * 64; i += 128)
+        for (uint i = tid; i < BK * 64; i += 256)
             stage[i] = half(k[base + key * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 scores[BK / 8];
+        #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
             scores[c] = simdgroup_float8x8(0.0f);
-            for (uint d = 0; d < 8; ++d) {
+            #pragma clang loop unroll(full)
+    for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km, stage + c * 8 * 64 + d * 8, 64, ulong2(0), true);
                 simdgroup_multiply_accumulate(scores[c], qm[d], km, scores[c]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < BK * 64; i += 128)
+        for (uint i = tid; i < BK * 64; i += 256)
             stage[i] = half(v[base + key * 64 + i]);
         float mx = -INFINITY;
+        #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
+            #pragma clang loop unroll(full)
             for (uint i = 0; i < 2; ++i) {
                 float s = scores[c].thread_elements()[i] * 0.125f;
                 s = key + c * 8 + matrix_col + i <= row ? s : -INFINITY;
@@ -67,7 +72,9 @@
         float alpha = fast::exp(maxima - mx);
         simdgroup_half8x8 pm[BK / 8];
         float sum = 0.0f;
+        #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
+            #pragma clang loop unroll(full)
             for (uint i = 0; i < 2; ++i) {
                 float p = fast::exp(scores[c].thread_elements()[i] - mx);
                 pm[c].thread_elements()[i] = half(p);
@@ -79,10 +86,12 @@
         denom = denom * alpha + sum;
         maxima = mx;
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint d = 0; d < 8; ++d) {
+        #pragma clang loop unroll(full)
+    for (uint d = 0; d < 8; ++d) {
             acc[d].thread_elements()[0] *= alpha;
             acc[d].thread_elements()[1] *= alpha;
-            for (uint c = 0; c < BK / 8; ++c) {
+            #pragma clang loop unroll(full)
+        for (uint c = 0; c < BK / 8; ++c) {
                 simdgroup_half8x8 vm;
                 simdgroup_load(vm, stage + c * 8 * 64 + d * 8, 64);
                 simdgroup_multiply_accumulate(acc[d], pm[c], vm, acc[d]);
@@ -90,6 +99,7 @@
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
+    #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d) {
         acc[d].thread_elements()[0] /= denom;
         acc[d].thread_elements()[1] /= denom;
@@ -104,5 +114,5 @@
     if q.shape != (8, 8, 1024, 64) or not all(x.is_contiguous() for x in (q, k, v)):
         return F.scaled_dot_product_attention(q, k, v, is_causal=True)
     out = torch.empty_like(q)
-    _LIB.attention(q, k, v, out, threads=(1024 // _BQ * 128, 64, 1), group_size=(128, 1, 1))
+    _LIB.attention(q, k, v, out, threads=(1024 // _BQ * 256, 64, 1), group_size=(256, 1, 1))
     return out

```

</details>

`python bench.py`  
→     _LIB.attention(q, k, v, out, threads=(1024 // _BQ * 256, 64, 1), group_size=(256, 1, 1))  
→     return out

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text().replace(\"_BQ = 64\", \"_BQ = 32\").replace(\"constant uint BQ = 64\", \"constant...`
- `python check.py`  
→ OK: relative error 3.99e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -20,6 +20,8 @@
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
     threadgroup half stage[(BQ > BK ? BQ : BK) * 64];
+    threadgroup float scores[BQ * BK];
+    threadgroup half probs[BQ * BK];
     const uint base = group.y * 1024 * 64;
     const uint query = group.x * BQ;
     for (uint i = tid; i < BQ * 64; i += 256)
@@ -27,7 +29,6 @@
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 acc[8];
-    #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d) {
         simdgroup_load(qm[d], stage + sg * 8 * 64 + d * 8, 64);
         acc[d] = simdgroup_float8x8(0.0f);
@@ -35,34 +36,33 @@
     float maxima = -INFINITY, denom = 0.0f;
     const uint matrix_row = ((lane & 7) >> 1) + ((lane >> 4) << 2);
     const uint matrix_col = ((lane & 8) >> 1) + ((lane & 1) << 1);
-    const uint row = query + sg * 8 + matrix_row;
+    const uint row = sg * 8 + matrix_row;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for (uint key = 0; key < query + BQ; key += BK) {
         for (uint i = tid; i < BK * 64; i += 256)
             stage[i] = half(k[base + key * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 scores[BK / 8];
-        #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
-            scores[c] = simdgroup_float8x8(0.0f);
-            #pragma clang loop unroll(full)
-    for (uint d = 0; d < 8; ++d) {
+            simdgroup_float8x8 s(0.0f);
+            for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km, stage + c * 8 * 64 + d * 8, 64, ulong2(0), true);
-                simdgroup_multiply_accumulate(scores[c], qm[d], km, scores[c]);
+                simdgroup_multiply_accumulate(s, qm[d], km, s);
             }
+            simdgroup_store(s, scores + sg * 8 * BK + c * 8, BK);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for (uint i = tid; i < BK * 64; i += 256)
             stage[i] = half(v[base + key * 64 + i]);
         float mx = -INFINITY;
+        float vals[BK / 4];
         #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
-            #pragma clang loop unroll(full)
             for (uint i = 0; i < 2; ++i) {
-                float s = scores[c].thread_elements()[i] * 0.125f;
-                s = key + c * 8 + matrix_col + i <= row ? s : -INFINITY;
-                scores[c].thread_elements()[i] = s;
+                const uint col = c * 8 + matrix_col + i;
+                float s = scores[row * BK + col] * 0.125f;
+                s = key + col <= query + row ? s : -INFINITY;
+                vals[c * 2 + i] = s;
                 mx = max(mx, s);
             }
         }
@@ -70,14 +70,12 @@
         mx = max(mx, simd_shuffle_xor(mx, 8));
         mx = max(mx, maxima);
         float alpha = fast::exp(maxima - mx);
-        simdgroup_half8x8 pm[BK / 8];
         float sum = 0.0f;
         #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
-            #pragma clang loop unroll(full)
             for (uint i = 0; i < 2; ++i) {
-                float p = fast::exp(scores[c].thread_elements()[i] - mx);
-                pm[c].thread_elements()[i] = half(p);
+                float p = fast::exp(vals[c * 2 + i] - mx);
+                probs[row * BK + c * 8 + matrix_col + i] = half(p);
                 sum += p;
             }
         }
@@ -86,20 +84,18 @@
         denom = denom * alpha + sum;
         maxima = mx;
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        #pragma clang loop unroll(full)
-    for (uint d = 0; d < 8; ++d) {
+        for (uint d = 0; d < 8; ++d) {
             acc[d].thread_elements()[0] *= alpha;
             acc[d].thread_elements()[1] *= alpha;
-            #pragma clang loop unroll(full)
-        for (uint c = 0; c < BK / 8; ++c) {
-                simdgroup_half8x8 vm;
+            for (uint c = 0; c < BK / 8; ++c) {
+                simdgroup_half8x8 pm, vm;
+                simdgroup_load(pm, probs + sg * 8 * BK + c * 8, BK);
                 simdgroup_load(vm, stage + c * 8 * 64 + d * 8, 64);
-                simdgroup_multiply_accumulate(acc[d], pm[c], vm, acc[d]);
+                simdgroup_multiply_accumulate(acc[d], pm, vm, acc[d]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
-    #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d) {
         acc[d].thread_elements()[0] /= denom;
         acc[d].thread_elements()[1] /= denom;

```

</details>

`python bench.py`  
→ run 2 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 71.37 ms, kernel 9.15 ms, speedup 7.80x

- `python -c 'from pathlib import Path; s=Path(\"queue/run2.py\").read_text(); s=s.replace(\"    for (uint d =\", \"    #pragma clang loop unroll(full)\\n    fo...`
- `python check.py`  
→ OK: relative error 3.44e-04 (allowed 2e-03)
> The register version passed correctness but slowed to 9.15 ms. I’ll test explicit loop unrolling to see whether the compiler can keep the small matrix arrays in registers more effectively; a smaller query tile is also being benchmarked.

- `cat kernel.py`
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -3,13 +3,13 @@
 
 
 _BQ = 64
-_BK = 64
+_BK = 32
 _SOURCE = r"""
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
 constant uint BQ = 64;
-constant uint BK = 64;
+constant uint BK = 32;
 kernel void attention(
     device const float *q [[buffer(0)]],
     device const float *k [[buffer(1)]],
@@ -20,8 +20,6 @@
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
     threadgroup half stage[(BQ > BK ? BQ : BK) * 64];
-    threadgroup float scores[BQ * BK];
-    threadgroup half probs[BQ * BK];
     const uint base = group.y * 1024 * 64;
     const uint query = group.x * BQ;
     for (uint i = tid; i < BQ * 64; i += 256)
@@ -29,6 +27,7 @@
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 acc[8];
+    #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d) {
         simdgroup_load(qm[d], stage + sg * 8 * 64 + d * 8, 64);
         acc[d] = simdgroup_float8x8(0.0f);
@@ -36,33 +35,34 @@
     float maxima = -INFINITY, denom = 0.0f;
     const uint matrix_row = ((lane & 7) >> 1) + ((lane >> 4) << 2);
     const uint matrix_col = ((lane & 8) >> 1) + ((lane & 1) << 1);
-    const uint row = sg * 8 + matrix_row;
+    const uint row = query + sg * 8 + matrix_row;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for (uint key = 0; key < query + BQ; key += BK) {
         for (uint i = tid; i < BK * 64; i += 256)
             stage[i] = half(k[base + key * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
+        simdgroup_float8x8 scores[BK / 8];
+        #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
-            simdgroup_float8x8 s(0.0f);
-            for (uint d = 0; d < 8; ++d) {
+            scores[c] = simdgroup_float8x8(0.0f);
+            #pragma clang loop unroll(full)
+    for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km, stage + c * 8 * 64 + d * 8, 64, ulong2(0), true);
-                simdgroup_multiply_accumulate(s, qm[d], km, s);
+                simdgroup_multiply_accumulate(scores[c], qm[d], km, scores[c]);
             }
-            simdgroup_store(s, scores + sg * 8 * BK + c * 8, BK);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for (uint i = tid; i < BK * 64; i += 256)
             stage[i] = half(v[base + key * 64 + i]);
         float mx = -INFINITY;
-        float vals[BK / 4];
         #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
+            #pragma clang loop unroll(full)
             for (uint i = 0; i < 2; ++i) {
-                const uint col = c * 8 + matrix_col + i;
-                float s = scores[row * BK + col] * 0.125f;
-                s = key + col <= query + row ? s : -INFINITY;
-                vals[c * 2 + i] = s;
+                float s = scores[c].thread_elements()[i] * 0.125f;
+                s = key + c * 8 + matrix_col + i <= row ? s : -INFINITY;
+                scores[c].thread_elements()[i] = s;
                 mx = max(mx, s);
             }
         }
@@ -70,12 +70,14 @@
         mx = max(mx, simd_shuffle_xor(mx, 8));
         mx = max(mx, maxima);
         float alpha = fast::exp(maxima - mx);
+        simdgroup_half8x8 pm[BK / 8];
         float sum = 0.0f;
         #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
+            #pragma clang loop unroll(full)
             for (uint i = 0; i < 2; ++i) {
-                float p = fast::exp(vals[c * 2 + i] - mx);
-                probs[row * BK + c * 8 + matrix_col + i] = half(p);
+                float p = fast::exp(scores[c].thread_elements()[i] - mx);
+                pm[c].thread_elements()[i] = half(p);
                 sum += p;
             }
         }
@@ -84,18 +86,20 @@
         denom = denom * alpha + sum;
         maxima = mx;
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint d = 0; d < 8; ++d) {
+        #pragma clang loop unroll(full)
+    for (uint d = 0; d < 8; ++d) {
             acc[d].thread_elements()[0] *= alpha;
             acc[d].thread_elements()[1] *= alpha;
-            for (uint c = 0; c < BK / 8; ++c) {
-                simdgroup_half8x8 pm, vm;
-                simdgroup_load(pm, probs + sg * 8 * BK + c * 8, BK);
+            #pragma clang loop unroll(full)
+        for (uint c = 0; c < BK / 8; ++c) {
+                simdgroup_half8x8 vm;
                 simdgroup_load(vm, stage + c * 8 * 64 + d * 8, 64);
-                simdgroup_multiply_accumulate(acc[d], pm, vm, acc[d]);
+                simdgroup_multiply_accumulate(acc[d], pm[c], vm, acc[d]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
+    #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d) {
         acc[d].thread_elements()[0] /= denom;
         acc[d].thread_elements()[1] /= denom;

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 70.44 ms, kernel 11.47 ms, speedup 6.14x

- `cp queue/run1.py kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.17e-04 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -3,13 +3,13 @@
 
 
 _BQ = 64
-_BK = 32
+_BK = 128
 _SOURCE = r"""
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
 constant uint BQ = 64;
-constant uint BK = 32;
+constant uint BK = 128;
 kernel void attention(
     device const float *q [[buffer(0)]],
     device const float *k [[buffer(1)]],

```

</details>

`python bench.py`  


- `python check.py`  
→ OK: relative error 3.55e-04 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -3,13 +3,13 @@
 
 
 _BQ = 64
-_BK = 128
+_BK = 64
 _SOURCE = r"""
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
 constant uint BQ = 64;
-constant uint BK = 128;
+constant uint BK = 64;
 kernel void attention(
     device const float *q [[buffer(0)]],
     device const float *k [[buffer(1)]],
@@ -19,11 +19,12 @@
     uint tid [[thread_index_in_threadgroup]],
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
-    threadgroup half stage[(BQ > BK ? BQ : BK) * 64];
+    threadgroup half4 stage_vec[(BQ > BK ? BQ : BK) * 16];
+    threadgroup half *stage = (threadgroup half *)stage_vec;
     const uint base = group.y * 1024 * 64;
     const uint query = group.x * BQ;
-    for (uint i = tid; i < BQ * 64; i += 256)
-        stage[i] = half(q[base + query * 64 + i]);
+    for (uint i = tid; i < BQ * 16; i += 256)
+        stage_vec[i] = half4(((device const float4 *)(q + base + query * 64))[i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 acc[8];
@@ -38,8 +39,8 @@
     const uint row = query + sg * 8 + matrix_row;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for (uint key = 0; key < query + BQ; key += BK) {
-        for (uint i = tid; i < BK * 64; i += 256)
-            stage[i] = half(k[base + key * 64 + i]);
+        for (uint i = tid; i < BK * 16; i += 256)
+            stage_vec[i] = half4(((device const float4 *)(k + base + key * 64))[i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 scores[BK / 8];
         #pragma clang loop unroll(full)
@@ -53,8 +54,8 @@
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < BK * 64; i += 256)
-            stage[i] = half(v[base + key * 64 + i]);
+        for (uint i = tid; i < BK * 16; i += 256)
+            stage_vec[i] = half4(((device const float4 *)(v + base + key * 64))[i]);
         float mx = -INFINITY;
         #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {

```

</details>

`python bench.py`  
→ run 5 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 70.30 ms, kernel 7.87 ms, speedup 8.93x

- `python -c 'from pathlib import Path; s=Path(\"queue/run4.py\").read_text().replace(\"_BK = 64\", \"_BK = 128\").replace(\"constant uint BK = 64\", \"constant...`
- `python check.py`  
→ OK: relative error 3.42e-04 (allowed 2e-03)
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -19,12 +19,11 @@
     uint tid [[thread_index_in_threadgroup]],
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
-    threadgroup half4 stage_vec[(BQ > BK ? BQ : BK) * 16];
-    threadgroup half *stage = (threadgroup half *)stage_vec;
+    threadgroup half stage[(BQ > BK ? BQ : BK) * 64];
     const uint base = group.y * 1024 * 64;
     const uint query = group.x * BQ;
-    for (uint i = tid; i < BQ * 16; i += 256)
-        stage_vec[i] = half4(((device const float4 *)(q + base + query * 64))[i]);
+    for (uint i = tid; i < BQ * 64; i += 256)
+        stage[i] = half(q[base + query * 64 + i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 acc[8];
@@ -39,23 +38,26 @@
     const uint row = query + sg * 8 + matrix_row;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for (uint key = 0; key < query + BQ; key += BK) {
-        for (uint i = tid; i < BK * 16; i += 256)
-            stage_vec[i] = half4(((device const float4 *)(k + base + key * 64))[i]);
+        for (uint i = tid; i < BK * 64; i += 256)
+            stage[i] = half(k[base + key * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 scores[BK / 8];
         #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
             scores[c] = simdgroup_float8x8(0.0f);
+        }
+        #pragma clang loop unroll(full)
+        for (uint d = 0; d < 8; ++d) {
             #pragma clang loop unroll(full)
-    for (uint d = 0; d < 8; ++d) {
+            for (uint c = 0; c < BK / 8; ++c) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km, stage + c * 8 * 64 + d * 8, 64, ulong2(0), true);
                 simdgroup_multiply_accumulate(scores[c], qm[d], km, scores[c]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < BK * 16; i += 256)
-            stage_vec[i] = half4(((device const float4 *)(v + base + key * 64))[i]);
+        for (uint i = tid; i < BK * 64; i += 256)
+            stage[i] = half(v[base + key * 64 + i]);
         float mx = -INFINITY;
         #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
@@ -91,8 +93,11 @@
     for (uint d = 0; d < 8; ++d) {
             acc[d].thread_elements()[0] *= alpha;
             acc[d].thread_elements()[1] *= alpha;
+        }
+        #pragma clang loop unroll(full)
+        for (uint c = 0; c < BK / 8; ++c) {
             #pragma clang loop unroll(full)
-        for (uint c = 0; c < BK / 8; ++c) {
+            for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 vm;
                 simdgroup_load(vm, stage + c * 8 * 64 + d * 8, 64);
                 simdgroup_multiply_accumulate(acc[d], pm[c], vm, acc[d]);

```

</details>

`python bench.py`  
→ run 6 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 68.25 ms, kernel 5.57 ms, speedup 12.25x

- `python -c 'from pathlib import Path`
- `python check.py`  
→ OK: relative error 3.69e-04 (allowed 2e-03)
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -19,11 +19,14 @@
     uint tid [[thread_index_in_threadgroup]],
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
-    threadgroup half stage[(BQ > BK ? BQ : BK) * 64];
+    threadgroup half4 stage_vec[(BQ > BK ? BQ : BK) * 16];
+    threadgroup half4 value_vec[BK * 16];
+    threadgroup half *stage = (threadgroup half *)stage_vec;
+    threadgroup half *values = (threadgroup half *)value_vec;
     const uint base = group.y * 1024 * 64;
     const uint query = group.x * BQ;
-    for (uint i = tid; i < BQ * 64; i += 256)
-        stage[i] = half(q[base + query * 64 + i]);
+    for (uint i = tid; i < BQ * 16; i += 256)
+        stage_vec[i] = half4(((device const float4 *)(q + base + query * 64))[i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 acc[8];
@@ -38,26 +41,22 @@
     const uint row = query + sg * 8 + matrix_row;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for (uint key = 0; key < query + BQ; key += BK) {
-        for (uint i = tid; i < BK * 64; i += 256)
-            stage[i] = half(k[base + key * 64 + i]);
+        for (uint i = tid; i < BK * 16; i += 256) {
+            stage_vec[i] = half4(((device const float4 *)(k + base + key * 64))[i]);
+            value_vec[i] = half4(((device const float4 *)(v + base + key * 64))[i]);
+        }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 scores[BK / 8];
         #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
             scores[c] = simdgroup_float8x8(0.0f);
-        }
-        #pragma clang loop unroll(full)
-        for (uint d = 0; d < 8; ++d) {
             #pragma clang loop unroll(full)
-            for (uint c = 0; c < BK / 8; ++c) {
+    for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km, stage + c * 8 * 64 + d * 8, 64, ulong2(0), true);
                 simdgroup_multiply_accumulate(scores[c], qm[d], km, scores[c]);
             }
         }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < BK * 64; i += 256)
-            stage[i] = half(v[base + key * 64 + i]);
         float mx = -INFINITY;
         #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
@@ -88,18 +87,14 @@
         sum += simd_shuffle_xor(sum, 8);
         denom = denom * alpha + sum;
         maxima = mx;
-        threadgroup_barrier(mem_flags::mem_threadgroup);
         #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d) {
             acc[d].thread_elements()[0] *= alpha;
             acc[d].thread_elements()[1] *= alpha;
-        }
-        #pragma clang loop unroll(full)
+            #pragma clang loop unroll(full)
         for (uint c = 0; c < BK / 8; ++c) {
-            #pragma clang loop unroll(full)
-            for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 vm;
-                simdgroup_load(vm, stage + c * 8 * 64 + d * 8, 64);
+                simdgroup_load(vm, values + c * 8 * 64 + d * 8, 64);
                 simdgroup_multiply_accumulate(acc[d], pm[c], vm, acc[d]);
             }
         }

```

</details>

`python bench.py`  


- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.14e-04 (allowed 2e-03)
### Bench run 11

`python bench.py`  
→ run 8 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 72.48 ms, kernel 4.13 ms, speedup 17.55x

- `cat kernel.py`
- `cp queue/run8.py kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.93e-04 (allowed 2e-03)
### Bench run 12

`python bench.py`  
→     _LIB.attention(q, k, v, out, threads=(1024 // _BQ * 256, 64, 1), group_size=(256, 1, 1))  
→     return out

- `python -c 'from pathlib import Path`
> The final variant took 4.10 ms, a 17.36× speedup, and passed the benchmark’s correctness check. I’m keeping it, updating LESSONS.md with all ten results, and checking it on fresh inputs.

- `python -c 'from pathlib import Path; Path(\"LESSONS.md\").write_text(\"The target is contiguous float32 causal attention: eight batches, eight heads, 1024 po...`
- `python -c 'import runpy`
- `python check.py`  
→ OK: relative error 3.63e-04 (allowed 2e-03)
- `cat LESSONS.md kernel.py`
> Measured **4.10 ms**, a **17.36× speedup**. All 11 fresh-input checks passed. LESSONS.md now contains 265 words covering results and lessons.
> 
> EDIT: kernel.py computes causal attention with a fused tiled Metal shader, vectorized loads, half precision operands, and float32 accumulation.

