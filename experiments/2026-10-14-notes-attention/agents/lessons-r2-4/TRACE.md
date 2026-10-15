# Agent `lessons-r2-4`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what is still true from the notes you were given. At most 300 words, in words, no code.

14 min, 101,439 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 8.289 ms | 7.81x |
| run2 | 11.857 ms | 5.74x |
| run3 | 4.376 ms | 15.83x |
| run4 | 4.397 ms | 15.39x |
| run5 | 5.858 ms | 11.61x |
| run6 | 4.951 ms | 15.2x |
| run7 | 4.809 ms | 14.37x |
| run8 | 4.238 ms | 15.92x |
| run9 | 4.254 ms | 16.79x |
| run10 | 3.884 ms | 17.61x |
| final | 3.687 ms | 16.43x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the task and earlier notes, optimize `kernel.py`, and use the benchmark budget to compare candidates. I’ll finish with a correctness check and updated notes.

- `pwd && rg --files -g '*' -g '"'!__pycache__/**`
### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -1,10 +1,108 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
+"""Fused causal attention for the task's M4 inputs."""
 import torch
 
 
+_lib = torch.mps.compile_shader(r"""
+#include <metal_stdlib>
+#include <metal_simdgroup_matrix>
+using namespace metal;
+
+kernel void attention(
+    device const float* Q [[buffer(0)]],
+    device const float* K [[buffer(1)]],
+    device const float* V [[buffer(2)]],
+    device float* O [[buffer(3)]],
+    uint3 tg [[threadgroup_position_in_grid]],
+    uint tid [[thread_index_in_threadgroup]],
+    uint sg [[simdgroup_index_in_threadgroup]],
+    uint lane [[thread_index_in_simdgroup]]) {
+    threadgroup half kv[8192];
+    threadgroup float scores[4096];
+    uint base = tg.y * 1024 * 64;
+    uint qb = tg.x * 64;
+    for (uint i = tid; i < 4096; i += 256)
+        kv[i] = half(Q[base + qb * 64 + i]);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    simdgroup_half8x8 qm[8];
+    for (uint d = 0; d < 8; ++d)
+        simdgroup_load(qm[d], kv + sg * 512 + d * 8, 64);
+    simdgroup_float8x8 acc[8];
+    for (uint d = 0; d < 8; ++d)
+        acc[d] = simdgroup_float8x8(0.0f);
+    uint row = (lane / 16) * 4 + (lane % 8) / 2;
+    uint col = (lane % 2) * 2 + (lane % 16 / 8) * 4;
+    float running_max = -INFINITY;
+    float running_sum = 0.0f;
+    for (uint kb = 0; kb <= qb; kb += 64) {
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for (uint i = tid; i < 4096; i += 256) {
+            kv[i] = half(K[base + kb * 64 + i]);
+            kv[4096 + i] = half(V[base + kb * 64 + i]);
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        simdgroup_float8x8 sm[8];
+        for (uint j = 0; j < 8; ++j)
+            sm[j] = simdgroup_float8x8(0.0f);
+        for (uint d = 0; d < 8; ++d) {
+            for (uint j = 0; j < 8; ++j) {
+                simdgroup_half8x8 km;
+                simdgroup_load(km, kv + j * 512 + d * 8, 64, ulong2(0), true);
+                simdgroup_multiply_accumulate(sm[j], qm[d], km, sm[j]);
+            }
+        }
+        for (uint j = 0; j < 8; ++j)
+            simdgroup_store(sm[j], scores + sg * 512 + j * 8, 64);
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        uint r = sg * 8 + row;
+        float local_max = -INFINITY;
+        for (uint j = col; j < 64; j += 8) {
+            float a = kb + j <= qb + r ? scores[r * 64 + j] * 0.125f : -INFINITY;
+            float b = kb + j + 1 <= qb + r ? scores[r * 64 + j + 1] * 0.125f : -INFINITY;
+            scores[r * 64 + j] = a;
+            scores[r * 64 + j + 1] = b;
+            local_max = max(local_max, max(a, b));
+        }
+        local_max = max(local_max, simd_shuffle_xor(local_max, 1));
+        local_max = max(local_max, simd_shuffle_xor(local_max, 8));
+        float new_max = max(running_max, local_max);
+        float alpha = fast::exp(running_max - new_max);
+        float local_sum = 0.0f;
+        for (uint j = col; j < 64; j += 8) {
+            float a = fast::exp(scores[r * 64 + j] - new_max);
+            float b = fast::exp(scores[r * 64 + j + 1] - new_max);
+            kv[r * 64 + j] = half(a);
+            kv[r * 64 + j + 1] = half(b);
+            local_sum += a + b;
+        }
+        local_sum += simd_shuffle_xor(local_sum, 1);
+        local_sum += simd_shuffle_xor(local_sum, 8);
+        running_sum = running_sum * alpha + local_sum;
+        running_max = new_max;
+        for (uint d = 0; d < 8; ++d) {
+            acc[d].thread_elements()[0] *= alpha;
+            acc[d].thread_elements()[1] *= alpha;
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for (uint j = 0; j < 8; ++j) {
+            simdgroup_half8x8 pm;
+            simdgroup_load(pm, kv + sg * 512 + j * 8, 64);
+            for (uint d = 0; d < 8; ++d) {
+                simdgroup_half8x8 vm;
+                simdgroup_load(vm, kv + 4096 + j * 512 + d * 8, 64);
+                simdgroup_multiply_accumulate(acc[d], pm, vm, acc[d]);
+            }
+        }
+    }
+    for (uint d = 0; d < 8; ++d) {
+        acc[d].thread_elements()[0] /= running_sum;
+        acc[d].thread_elements()[1] /= running_sum;
+        simdgroup_store(acc[d], O + base + (qb + sg * 8) * 64 + d * 8, 64);
+    }
+}
+""")
+
+
 def kernel(q, k, v):
-    T = q.size(-2)
-    s = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
-    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
-    e = (s - s.max(-1, keepdim=True).values).exp()
-    return (e / e.sum(-1, keepdim=True)) @ v
+    out = torch.empty_like(q)
+    _lib.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
+    return out

```

</details>

`cat kernel.py reference.py bench.py check.py LESSONS.md`  
→ Fully fused online attention was slower: query/key tile sizes 32/32, 64/32, and 64/64 took 10.45, 10.40, and 11.25 milliseconds. Reusing shared memory and retaining probabilities in registers with 128/64 tiles took 10.31. One trial failed compilation because Metal max is not a constant expression for array sizes; a ternary expression works.  
→ All timed successful variants passed correctness. GPU contention changes reference times, so compare absolute kernel times too.

- `python -c 'import torch; print(torch.__version__); print(torch.mps.compile_shader.__doc__)`
- `python - <<'PY` (exited 1 in 0ms:)
- `python -c 'import torch`
> The notes point to a fused Metal kernel as the strongest approach: half precision matrix operands with float32 softmax and accumulation. I also checked Metal’s matrix element layout so I can keep the running output in registers.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.51e-04 (allowed 2e-03)
### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -17,7 +17,6 @@
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
     threadgroup half kv[8192];
-    threadgroup float scores[4096];
     uint base = tg.y * 1024 * 64;
     uint qb = tg.x * 64;
     for (uint i = tid; i < 4096; i += 256)
@@ -50,16 +49,13 @@
                 simdgroup_multiply_accumulate(sm[j], qm[d], km, sm[j]);
             }
         }
-        for (uint j = 0; j < 8; ++j)
-            simdgroup_store(sm[j], scores + sg * 512 + j * 8, 64);
-        threadgroup_barrier(mem_flags::mem_threadgroup);
         uint r = sg * 8 + row;
         float local_max = -INFINITY;
-        for (uint j = col; j < 64; j += 8) {
-            float a = kb + j <= qb + r ? scores[r * 64 + j] * 0.125f : -INFINITY;
-            float b = kb + j + 1 <= qb + r ? scores[r * 64 + j + 1] * 0.125f : -INFINITY;
-            scores[r * 64 + j] = a;
-            scores[r * 64 + j + 1] = b;
+        for (uint j = 0; j < 8; ++j) {
+            float a = kb + j * 8 + col <= qb + r ? sm[j].thread_elements()[0] * 0.125f : -INFINITY;
+            float b = kb + j * 8 + col + 1 <= qb + r ? sm[j].thread_elements()[1] * 0.125f : -INFINITY;
+            sm[j].thread_elements()[0] = a;
+            sm[j].thread_elements()[1] = b;
             local_max = max(local_max, max(a, b));
         }
         local_max = max(local_max, simd_shuffle_xor(local_max, 1));
@@ -67,11 +63,12 @@
         float new_max = max(running_max, local_max);
         float alpha = fast::exp(running_max - new_max);
         float local_sum = 0.0f;
-        for (uint j = col; j < 64; j += 8) {
-            float a = fast::exp(scores[r * 64 + j] - new_max);
-            float b = fast::exp(scores[r * 64 + j + 1] - new_max);
-            kv[r * 64 + j] = half(a);
-            kv[r * 64 + j + 1] = half(b);
+        simdgroup_half8x8 pm[8];
+        for (uint j = 0; j < 8; ++j) {
+            float a = fast::exp(sm[j].thread_elements()[0] - new_max);
+            float b = fast::exp(sm[j].thread_elements()[1] - new_max);
+            pm[j].thread_elements()[0] = half(a);
+            pm[j].thread_elements()[1] = half(b);
             local_sum += a + b;
         }
         local_sum += simd_shuffle_xor(local_sum, 1);
@@ -82,14 +79,11 @@
             acc[d].thread_elements()[0] *= alpha;
             acc[d].thread_elements()[1] *= alpha;
         }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
         for (uint j = 0; j < 8; ++j) {
-            simdgroup_half8x8 pm;
-            simdgroup_load(pm, kv + sg * 512 + j * 8, 64);
             for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 vm;
                 simdgroup_load(vm, kv + 4096 + j * 512 + d * 8, 64);
-                simdgroup_multiply_accumulate(acc[d], pm, vm, acc[d]);
+                simdgroup_multiply_accumulate(acc[d], pm[j], vm, acc[d]);
             }
         }
     }

```

</details>

`python bench.py`  
→ run 1 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 64.70 ms, kernel 8.29 ms, speedup 7.81x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.21e-04 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -23,9 +23,11 @@
         kv[i] = half(Q[base + qb * 64 + i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
+    #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d)
         simdgroup_load(qm[d], kv + sg * 512 + d * 8, 64);
     simdgroup_float8x8 acc[8];
+    #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d)
         acc[d] = simdgroup_float8x8(0.0f);
     uint row = (lane / 16) * 4 + (lane % 8) / 2;
@@ -40,10 +42,13 @@
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 sm[8];
+        #pragma clang loop unroll(full)
         for (uint j = 0; j < 8; ++j)
             sm[j] = simdgroup_float8x8(0.0f);
-        for (uint d = 0; d < 8; ++d) {
-            for (uint j = 0; j < 8; ++j) {
+        #pragma clang loop unroll(full)
+    for (uint d = 0; d < 8; ++d) {
+            #pragma clang loop unroll(full)
+        for (uint j = 0; j < 8; ++j) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km, kv + j * 512 + d * 8, 64, ulong2(0), true);
                 simdgroup_multiply_accumulate(sm[j], qm[d], km, sm[j]);
@@ -51,6 +56,7 @@
         }
         uint r = sg * 8 + row;
         float local_max = -INFINITY;
+        #pragma clang loop unroll(full)
         for (uint j = 0; j < 8; ++j) {
             float a = kb + j * 8 + col <= qb + r ? sm[j].thread_elements()[0] * 0.125f : -INFINITY;
             float b = kb + j * 8 + col + 1 <= qb + r ? sm[j].thread_elements()[1] * 0.125f : -INFINITY;
@@ -64,6 +70,7 @@
         float alpha = fast::exp(running_max - new_max);
         float local_sum = 0.0f;
         simdgroup_half8x8 pm[8];
+        #pragma clang loop unroll(full)
         for (uint j = 0; j < 8; ++j) {
             float a = fast::exp(sm[j].thread_elements()[0] - new_max);
             float b = fast::exp(sm[j].thread_elements()[1] - new_max);
@@ -75,18 +82,22 @@
         local_sum += simd_shuffle_xor(local_sum, 8);
         running_sum = running_sum * alpha + local_sum;
         running_max = new_max;
-        for (uint d = 0; d < 8; ++d) {
+        #pragma clang loop unroll(full)
+    for (uint d = 0; d < 8; ++d) {
             acc[d].thread_elements()[0] *= alpha;
             acc[d].thread_elements()[1] *= alpha;
         }
+        #pragma clang loop unroll(full)
         for (uint j = 0; j < 8; ++j) {
-            for (uint d = 0; d < 8; ++d) {
+            #pragma clang loop unroll(full)
+    for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 vm;
                 simdgroup_load(vm, kv + 4096 + j * 512 + d * 8, 64);
                 simdgroup_multiply_accumulate(acc[d], pm[j], vm, acc[d]);
             }
         }
     }
+    #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d) {
         acc[d].thread_elements()[0] /= running_sum;
         acc[d].thread_elements()[1] /= running_sum;

```

</details>

`python bench.py`  
→ run 2 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 68.01 ms, kernel 11.86 ms, speedup 5.74x

- `cat kernel.py`
- `python -c 'from pathlib import Path`
- `python check.py`  
→ OK: relative error 3.29e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -6,6 +6,10 @@
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
+#define BQ 64
+#define BK 32
+#define NT (BQ * 4)
+#define NK (BK / 8)
 
 kernel void attention(
     device const float* Q [[buffer(0)]],
@@ -16,10 +20,10 @@
     uint tid [[thread_index_in_threadgroup]],
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
-    threadgroup half kv[8192];
+    threadgroup half kv[(BQ > 2 * BK ? BQ : 2 * BK) * 64];
     uint base = tg.y * 1024 * 64;
-    uint qb = tg.x * 64;
-    for (uint i = tid; i < 4096; i += 256)
+    uint qb = tg.x * BQ;
+    for (uint i = tid; i < BQ * 64; i += NT)
         kv[i] = half(Q[base + qb * 64 + i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
@@ -34,21 +38,21 @@
     uint col = (lane % 2) * 2 + (lane % 16 / 8) * 4;
     float running_max = -INFINITY;
     float running_sum = 0.0f;
-    for (uint kb = 0; kb <= qb; kb += 64) {
+    for (uint kb = 0; kb < qb + BQ; kb += BK) {
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < 4096; i += 256) {
+        for (uint i = tid; i < BK * 64; i += NT) {
             kv[i] = half(K[base + kb * 64 + i]);
-            kv[4096 + i] = half(V[base + kb * 64 + i]);
+            kv[BK * 64 + i] = half(V[base + kb * 64 + i]);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 sm[8];
+        simdgroup_float8x8 sm[NK];
         #pragma clang loop unroll(full)
-        for (uint j = 0; j < 8; ++j)
+        for (uint j = 0; j < NK; ++j)
             sm[j] = simdgroup_float8x8(0.0f);
         #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d) {
             #pragma clang loop unroll(full)
-        for (uint j = 0; j < 8; ++j) {
+        for (uint j = 0; j < NK; ++j) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km, kv + j * 512 + d * 8, 64, ulong2(0), true);
                 simdgroup_multiply_accumulate(sm[j], qm[d], km, sm[j]);
@@ -57,7 +61,7 @@
         uint r = sg * 8 + row;
         float local_max = -INFINITY;
         #pragma clang loop unroll(full)
-        for (uint j = 0; j < 8; ++j) {
+        for (uint j = 0; j < NK; ++j) {
             float a = kb + j * 8 + col <= qb + r ? sm[j].thread_elements()[0] * 0.125f : -INFINITY;
             float b = kb + j * 8 + col + 1 <= qb + r ? sm[j].thread_elements()[1] * 0.125f : -INFINITY;
             sm[j].thread_elements()[0] = a;
@@ -69,9 +73,9 @@
         float new_max = max(running_max, local_max);
         float alpha = fast::exp(running_max - new_max);
         float local_sum = 0.0f;
-        simdgroup_half8x8 pm[8];
+        simdgroup_half8x8 pm[NK];
         #pragma clang loop unroll(full)
-        for (uint j = 0; j < 8; ++j) {
+        for (uint j = 0; j < NK; ++j) {
             float a = fast::exp(sm[j].thread_elements()[0] - new_max);
             float b = fast::exp(sm[j].thread_elements()[1] - new_max);
             pm[j].thread_elements()[0] = half(a);
@@ -88,11 +92,11 @@
             acc[d].thread_elements()[1] *= alpha;
         }
         #pragma clang loop unroll(full)
-        for (uint j = 0; j < 8; ++j) {
+        for (uint j = 0; j < NK; ++j) {
             #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 vm;
-                simdgroup_load(vm, kv + 4096 + j * 512 + d * 8, 64);
+                simdgroup_load(vm, kv + BK * 64 + j * 512 + d * 8, 64);
                 simdgroup_multiply_accumulate(acc[d], pm[j], vm, acc[d]);
             }
         }

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 69.29 ms, kernel 4.38 ms, speedup 15.83x

> The best candidate so far takes 4.38 ms, a 15.83× speedup in that run, and passes correctness comfortably. Explicitly unrolling the small matrix loops made the biggest difference. I’m testing tile sizes next.

- `python -c 'from pathlib import Path`
- `python check.py`  
→ OK: relative error 3.97e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -6,8 +6,8 @@
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-#define BQ 64
-#define BK 32
+#define BQ 32
+#define BK 64
 #define NT (BQ * 4)
 #define NK (BK / 8)
 
@@ -113,5 +113,5 @@
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _lib.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
+    _lib.attention(q, k, v, out, threads=(32 * 128, 64, 1), group_size=(128, 1, 1))
     return out

```

</details>

`python bench.py`  


- `python check.py`  
→ OK: relative error 3.08e-04 (allowed 2e-03)
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -6,7 +6,7 @@
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-#define BQ 32
+#define BQ 128
 #define BK 64
 #define NT (BQ * 4)
 #define NK (BK / 8)
@@ -113,5 +113,5 @@
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _lib.attention(q, k, v, out, threads=(32 * 128, 64, 1), group_size=(128, 1, 1))
+    _lib.attention(q, k, v, out, threads=(8 * 512, 64, 1), group_size=(512, 1, 1))
     return out

```

</details>

`python bench.py`  


- `python check.py`  
→ OK: relative error 3.39e-04 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -6,24 +6,31 @@
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-#define BQ 128
-#define BK 64
-#define NT (BQ * 4)
-#define NK (BK / 8)
+kernel void convert_inputs(
+    device const float4* Q [[buffer(0)]],
+    device const float4* K [[buffer(1)]],
+    device const float4* V [[buffer(2)]],
+    device half4* H [[buffer(3)]],
+    uint i [[thread_position_in_grid]]) {
+    H[i] = half4(Q[i]);
+    H[1048576 + i] = half4(K[i]);
+    H[2097152 + i] = half4(V[i]);
+}
+
 
 kernel void attention(
-    device const float* Q [[buffer(0)]],
-    device const float* K [[buffer(1)]],
-    device const float* V [[buffer(2)]],
+    device const half* Q [[buffer(0)]],
+    device const half* K [[buffer(1)]],
+    device const half* V [[buffer(2)]],
     device float* O [[buffer(3)]],
     uint3 tg [[threadgroup_position_in_grid]],
     uint tid [[thread_index_in_threadgroup]],
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
-    threadgroup half kv[(BQ > 2 * BK ? BQ : 2 * BK) * 64];
+    threadgroup half kv[8192];
     uint base = tg.y * 1024 * 64;
-    uint qb = tg.x * BQ;
-    for (uint i = tid; i < BQ * 64; i += NT)
+    uint qb = tg.x * 64;
+    for (uint i = tid; i < 4096; i += 256)
         kv[i] = half(Q[base + qb * 64 + i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
@@ -38,21 +45,21 @@
     uint col = (lane % 2) * 2 + (lane % 16 / 8) * 4;
     float running_max = -INFINITY;
     float running_sum = 0.0f;
-    for (uint kb = 0; kb < qb + BQ; kb += BK) {
+    for (uint kb = 0; kb <= qb; kb += 64) {
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < BK * 64; i += NT) {
-            kv[i] = half(K[base + kb * 64 + i]);
-            kv[BK * 64 + i] = half(V[base + kb * 64 + i]);
+        for (uint i = tid; i < 4096; i += 256) {
+            kv[i] = half(K[4194304 + base + kb * 64 + i]);
+            kv[4096 + i] = half(V[8388608 + base + kb * 64 + i]);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 sm[NK];
+        simdgroup_float8x8 sm[8];
         #pragma clang loop unroll(full)
-        for (uint j = 0; j < NK; ++j)
+        for (uint j = 0; j < 8; ++j)
             sm[j] = simdgroup_float8x8(0.0f);
         #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d) {
             #pragma clang loop unroll(full)
-        for (uint j = 0; j < NK; ++j) {
+        for (uint j = 0; j < 8; ++j) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km, kv + j * 512 + d * 8, 64, ulong2(0), true);
                 simdgroup_multiply_accumulate(sm[j], qm[d], km, sm[j]);
@@ -61,7 +68,7 @@
         uint r = sg * 8 + row;
         float local_max = -INFINITY;
         #pragma clang loop unroll(full)
-        for (uint j = 0; j < NK; ++j) {
+        for (uint j = 0; j < 8; ++j) {
             float a = kb + j * 8 + col <= qb + r ? sm[j].thread_elements()[0] * 0.125f : -INFINITY;
             float b = kb + j * 8 + col + 1 <= qb + r ? sm[j].thread_elements()[1] * 0.125f : -INFINITY;
             sm[j].thread_elements()[0] = a;
@@ -73,9 +80,9 @@
         float new_max = max(running_max, local_max);
         float alpha = fast::exp(running_max - new_max);
         float local_sum = 0.0f;
-        simdgroup_half8x8 pm[NK];
+        simdgroup_half8x8 pm[8];
         #pragma clang loop unroll(full)
-        for (uint j = 0; j < NK; ++j) {
+        for (uint j = 0; j < 8; ++j) {
             float a = fast::exp(sm[j].thread_elements()[0] - new_max);
             float b = fast::exp(sm[j].thread_elements()[1] - new_max);
             pm[j].thread_elements()[0] = half(a);
@@ -92,11 +99,11 @@
             acc[d].thread_elements()[1] *= alpha;
         }
         #pragma clang loop unroll(full)
-        for (uint j = 0; j < NK; ++j) {
+        for (uint j = 0; j < 8; ++j) {
             #pragma clang loop unroll(full)
     for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 vm;
-                simdgroup_load(vm, kv + BK * 64 + j * 512 + d * 8, 64);
+                simdgroup_load(vm, kv + 4096 + j * 512 + d * 8, 64);
                 simdgroup_multiply_accumulate(acc[d], pm[j], vm, acc[d]);
             }
         }
@@ -113,5 +120,7 @@
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _lib.attention(q, k, v, out, threads=(8 * 512, 64, 1), group_size=(512, 1, 1))
+    packed = torch.empty((3, q.numel()), device=q.device, dtype=torch.float16)
+    _lib.convert_inputs(q, k, v, packed, threads=1048576, group_size=256)
+    _lib.attention(packed, packed, packed, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
     return out

```

</details>

`python bench.py`  


- `python check.py`  
→ OK: relative error 3.20e-04 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -6,22 +6,11 @@
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-kernel void convert_inputs(
-    device const float4* Q [[buffer(0)]],
-    device const float4* K [[buffer(1)]],
-    device const float4* V [[buffer(2)]],
-    device half4* H [[buffer(3)]],
-    uint i [[thread_position_in_grid]]) {
-    H[i] = half4(Q[i]);
-    H[1048576 + i] = half4(K[i]);
-    H[2097152 + i] = half4(V[i]);
-}
-
 
 kernel void attention(
-    device const half* Q [[buffer(0)]],
-    device const half* K [[buffer(1)]],
-    device const half* V [[buffer(2)]],
+    device const float* Q [[buffer(0)]],
+    device const float* K [[buffer(1)]],
+    device const float* V [[buffer(2)]],
     device float* O [[buffer(3)]],
     uint3 tg [[threadgroup_position_in_grid]],
     uint tid [[thread_index_in_threadgroup]],
@@ -48,8 +37,8 @@
     for (uint kb = 0; kb <= qb; kb += 64) {
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for (uint i = tid; i < 4096; i += 256) {
-            kv[i] = half(K[4194304 + base + kb * 64 + i]);
-            kv[4096 + i] = half(V[8388608 + base + kb * 64 + i]);
+            kv[i] = half(K[base + kb * 64 + i]);
+            kv[4096 + i] = half(V[base + kb * 64 + i]);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 sm[8];
@@ -99,12 +88,16 @@
             acc[d].thread_elements()[1] *= alpha;
         }
         #pragma clang loop unroll(full)
-        for (uint j = 0; j < 8; ++j) {
+        for (uint j = 0; j < 8; j += 2) {
             #pragma clang loop unroll(full)
-    for (uint d = 0; d < 8; ++d) {
-                simdgroup_half8x8 vm;
+            for (uint d = 0; d < 8; ++d) {
+                simdgroup_half8x8 vm, part(half(0));
                 simdgroup_load(vm, kv + 4096 + j * 512 + d * 8, 64);
-                simdgroup_multiply_accumulate(acc[d], pm[j], vm, acc[d]);
+                simdgroup_multiply_accumulate(part, pm[j], vm, part);
+                simdgroup_load(vm, kv + 4096 + (j + 1) * 512 + d * 8, 64);
+                simdgroup_multiply_accumulate(part, pm[j + 1], vm, part);
+                acc[d].thread_elements()[0] += float(part.thread_elements()[0]);
+                acc[d].thread_elements()[1] += float(part.thread_elements()[1]);
             }
         }
     }
@@ -120,7 +113,5 @@
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    packed = torch.empty((3, q.numel()), device=q.device, dtype=torch.float16)
-    _lib.convert_inputs(q, k, v, packed, threads=1048576, group_size=256)
-    _lib.attention(packed, packed, packed, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
+    _lib.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
     return out

```

</details>

`python bench.py`  


- `python check.py`  
→ OK: relative error 8.34e-04 (allowed 2e-03)
- `python -c 'import torch; from kernel import kernel; from reference import inputs,reference,TOL; errors=[]`
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -8,19 +8,23 @@
 using namespace metal;
 
 kernel void attention(
-    device const float* Q [[buffer(0)]],
-    device const float* K [[buffer(1)]],
-    device const float* V [[buffer(2)]],
+    device const float4* Q [[buffer(0)]],
+    device const float4* K [[buffer(1)]],
+    device const float4* V [[buffer(2)]],
     device float* O [[buffer(3)]],
     uint3 tg [[threadgroup_position_in_grid]],
     uint tid [[thread_index_in_threadgroup]],
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
     threadgroup half kv[8192];
+    threadgroup half4* kv4 = (threadgroup half4*)kv;
     uint base = tg.y * 1024 * 64;
     uint qb = tg.x * 64;
-    for (uint i = tid; i < 4096; i += 256)
-        kv[i] = half(Q[base + qb * 64 + i]);
+    #pragma clang loop unroll(full)
+    for (uint step = 0; step < 4; ++step) {
+        uint i = tid + step * 256;
+        kv4[i] = half4(Q[base / 4 + qb * 16 + i]);
+    }
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
     #pragma clang loop unroll(full)
@@ -36,9 +40,11 @@
     float running_sum = 0.0f;
     for (uint kb = 0; kb <= qb; kb += 64) {
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < 4096; i += 256) {
-            kv[i] = half(K[base + kb * 64 + i]);
-            kv[4096 + i] = half(V[base + kb * 64 + i]);
+        #pragma clang loop unroll(full)
+        for (uint step = 0; step < 4; ++step) {
+            uint i = tid + step * 256;
+            kv4[i] = half4(K[base / 4 + kb * 16 + i]);
+            kv4[1024 + i] = half4(V[base / 4 + kb * 16 + i]);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 sm[8];

```

</details>

`python bench.py`  
→ run 8 of 10 submitted.  
→ attention: correct (relative error 1.2e-03). reference 67.46 ms, kernel 4.24 ms, speedup 15.92x

- `python -c 'from pathlib import Path`
- `python check.py`  
→ OK: relative error 7.66e-04 (allowed 2e-03)
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -61,11 +61,20 @@
             }
         }
         uint r = sg * 8 + row;
+        if (kb == qb) {
+            #pragma clang loop unroll(full)
+            for (uint j = 0; j < 8; ++j) {
+                if (j * 8 + col > r)
+                    sm[j].thread_elements()[0] = -INFINITY;
+                if (j * 8 + col + 1 > r)
+                    sm[j].thread_elements()[1] = -INFINITY;
+            }
+        }
         float local_max = -INFINITY;
         #pragma clang loop unroll(full)
         for (uint j = 0; j < 8; ++j) {
-            float a = kb + j * 8 + col <= qb + r ? sm[j].thread_elements()[0] * 0.125f : -INFINITY;
-            float b = kb + j * 8 + col + 1 <= qb + r ? sm[j].thread_elements()[1] * 0.125f : -INFINITY;
+            float a = sm[j].thread_elements()[0] * 0.125f;
+            float b = sm[j].thread_elements()[1] * 0.125f;
             sm[j].thread_elements()[0] = a;
             sm[j].thread_elements()[1] = b;
             local_max = max(local_max, max(a, b));

```

</details>

`python bench.py`  
→ run 9 of 10 submitted.  
→ attention: correct (relative error 1.2e-03). reference 71.42 ms, kernel 4.25 ms, speedup 16.79x

> Summing value products in 16-key half precision blocks, then adding them into float32 accumulators, reached 4.24 ms. Vectorized input loads took a similar 4.25 ms. For the last benchmark, I’ll apply per-element causal masking only on the diagonal tile.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 7.79e-04 (allowed 2e-03)
### Bench run 11

`python bench.py`  
→     _lib.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))  
→     return out

- `python -c 'import torch; from kernel import kernel; from reference import inputs,reference,TOL; errors=[]`
> The final change brought the kernel to 3.88 ms, a 17.60× benchmark speedup. Twenty more fresh-input checks passed, with maximum relative error 0.00114 against the 0.002 tolerance. I’m recording the results and running the final correctness check.

- edited `solution.py`
- `python -c 'from pathlib import Path`
- `python check.py`  
→ OK: relative error 6.12e-04 (allowed 2e-03)
> Measured **3.88 ms**, a **17.60× speedup**. Final correctness check passed: relative error **0.000612**, below **0.002**.
> 
> Updated LESSONS.md with all ten benchmark results and practical guidance.
> 
> EDIT: kernel.py now computes fused causal attention using a tiled Metal shader, register accumulators, and stable online softmax.

