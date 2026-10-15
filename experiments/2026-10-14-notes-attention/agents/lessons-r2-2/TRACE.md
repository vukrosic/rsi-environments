# Agent `lessons-r2-2`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what is still true from the notes you were given. At most 300 words, in words, no code.

11 min, 89,717 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 9.153 ms | 6.98x |
| run2 | 11.841 ms | 6.02x |
| run3 | 11.641 ms | 5.76x |
| run4 | 13.353 ms | 5.11x |
| run5 | 9.927 ms | 6.85x |
| run6 | 9.68 ms | 6.97x |
| run7 | 8.783 ms | 7.71x |
| run8 | 8.495 ms | 8.2x |
| run9 | 6.091 ms | 10.99x |
| run10 | 9.553 ms | 7.21x |
| final | 6.287 ms | 10.83x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the reference and earlier measurements, then optimize the attention kernel within the ten-run limit and check the final result.

### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -1,10 +1,103 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
+"""Fused causal attention for the fixed MPS workload."""
 import torch
 
+_SOURCE = r'''
+#include <metal_stdlib>
+#include <metal_simdgroup_matrix>
+using namespace metal;
+
+kernel void attention(
+    device const float* Q [[buffer(0)]],
+    device const float* K [[buffer(1)]],
+    device const float* V [[buffer(2)]],
+    device float* O [[buffer(3)]],
+    uint3 gid [[threadgroup_position_in_grid]],
+    uint tid [[thread_index_in_threadgroup]],
+    uint sg [[simdgroup_index_in_threadgroup]],
+    uint lane [[thread_index_in_simdgroup]]) {
+    threadgroup half kv[64 * 64];
+    threadgroup float scores[64 * 64];
+    threadgroup half probs[64 * 64];
+    const uint base = gid.y * 1024 * 64;
+    const uint qb = gid.x * 64;
+    for (uint i = tid; i < 4096; i += 256)
+        kv[i] = half(Q[base + qb * 64 + i]);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    simdgroup_half8x8 qm[8];
+    simdgroup_float8x8 out[8];
+    for (uint d = 0; d < 8; ++d) {
+        simdgroup_load(qm[d], kv + sg * 8 * 64 + d * 8, 64);
+        out[d] = simdgroup_float8x8(0.0f);
+    }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    float mx = -INFINITY;
+    float den = 0.0f;
+    const uint lane_col = lane % 2 + ((lane / 8) % 2) * 2;
+    const uint row = sg * 8 + (lane % 8) / 2 + (lane / 16) * 4;
+    for (uint kb = 0; kb <= qb; kb += 64) {
+        for (uint i = tid; i < 4096; i += 256)
+            kv[i] = half(K[base + kb * 64 + i]);
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for (uint j = 0; j < 8; ++j) {
+            simdgroup_float8x8 acc(0.0f);
+            for (uint d = 0; d < 8; ++d) {
+                simdgroup_half8x8 km;
+                simdgroup_load(km, kv + j * 8 * 64 + d * 8, 64, ulong2(0), true);
+                simdgroup_multiply_accumulate(acc, qm[d], km, acc);
+            }
+            simdgroup_store(acc, scores + sg * 8 * 64 + j * 8, 64);
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        float vals[16];
+        float tilemax = -INFINITY;
+        for (uint j = 0; j < 16; ++j) {
+            uint col = lane_col + j * 4;
+            float z = kb + col <= qb + row ? scores[row * 64 + col] * 0.125f : -INFINITY;
+            vals[j] = z;
+            tilemax = max(tilemax, z);
+        }
+        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
+        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
+        float newmx = max(mx, tilemax);
+        float alpha = fast::exp(mx - newmx);
+        float tilesum = 0.0f;
+        for (uint j = 0; j < 16; ++j) {
+            float p = fast::exp(vals[j] - newmx);
+            probs[row * 64 + lane_col + j * 4] = half(p);
+            tilesum += p;
+        }
+        tilesum += simd_shuffle_xor(tilesum, 1);
+        tilesum += simd_shuffle_xor(tilesum, 8);
+        den = den * alpha + tilesum;
+        mx = newmx;
+        for (uint d = 0; d < 8; ++d) {
+            out[d].thread_elements()[0] *= alpha;
+            out[d].thread_elements()[1] *= alpha;
+        }
+        for (uint i = tid; i < 4096; i += 256)
+            kv[i] = half(V[base + kb * 64 + i]);
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for (uint j = 0; j < 8; ++j) {
+            simdgroup_half8x8 pm;
+            simdgroup_load(pm, probs + sg * 8 * 64 + j * 8, 64);
+            for (uint d = 0; d < 8; ++d) {
+                simdgroup_half8x8 vm;
+                simdgroup_load(vm, kv + j * 8 * 64 + d * 8, 64);
+                simdgroup_multiply_accumulate(out[d], pm, vm, out[d]);
+            }
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+    }
+    for (uint d = 0; d < 8; ++d) {
+        out[d].thread_elements()[0] /= den;
+        out[d].thread_elements()[1] /= den;
+        simdgroup_store(out[d], O + base + (qb + sg * 8) * 64 + d * 8, 64);
+    }
+}
+'''
+_LIB = torch.mps.compile_shader(_SOURCE)
 
 def kernel(q, k, v):
-    T = q.size(-2)
-    s = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
-    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
-    e = (s - s.max(-1, keepdim=True).values).exp()
-    return (e / e.sum(-1, keepdim=True)) @ v
+    out = torch.empty_like(q)
+    _LIB.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
+    return out

```

</details>

`pwd; rg --files -g 'AGENTS.md' -g 'LESSONS.md' -g 'kernel.py' -g 'reference.py' -g 'bench.py' -g 'check.py`  
→ check.py  
→ kernel.py

### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -16,55 +16,53 @@
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
     threadgroup half kv[64 * 64];
-    threadgroup float scores[64 * 64];
-    threadgroup half probs[64 * 64];
     const uint base = gid.y * 1024 * 64;
     const uint qb = gid.x * 64;
-    for (uint i = tid; i < 4096; i += 256)
-        kv[i] = half(Q[base + qb * 64 + i]);
-    threadgroup_barrier(mem_flags::mem_threadgroup);
+    const uint lane_col = (lane % 2) * 2 + ((lane / 8) % 2) * 4;
+    const uint row = sg * 8 + (lane % 8) / 2 + (lane / 16) * 4;
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 out[8];
     for (uint d = 0; d < 8; ++d) {
-        simdgroup_load(qm[d], kv + sg * 8 * 64 + d * 8, 64);
+        qm[d].thread_elements()[0] = half(Q[base + (qb + row) * 64 + d * 8 + lane_col]);
+        qm[d].thread_elements()[1] = half(Q[base + (qb + row) * 64 + d * 8 + lane_col + 1]);
         out[d] = simdgroup_float8x8(0.0f);
     }
-    threadgroup_barrier(mem_flags::mem_threadgroup);
     float mx = -INFINITY;
     float den = 0.0f;
-    const uint lane_col = lane % 2 + ((lane / 8) % 2) * 2;
-    const uint row = sg * 8 + (lane % 8) / 2 + (lane / 16) * 4;
     for (uint kb = 0; kb <= qb; kb += 64) {
         for (uint i = tid; i < 4096; i += 256)
             kv[i] = half(K[base + kb * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
+        simdgroup_float8x8 score[8];
         for (uint j = 0; j < 8; ++j) {
-            simdgroup_float8x8 acc(0.0f);
+            score[j] = simdgroup_float8x8(0.0f);
             for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km, kv + j * 8 * 64 + d * 8, 64, ulong2(0), true);
-                simdgroup_multiply_accumulate(acc, qm[d], km, acc);
+                simdgroup_multiply_accumulate(score[j], qm[d], km, score[j]);
             }
-            simdgroup_store(acc, scores + sg * 8 * 64 + j * 8, 64);
         }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        float vals[16];
         float tilemax = -INFINITY;
-        for (uint j = 0; j < 16; ++j) {
-            uint col = lane_col + j * 4;
-            float z = kb + col <= qb + row ? scores[row * 64 + col] * 0.125f : -INFINITY;
-            vals[j] = z;
-            tilemax = max(tilemax, z);
+        for (uint j = 0; j < 8; ++j) {
+            for (uint e = 0; e < 2; ++e) {
+                uint col = lane_col + j * 8 + e;
+                float z = kb + col <= qb + row ? score[j].thread_elements()[e] * 0.125f : -INFINITY;
+                score[j].thread_elements()[e] = z;
+                tilemax = max(tilemax, z);
+            }
         }
         tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
         tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
         float newmx = max(mx, tilemax);
         float alpha = fast::exp(mx - newmx);
         float tilesum = 0.0f;
-        for (uint j = 0; j < 16; ++j) {
-            float p = fast::exp(vals[j] - newmx);
-            probs[row * 64 + lane_col + j * 4] = half(p);
-            tilesum += p;
+        simdgroup_half8x8 prob[8];
+        for (uint j = 0; j < 8; ++j) {
+            for (uint e = 0; e < 2; ++e) {
+                float p = fast::exp(score[j].thread_elements()[e] - newmx);
+                prob[j].thread_elements()[e] = half(p);
+                tilesum += p;
+            }
         }
         tilesum += simd_shuffle_xor(tilesum, 1);
         tilesum += simd_shuffle_xor(tilesum, 8);
@@ -74,16 +72,15 @@
             out[d].thread_elements()[0] *= alpha;
             out[d].thread_elements()[1] *= alpha;
         }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
         for (uint i = tid; i < 4096; i += 256)
             kv[i] = half(V[base + kb * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for (uint j = 0; j < 8; ++j) {
-            simdgroup_half8x8 pm;
-            simdgroup_load(pm, probs + sg * 8 * 64 + j * 8, 64);
             for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 vm;
                 simdgroup_load(vm, kv + j * 8 * 64 + d * 8, 64);
-                simdgroup_multiply_accumulate(out[d], pm, vm, out[d]);
+                simdgroup_multiply_accumulate(out[d], prob[j], vm, out[d]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);

```

</details>

`cat LESSONS.md kernel.py reference.py bench.py check.py`  
→     err = (got.float() - want).abs().max().item() / want.abs().max().item()  
→ print(f"{'OK' if got.shape == want.shape and err <= TOL else 'WRONG'}: relative error {err:.2e} (allowed {TOL:.0e})")

- `rg --files -g 'AGENTS.md' -g '"'!queue/**'"' -g '"'!results/**'"'; python -c 'import torch; print(torch.__version__); print(torch.mps.compile_shader.__doc__)`
> The notes point to a fused Metal shader as the fastest approach: half-precision matrix operands with float32 accumulation and softmax. I’ll start with the successful 64-by-64 tile design, then use the remaining runs to tune it.

- `cat > kernel.py <<'PY` (exited 1 in 641ms:)
- edited `solution.py`
- `python check.py`  
→ WRONG: relative error 6.08e-01 (allowed 2e-03)
- edited `solution.py`
- `python -c 'import torch, kernel; t=torch.empty((8,8), device=\"mps\"); kernel._LIB.layout(t, threads=32, group_size=32); print(t.cpu())`
- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text(); a=s.index(\"kernel void layout\"); b=s.index(\"kernel void attention\",a); s=s[:...`
- `python check.py`  
→ OK: relative error 4.60e-04 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -22,6 +22,7 @@
     const uint row = sg * 8 + (lane % 8) / 2 + (lane / 16) * 4;
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 out[8];
+    #pragma unroll
     for (uint d = 0; d < 8; ++d) {
         qm[d].thread_elements()[0] = half(Q[base + (qb + row) * 64 + d * 8 + lane_col]);
         qm[d].thread_elements()[1] = half(Q[base + (qb + row) * 64 + d * 8 + lane_col + 1]);
@@ -34,17 +35,21 @@
             kv[i] = half(K[base + kb * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 score[8];
-        for (uint j = 0; j < 8; ++j) {
+        #pragma unroll
+    for (uint j = 0; j < 8; ++j) {
             score[j] = simdgroup_float8x8(0.0f);
-            for (uint d = 0; d < 8; ++d) {
+            #pragma unroll
+    for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km, kv + j * 8 * 64 + d * 8, 64, ulong2(0), true);
                 simdgroup_multiply_accumulate(score[j], qm[d], km, score[j]);
             }
         }
         float tilemax = -INFINITY;
-        for (uint j = 0; j < 8; ++j) {
-            for (uint e = 0; e < 2; ++e) {
+        #pragma unroll
+    for (uint j = 0; j < 8; ++j) {
+            #pragma unroll
+    for (uint e = 0; e < 2; ++e) {
                 uint col = lane_col + j * 8 + e;
                 float z = kb + col <= qb + row ? score[j].thread_elements()[e] * 0.125f : -INFINITY;
                 score[j].thread_elements()[e] = z;
@@ -57,8 +62,10 @@
         float alpha = fast::exp(mx - newmx);
         float tilesum = 0.0f;
         simdgroup_half8x8 prob[8];
-        for (uint j = 0; j < 8; ++j) {
-            for (uint e = 0; e < 2; ++e) {
+        #pragma unroll
+    for (uint j = 0; j < 8; ++j) {
+            #pragma unroll
+    for (uint e = 0; e < 2; ++e) {
                 float p = fast::exp(score[j].thread_elements()[e] - newmx);
                 prob[j].thread_elements()[e] = half(p);
                 tilesum += p;
@@ -68,7 +75,8 @@
         tilesum += simd_shuffle_xor(tilesum, 8);
         den = den * alpha + tilesum;
         mx = newmx;
-        for (uint d = 0; d < 8; ++d) {
+        #pragma unroll
+    for (uint d = 0; d < 8; ++d) {
             out[d].thread_elements()[0] *= alpha;
             out[d].thread_elements()[1] *= alpha;
         }
@@ -76,8 +84,10 @@
         for (uint i = tid; i < 4096; i += 256)
             kv[i] = half(V[base + kb * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint j = 0; j < 8; ++j) {
-            for (uint d = 0; d < 8; ++d) {
+        #pragma unroll
+    for (uint j = 0; j < 8; ++j) {
+            #pragma unroll
+    for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 vm;
                 simdgroup_load(vm, kv + j * 8 * 64 + d * 8, 64);
                 simdgroup_multiply_accumulate(out[d], prob[j], vm, out[d]);
@@ -85,6 +95,7 @@
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
+    #pragma unroll
     for (uint d = 0; d < 8; ++d) {
         out[d].thread_elements()[0] /= den;
         out[d].thread_elements()[1] /= den;

```

</details>

`python bench.py`  
→ run 1 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 63.91 ms, kernel 9.15 ms, speedup 6.98x

> The first fused shader passes and runs in 9.15 ms, a 6.98× speedup. I’ll now keep scores and probabilities in registers to reduce shared-memory traffic.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.77e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -1,10 +1,15 @@
 """Fused causal attention for the fixed MPS workload."""
 import torch
+
+_BQ = 32
+_BK = 32
 
 _SOURCE = r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
+#define BQ __BQ__
+#define BK __BK__
 
 kernel void attention(
     device const float* Q [[buffer(0)]],
@@ -15,9 +20,9 @@
     uint tid [[thread_index_in_threadgroup]],
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
-    threadgroup half kv[64 * 64];
+    threadgroup half kv[BK * 64];
     const uint base = gid.y * 1024 * 64;
-    const uint qb = gid.x * 64;
+    const uint qb = gid.x * BQ;
     const uint lane_col = (lane % 2) * 2 + ((lane / 8) % 2) * 4;
     const uint row = sg * 8 + (lane % 8) / 2 + (lane / 16) * 4;
     simdgroup_half8x8 qm[8];
@@ -30,13 +35,13 @@
     }
     float mx = -INFINITY;
     float den = 0.0f;
-    for (uint kb = 0; kb <= qb; kb += 64) {
-        for (uint i = tid; i < 4096; i += 256)
+    for (uint kb = 0; kb < qb + BQ; kb += BK) {
+        for (uint i = tid; i < BK * 64; i += BQ * 4)
             kv[i] = half(K[base + kb * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 score[8];
+        simdgroup_float8x8 score[BK / 8];
         #pragma unroll
-    for (uint j = 0; j < 8; ++j) {
+    for (uint j = 0; j < BK / 8; ++j) {
             score[j] = simdgroup_float8x8(0.0f);
             #pragma unroll
     for (uint d = 0; d < 8; ++d) {
@@ -47,7 +52,7 @@
         }
         float tilemax = -INFINITY;
         #pragma unroll
-    for (uint j = 0; j < 8; ++j) {
+    for (uint j = 0; j < BK / 8; ++j) {
             #pragma unroll
     for (uint e = 0; e < 2; ++e) {
                 uint col = lane_col + j * 8 + e;
@@ -61,9 +66,9 @@
         float newmx = max(mx, tilemax);
         float alpha = fast::exp(mx - newmx);
         float tilesum = 0.0f;
-        simdgroup_half8x8 prob[8];
+        simdgroup_half8x8 prob[BK / 8];
         #pragma unroll
-    for (uint j = 0; j < 8; ++j) {
+    for (uint j = 0; j < BK / 8; ++j) {
             #pragma unroll
     for (uint e = 0; e < 2; ++e) {
                 float p = fast::exp(score[j].thread_elements()[e] - newmx);
@@ -81,11 +86,11 @@
             out[d].thread_elements()[1] *= alpha;
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < 4096; i += 256)
+        for (uint i = tid; i < BK * 64; i += BQ * 4)
             kv[i] = half(V[base + kb * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
         #pragma unroll
-    for (uint j = 0; j < 8; ++j) {
+    for (uint j = 0; j < BK / 8; ++j) {
             #pragma unroll
     for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 vm;
@@ -103,9 +108,9 @@
     }
 }
 '''
-_LIB = torch.mps.compile_shader(_SOURCE)
+_LIB = torch.mps.compile_shader(_SOURCE.replace("__BQ__", str(_BQ)).replace("__BK__", str(_BK)))
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _LIB.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
+    _LIB.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(_BQ * 4, 1, 1))
     return out

```

</details>

`python bench.py`  
→     _LIB.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))  
→     return out

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text(); s=s.replace(\"    for (uint d =\", \"    #pragma unroll\\n    for (uint d =\").r...`
- `python check.py`  
→ OK: relative error 3.44e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -1,15 +1,10 @@
 """Fused causal attention for the fixed MPS workload."""
 import torch
-
-_BQ = 32
-_BK = 32
 
 _SOURCE = r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-#define BQ __BQ__
-#define BK __BK__
 
 kernel void attention(
     device const float* Q [[buffer(0)]],
@@ -20,87 +15,80 @@
     uint tid [[thread_index_in_threadgroup]],
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
-    threadgroup half kv[BK * 64];
+    threadgroup half kv[64 * 64];
+    threadgroup float scores[64 * 64];
+    threadgroup half queries[64 * 64];
+    threadgroup half* probs = reinterpret_cast<threadgroup half*>(scores);
     const uint base = gid.y * 1024 * 64;
-    const uint qb = gid.x * BQ;
-    const uint lane_col = (lane % 2) * 2 + ((lane / 8) % 2) * 4;
-    const uint row = sg * 8 + (lane % 8) / 2 + (lane / 16) * 4;
-    simdgroup_half8x8 qm[8];
+    const uint qb = gid.x * 64;
+    for (uint i = tid; i < 4096; i += 256)
+        queries[i] = half(Q[base + qb * 64 + i]);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_float8x8 out[8];
-    #pragma unroll
     for (uint d = 0; d < 8; ++d) {
-        qm[d].thread_elements()[0] = half(Q[base + (qb + row) * 64 + d * 8 + lane_col]);
-        qm[d].thread_elements()[1] = half(Q[base + (qb + row) * 64 + d * 8 + lane_col + 1]);
         out[d] = simdgroup_float8x8(0.0f);
     }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     float mx = -INFINITY;
     float den = 0.0f;
-    for (uint kb = 0; kb < qb + BQ; kb += BK) {
-        for (uint i = tid; i < BK * 64; i += BQ * 4)
+    const uint lane_col = lane % 2 + ((lane / 8) % 2) * 2;
+    const uint row = sg * 8 + (lane % 8) / 2 + (lane / 16) * 4;
+    for (uint kb = 0; kb <= qb; kb += 64) {
+        for (uint i = tid; i < 4096; i += 256)
             kv[i] = half(K[base + kb * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 score[BK / 8];
-        #pragma unroll
-    for (uint j = 0; j < BK / 8; ++j) {
-            score[j] = simdgroup_float8x8(0.0f);
-            #pragma unroll
-    for (uint d = 0; d < 8; ++d) {
-                simdgroup_half8x8 km;
+        for (uint j = 0; j < 8; ++j) {
+            simdgroup_float8x8 acc(0.0f);
+            for (uint d = 0; d < 8; ++d) {
+                simdgroup_half8x8 km, qm;
+                simdgroup_load(qm, queries + sg * 8 * 64 + d * 8, 64);
                 simdgroup_load(km, kv + j * 8 * 64 + d * 8, 64, ulong2(0), true);
-                simdgroup_multiply_accumulate(score[j], qm[d], km, score[j]);
+                simdgroup_multiply_accumulate(acc, qm, km, acc);
             }
+            simdgroup_store(acc, scores + sg * 8 * 64 + j * 8, 64);
         }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        float vals[16];
         float tilemax = -INFINITY;
-        #pragma unroll
-    for (uint j = 0; j < BK / 8; ++j) {
-            #pragma unroll
-    for (uint e = 0; e < 2; ++e) {
-                uint col = lane_col + j * 8 + e;
-                float z = kb + col <= qb + row ? score[j].thread_elements()[e] * 0.125f : -INFINITY;
-                score[j].thread_elements()[e] = z;
-                tilemax = max(tilemax, z);
-            }
+        for (uint j = 0; j < 16; ++j) {
+            uint col = lane_col + j * 4;
+            float z = kb + col <= qb + row ? scores[row * 64 + col] * 0.125f : -INFINITY;
+            vals[j] = z;
+            tilemax = max(tilemax, z);
         }
         tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
         tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
         float newmx = max(mx, tilemax);
         float alpha = fast::exp(mx - newmx);
         float tilesum = 0.0f;
-        simdgroup_half8x8 prob[BK / 8];
-        #pragma unroll
-    for (uint j = 0; j < BK / 8; ++j) {
-            #pragma unroll
-    for (uint e = 0; e < 2; ++e) {
-                float p = fast::exp(score[j].thread_elements()[e] - newmx);
-                prob[j].thread_elements()[e] = half(p);
-                tilesum += p;
-            }
+        simdgroup_barrier(mem_flags::mem_threadgroup);
+        for (uint j = 0; j < 16; ++j) {
+            float p = fast::exp(vals[j] - newmx);
+            probs[row * 128 + lane_col + j * 4] = half(p);
+            tilesum += p;
         }
         tilesum += simd_shuffle_xor(tilesum, 1);
         tilesum += simd_shuffle_xor(tilesum, 8);
         den = den * alpha + tilesum;
         mx = newmx;
-        #pragma unroll
-    for (uint d = 0; d < 8; ++d) {
+        for (uint d = 0; d < 8; ++d) {
             out[d].thread_elements()[0] *= alpha;
             out[d].thread_elements()[1] *= alpha;
         }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < BK * 64; i += BQ * 4)
+        for (uint i = tid; i < 4096; i += 256)
             kv[i] = half(V[base + kb * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        #pragma unroll
-    for (uint j = 0; j < BK / 8; ++j) {
-            #pragma unroll
-    for (uint d = 0; d < 8; ++d) {
+        for (uint j = 0; j < 8; ++j) {
+            simdgroup_half8x8 pm;
+            simdgroup_load(pm, probs + sg * 8 * 128 + j * 8, 128);
+            for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 vm;
                 simdgroup_load(vm, kv + j * 8 * 64 + d * 8, 64);
-                simdgroup_multiply_accumulate(out[d], prob[j], vm, out[d]);
+                simdgroup_multiply_accumulate(out[d], pm, vm, out[d]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
-    #pragma unroll
     for (uint d = 0; d < 8; ++d) {
         out[d].thread_elements()[0] /= den;
         out[d].thread_elements()[1] /= den;
@@ -108,9 +96,9 @@
     }
 }
 '''
-_LIB = torch.mps.compile_shader(_SOURCE.replace("__BQ__", str(_BQ)).replace("__BK__", str(_BK)))
+_LIB = torch.mps.compile_shader(_SOURCE)
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _LIB.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(_BQ * 4, 1, 1))
+    _LIB.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
     return out

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 67.04 ms, kernel 11.64 ms, speedup 5.76x

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text(); s=s.replace(\"import torch\\n\", \"import torch\\n\\n_BQ = 32\\n_BK = 32\\n\"); ...`
- `python check.py`  
→ OK: relative error 3.94e-04 (allowed 2e-03)
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -1,10 +1,15 @@
 """Fused causal attention for the fixed MPS workload."""
 import torch
+
+_BQ = 64
+_BK = 32
 
 _SOURCE = r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
+#define BQ __BQ__
+#define BK __BK__
 
 kernel void attention(
     device const float* Q [[buffer(0)]],
@@ -15,17 +20,18 @@
     uint tid [[thread_index_in_threadgroup]],
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
-    threadgroup half kv[64 * 64];
-    threadgroup float scores[64 * 64];
-    threadgroup half queries[64 * 64];
-    threadgroup half* probs = reinterpret_cast<threadgroup half*>(scores);
+    threadgroup half kv[(BQ > BK ? BQ : BK) * 64];
+    threadgroup float scores[BQ * BK];
+    threadgroup half probs[BQ * BK];
     const uint base = gid.y * 1024 * 64;
-    const uint qb = gid.x * 64;
-    for (uint i = tid; i < 4096; i += 256)
-        queries[i] = half(Q[base + qb * 64 + i]);
+    const uint qb = gid.x * BQ;
+    for (uint i = tid; i < BQ * 64; i += BQ * 4)
+        kv[i] = half(Q[base + qb * 64 + i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
+    simdgroup_half8x8 qm[8];
     simdgroup_float8x8 out[8];
     for (uint d = 0; d < 8; ++d) {
+        simdgroup_load(qm[d], kv + sg * 8 * 64 + d * 8, 64);
         out[d] = simdgroup_float8x8(0.0f);
     }
     threadgroup_barrier(mem_flags::mem_threadgroup);
@@ -33,26 +39,25 @@
     float den = 0.0f;
     const uint lane_col = lane % 2 + ((lane / 8) % 2) * 2;
     const uint row = sg * 8 + (lane % 8) / 2 + (lane / 16) * 4;
-    for (uint kb = 0; kb <= qb; kb += 64) {
-        for (uint i = tid; i < 4096; i += 256)
+    for (uint kb = 0; kb < qb + BQ; kb += BK) {
+        for (uint i = tid; i < BK * 64; i += BQ * 4)
             kv[i] = half(K[base + kb * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint j = 0; j < 8; ++j) {
+        for (uint j = 0; j < BK / 8; ++j) {
             simdgroup_float8x8 acc(0.0f);
             for (uint d = 0; d < 8; ++d) {
-                simdgroup_half8x8 km, qm;
-                simdgroup_load(qm, queries + sg * 8 * 64 + d * 8, 64);
+                simdgroup_half8x8 km;
                 simdgroup_load(km, kv + j * 8 * 64 + d * 8, 64, ulong2(0), true);
-                simdgroup_multiply_accumulate(acc, qm, km, acc);
+                simdgroup_multiply_accumulate(acc, qm[d], km, acc);
             }
-            simdgroup_store(acc, scores + sg * 8 * 64 + j * 8, 64);
+            simdgroup_store(acc, scores + sg * 8 * BK + j * 8, BK);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        float vals[16];
+        float vals[BK / 4];
         float tilemax = -INFINITY;
-        for (uint j = 0; j < 16; ++j) {
+        for (uint j = 0; j < BK / 4; ++j) {
             uint col = lane_col + j * 4;
-            float z = kb + col <= qb + row ? scores[row * 64 + col] * 0.125f : -INFINITY;
+            float z = kb + col <= qb + row ? scores[row * BK + col] * 0.125f : -INFINITY;
             vals[j] = z;
             tilemax = max(tilemax, z);
         }
@@ -61,10 +66,9 @@
         float newmx = max(mx, tilemax);
         float alpha = fast::exp(mx - newmx);
         float tilesum = 0.0f;
-        simdgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint j = 0; j < 16; ++j) {
+        for (uint j = 0; j < BK / 4; ++j) {
             float p = fast::exp(vals[j] - newmx);
-            probs[row * 128 + lane_col + j * 4] = half(p);
+            probs[row * BK + lane_col + j * 4] = half(p);
             tilesum += p;
         }
         tilesum += simd_shuffle_xor(tilesum, 1);
@@ -75,12 +79,12 @@
             out[d].thread_elements()[0] *= alpha;
             out[d].thread_elements()[1] *= alpha;
         }
-        for (uint i = tid; i < 4096; i += 256)
+        for (uint i = tid; i < BK * 64; i += BQ * 4)
             kv[i] = half(V[base + kb * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint j = 0; j < 8; ++j) {
+        for (uint j = 0; j < BK / 8; ++j) {
             simdgroup_half8x8 pm;
-            simdgroup_load(pm, probs + sg * 8 * 128 + j * 8, 128);
+            simdgroup_load(pm, probs + sg * 8 * BK + j * 8, BK);
             for (uint d = 0; d < 8; ++d) {
                 simdgroup_half8x8 vm;
                 simdgroup_load(vm, kv + j * 8 * 64 + d * 8, 64);
@@ -96,9 +100,9 @@
     }
 }
 '''
-_LIB = torch.mps.compile_shader(_SOURCE)
+_LIB = torch.mps.compile_shader(_SOURCE.replace("__BQ__", str(_BQ)).replace("__BK__", str(_BK)))
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _LIB.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
+    _LIB.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(_BQ * 4, 1, 1))
     return out

```

</details>

`python bench.py`  


- `python check.py`  
→ OK: relative error 4.56e-04 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -2,7 +2,7 @@
 import torch
 
 _BQ = 64
-_BK = 32
+_BK = 64
 
 _SOURCE = r'''
 #include <metal_stdlib>
@@ -82,13 +82,17 @@
         for (uint i = tid; i < BK * 64; i += BQ * 4)
             kv[i] = half(V[base + kb * 64 + i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint j = 0; j < BK / 8; ++j) {
-            simdgroup_half8x8 pm;
-            simdgroup_load(pm, probs + sg * 8 * BK + j * 8, BK);
-            for (uint d = 0; d < 8; ++d) {
-                simdgroup_half8x8 vm;
-                simdgroup_load(vm, kv + j * 8 * 64 + d * 8, 64);
-                simdgroup_multiply_accumulate(out[d], pm, vm, out[d]);
+        for (uint d = 0; d < 8; ++d) {
+            for (uint j = 0; j < BK / 8; j += 2) {
+                simdgroup_half8x8 part(half(0.0h));
+                for (uint x = 0; x < 2; ++x) {
+                    simdgroup_half8x8 pm, vm;
+                    simdgroup_load(pm, probs + sg * 8 * BK + (j + x) * 8, BK);
+                    simdgroup_load(vm, kv + (j + x) * 8 * 64 + d * 8, 64);
+                    simdgroup_multiply_accumulate(part, pm, vm, part);
+                }
+                out[d].thread_elements()[0] += float(part.thread_elements()[0]);
+                out[d].thread_elements()[1] += float(part.thread_elements()[1]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);

```

</details>

`python bench.py`  
→ run 4 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 68.20 ms, kernel 13.35 ms, speedup 5.11x

- `rg --files results queue`
> The smaller register-only tiles took 13.35 ms, and the shared-query version took 9.93 ms. The original 9.15 ms version still leads. I’ll test a smaller key tile and half-precision value products accumulated in short blocks; the earlier notes suggest both may help.

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=Path(\"queue/run1.py\").read_text(); s=s.replace(\"import torch\\n\", \"import torch\\n\\n_BQ =...`
- `python check.py`  
→ OK: relative error 3.63e-04 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -2,7 +2,7 @@
 import torch
 
 _BQ = 64
-_BK = 64
+_BK = 32
 
 _SOURCE = r'''
 #include <metal_stdlib>

```

</details>

`python bench.py`  


- `python check.py`  
→ OK: relative error 6.98e-04 (allowed 2e-03)
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -2,7 +2,7 @@
 import torch
 
 _BQ = 64
-_BK = 32
+_BK = 64
 
 _SOURCE = r'''
 #include <metal_stdlib>
@@ -21,8 +21,8 @@
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
     threadgroup half kv[(BQ > BK ? BQ : BK) * 64];
-    threadgroup float scores[BQ * BK];
     threadgroup half probs[BQ * BK];
+    threadgroup half* scores = probs;
     const uint base = gid.y * 1024 * 64;
     const uint qb = gid.x * BQ;
     for (uint i = tid; i < BQ * 64; i += BQ * 4)
@@ -50,7 +50,10 @@
                 simdgroup_load(km, kv + j * 8 * 64 + d * 8, 64, ulong2(0), true);
                 simdgroup_multiply_accumulate(acc, qm[d], km, acc);
             }
-            simdgroup_store(acc, scores + sg * 8 * BK + j * 8, BK);
+            simdgroup_half8x8 stored;
+            stored.thread_elements()[0] = half(acc.thread_elements()[0]);
+            stored.thread_elements()[1] = half(acc.thread_elements()[1]);
+            simdgroup_store(stored, scores + sg * 8 * BK + j * 8, BK);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         float vals[BK / 4];
@@ -66,6 +69,7 @@
         float newmx = max(mx, tilemax);
         float alpha = fast::exp(mx - newmx);
         float tilesum = 0.0f;
+        simdgroup_barrier(mem_flags::mem_threadgroup);
         for (uint j = 0; j < BK / 4; ++j) {
             float p = fast::exp(vals[j] - newmx);
             probs[row * BK + lane_col + j * 4] = half(p);

```

</details>

`python bench.py`  
→ run 6 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 67.45 ms, kernel 9.68 ms, speedup 6.97x

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text().replace(\"_BQ = 64\", \"_BQ = 32\"); p.write_text(s)`
- `python check.py`  
→ run 7 of 10 submitted.  
→ attention: correct (relative error 1.2e-03). reference 67.69 ms, kernel 8.78 ms, speedup 7.71x
> Short half-precision value products improved the time to 8.78 ms, a 7.71× speedup. That run’s error was 0.0012 against the 0.002 limit. I’ll tune this version further and check it repeatedly on fresh inputs.

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text().replace(\"_BQ = 32\", \"_BQ = 64\").replace(\"_BK = 64\", \"_BK = 32\"); p.write_...`
- `python check.py`  
→ OK: relative error 7.50e-04 (allowed 2e-03)
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -2,7 +2,7 @@
 import torch
 
 _BQ = 64
-_BK = 64
+_BK = 32
 
 _SOURCE = r'''
 #include <metal_stdlib>
@@ -20,21 +20,20 @@
     uint tid [[thread_index_in_threadgroup]],
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
-    threadgroup half kv[(BQ > BK ? BQ : BK) * 64];
+    threadgroup half kv[BK * 64];
     threadgroup half probs[BQ * BK];
     threadgroup half* scores = probs;
     const uint base = gid.y * 1024 * 64;
     const uint qb = gid.x * BQ;
-    for (uint i = tid; i < BQ * 64; i += BQ * 4)
-        kv[i] = half(Q[base + qb * 64 + i]);
-    threadgroup_barrier(mem_flags::mem_threadgroup);
+    const uint matrix_row = sg * 8 + (lane % 8) / 2 + (lane / 16) * 4;
+    const uint matrix_col = (lane % 2) * 2 + ((lane / 8) % 2) * 4;
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 out[8];
     for (uint d = 0; d < 8; ++d) {
-        simdgroup_load(qm[d], kv + sg * 8 * 64 + d * 8, 64);
+        qm[d].thread_elements()[0] = half(Q[base + (qb + matrix_row) * 64 + d * 8 + matrix_col]);
+        qm[d].thread_elements()[1] = half(Q[base + (qb + matrix_row) * 64 + d * 8 + matrix_col + 1]);
         out[d] = simdgroup_float8x8(0.0f);
     }
-    threadgroup_barrier(mem_flags::mem_threadgroup);
     float mx = -INFINITY;
     float den = 0.0f;
     const uint lane_col = lane % 2 + ((lane / 8) % 2) * 2;

```

</details>

`python bench.py`  


- `python check.py`  
→ OK: relative error 8.71e-04 (allowed 2e-03)
### Bench run 11

`python bench.py`  
→ run 8 of 10 submitted.  
→ attention: correct (relative error 1.2e-03). reference 69.65 ms, kernel 8.49 ms, speedup 8.20x

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text().replace(\"_BK = 64\", \"_BK = 32\").replace(\"kv[(BQ > BK ? BQ : BK) * 64]\", \"k...`
- `python check.py`  
→ OK: relative error 7.71e-04 (allowed 2e-03)
### Bench run 12

`python bench.py`  
→ run 9 of 10 submitted.  
→ attention: correct (relative error 1.3e-03). reference 66.92 ms, kernel 6.09 ms, speedup 10.99x

- `cat kernel.py`
> Reusing the same half-precision buffer for scores and probabilities cut the time to 6.09 ms, a 10.99× speedup. The tenth run is testing a 32-key tile with queries loaded directly into matrix registers. I’ll retain whichever is faster.

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=Path(\"queue/run9.py\").read_text(); s=s.replace(\"    threadgroup half* scores = probs;\", \" ...`
- `python -c 'import runpy; [runpy.run_path(\"check.py\") for _ in range(20)]`  
→ OK: relative error 1.08e-03 (allowed 2e-03)  
→ OK: relative error 9.39e-04 (allowed 2e-03)
> The last variant took 9.55 ms, so I restored the 6.09 ms version. I’m checking it on twenty fresh input sets and updating LESSONS.md with the measured results.

- `python -c 'from pathlib import Path; notes=\"\"\"The workload is float32 causal attention on MPS: eight batches, eight heads, 1024 positions, and 64 channels...`
- `python check.py`  
→ OK: relative error 8.98e-04 (allowed 2e-03)
> Measured **6.09 ms**, a **10.99× speedup**. Passed 21 fresh-input correctness checks. Rewrote LESSONS.md with results and guidance.
> 
> EDIT: Final kernel.py fuses tiled causal attention in Metal, reuses half precision score storage for probabilities, and keeps softmax and cumulative output in float32.

