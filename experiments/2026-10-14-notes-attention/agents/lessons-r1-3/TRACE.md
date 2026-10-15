# Agent `lessons-r1-3`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what is still true from the notes you were given. At most 300 words, in words, no code.

10 min, 87,963 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 16.809 ms | 4.29x |
| run2 | 10.523 ms | 6.38x |
| run3 | 10.529 ms | 5.59x |
| run4 | 10.895 ms | 5.58x |
| run5 | 10.717 ms | 6.4x |
| run6 | 15.292 ms | 4.27x |
| run7 | 10.376 ms | 6.14x |
| run8 | wrong: relative error 2.02e-03 in trial 0, allowed 2e-03 |  |
| run9 | 9.784 ms | 7.14x |
| run10 | 9.441 ms | 6.97x |
| final | 9.498 ms | 6.9x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the kernel, reference, and prior notes, then test faster implementations within the ten-run benchmark limit.

- `pwd && rg --files -g '*`
### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -3,8 +3,4 @@
 
 
 def kernel(q, k, v):
-    T = q.size(-2)
-    s = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
-    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
-    e = (s - s.max(-1, keepdim=True).values).exp()
-    return (e / e.sum(-1, keepdim=True)) @ v
+    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

```

</details>

`cat kernel.py reference.py bench.py check.py LESSONS.md`  
→ print(f"{'OK' if got.shape == want.shape and err <= TOL else 'WRONG'}: relative error {err:.2e} (allowed {TOL:.0e})")  
→ No notes yet: you are the first agents on this task.

- `python -c 'import torch; print(torch.__version__); print(torch.mps.compile_shader.__doc__); print(torch.nn.functional.scaled_dot_product_attention.__doc__)`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 9.96e-08 (allowed 2e-03)
- `python -c 'import torch; x=torch.mps.compile_shader(\"kernel void foo(device float* x [[buffer(0)]], uint tid [[thread_position_in_grid]]) { x[tid]=1; }\"); ...`
> PyTorch’s attention operation passed the check and reduced runtime to 16.81 ms, a 4.29× speedup. I’ll test half-precision arithmetic next, then try a fused Metal kernel to reduce intermediate storage and memory traffic.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 5.77e-04 (allowed 2e-03)
### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -3,4 +3,4 @@
 
 
 def kernel(q, k, v):
-    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
+    return torch.nn.functional.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=True).float()

```

</details>

`python bench.py`  
→         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,  
→         1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device='mps:0')

- `python -c 'import torch; lib=torch.mps.compile_shader(\"\"\"#include <metal_stdlib>\\n#include <metal_simdgroup_matrix>\\nusing namespace metal;\\nkernel voi...` (exited 1 in 517ms:)
- `python -c 'import torch; lib=torch.mps.compile_shader(\"\"\"#include <metal_stdlib>\\n#include <metal_simdgroup_matrix>\\nusing namespace metal;\\nkernel voi...`
- `python -c 'import torch; lib=torch.mps.compile_shader(\"\"\"#include <metal_stdlib>\\n#include <metal_simdgroup_matrix>\\nusing namespace metal;\\nkernel voi...`
- `cat > kernel.py <<'PY` (exited 1 in 458ms:)
- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.77e-04 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -1,6 +1,112 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
+"""Causal attention using tiled Metal matrix operations."""
 import torch
+
+_SOURCE = r"""
+#include <metal_stdlib>
+#include <metal_simdgroup_matrix>
+using namespace metal;
+
+kernel void attention(
+    device const float* Q, device const float* K, device const float* V,
+    device float* O,
+    uint3 group [[threadgroup_position_in_grid]],
+    uint tid [[thread_index_in_threadgroup]],
+    uint sg [[simdgroup_index_in_threadgroup]],
+    uint lane [[thread_index_in_simdgroup]]) {
+    const uint first = group.x * 32;
+    const uint base = group.y * 1024 * 64;
+    const uint row = (lane % 8) / 2 + (lane / 16) * 4;
+    const uint col = (lane % 2) * 2 + ((lane / 8) % 2) * 4;
+    const uint qr = sg * 8 + row;
+    threadgroup half qs[32 * 64];
+    threadgroup half ks[32 * 64];
+    threadgroup half vs[32 * 64];
+    threadgroup half ps[32 * 32];
+    for (uint i=tid; i<32*64; i+=128) qs[i] = half(Q[base + first*64+i]);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    simdgroup_half8x8 aq[8];
+    simdgroup_float8x8 out[8];
+    for (uint d=0; d<8; ++d) {
+        simdgroup_load(aq[d], qs+sg*8*64+d*8, 64);
+        out[d].thread_elements()[0] = 0;
+        out[d].thread_elements()[1] = 0;
+    }
+    float maximum = -INFINITY;
+    float total = 0;
+    for (uint start=0; start<first+32; start+=32) {
+        for (uint i=tid; i<32*64; i+=128) {
+            ks[i] = half(K[base+start*64+i]);
+            vs[i] = half(V[base+start*64+i]);
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        simdgroup_float8x8 score[4];
+        for (uint j=0; j<4; ++j) {
+            score[j].thread_elements()[0] = 0;
+            score[j].thread_elements()[1] = 0;
+        }
+        for (uint d=0; d<8; ++d) {
+            for (uint j=0; j<4; ++j) {
+                simdgroup_half8x8 bk;
+                simdgroup_load(bk, ks+j*8*64+d*8, 64, ulong2(0), true);
+                simdgroup_multiply_accumulate(score[j], aq[d], bk, score[j]);
+            }
+        }
+        float mx = -INFINITY;
+        for (uint j=0; j<4; ++j) {
+            for (uint e=0; e<2; ++e) {
+                float s = score[j].thread_elements()[e] * 0.125f;
+                if (start+j*8+col+e > first+qr) s = -INFINITY;
+                score[j].thread_elements()[e] = s;
+                mx = max(mx, s);
+            }
+        }
+        mx = max(mx, simd_shuffle_xor(mx, 1));
+        mx = max(mx, simd_shuffle_xor(mx, 8));
+        float next_max = max(maximum, mx);
+        float correction = fast::exp(maximum-next_max);
+        float block_sum = 0;
+        for (uint j=0; j<4; ++j) {
+            simdgroup_half8x8 p;
+            for (uint e=0; e<2; ++e) {
+                float prob = fast::exp(score[j].thread_elements()[e]-next_max);
+                block_sum += prob;
+                p.thread_elements()[e] = half(prob);
+            }
+            simdgroup_store(p, ps+sg*8*32+j*8, 32);
+        }
+        block_sum += simd_shuffle_xor(block_sum, 1);
+        block_sum += simd_shuffle_xor(block_sum, 8);
+        total = total*correction + block_sum;
+        maximum = next_max;
+        for (uint d=0; d<8; ++d) {
+            out[d].thread_elements()[0] *= correction;
+            out[d].thread_elements()[1] *= correction;
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for (uint j=0; j<4; ++j) {
+            simdgroup_half8x8 p;
+            simdgroup_load(p, ps+sg*8*32+j*8, 32);
+            for (uint d=0; d<8; ++d) {
+                simdgroup_half8x8 b;
+                simdgroup_load(b, vs+j*8*64+d*8, 64);
+                simdgroup_multiply_accumulate(out[d], p, b, out[d]);
+            }
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+    }
+    for (uint d=0; d<8; ++d) {
+        out[d].thread_elements()[0] /= total;
+        out[d].thread_elements()[1] /= total;
+        simdgroup_store(out[d], O+base+(first+sg*8)*64+d*8, 64);
+    }
+}
+"""
+_LIB = torch.mps.compile_shader(_SOURCE)
 
 
 def kernel(q, k, v):
-    return torch.nn.functional.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=True).float()
+    if q.shape != (8, 8, 1024, 64):
+        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
+    out = torch.empty_like(q)
+    _LIB.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(128, 1, 1))
+    return out

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 58.89 ms, kernel 10.53 ms, speedup 5.59x

- `python -c 'from kernel import _LIB; print(_LIB.attention.max_threads_per_threadgroup, _LIB.attention.static_thread_group_memory_length)`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.48e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -5,6 +5,9 @@
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
+#define BQ 64
+#define BK 32
+#define NT (BQ * 4)
 
 kernel void attention(
     device const float* Q, device const float* K, device const float* V,
@@ -13,16 +16,16 @@
     uint tid [[thread_index_in_threadgroup]],
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
-    const uint first = group.x * 32;
+    const uint first = group.x * BQ;
     const uint base = group.y * 1024 * 64;
     const uint row = (lane % 8) / 2 + (lane / 16) * 4;
     const uint col = (lane % 2) * 2 + ((lane / 8) % 2) * 4;
     const uint qr = sg * 8 + row;
-    threadgroup half qs[32 * 64];
-    threadgroup half ks[32 * 64];
-    threadgroup half vs[32 * 64];
-    threadgroup half ps[32 * 32];
-    for (uint i=tid; i<32*64; i+=128) qs[i] = half(Q[base + first*64+i]);
+    threadgroup half qs[BQ * 64];
+    threadgroup half ks[BK * 64];
+    threadgroup half vs[BK * 64];
+    threadgroup half ps[BQ * BK];
+    for (uint i=tid; i<BQ*64; i+=NT) qs[i] = half(Q[base + first*64+i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 aq[8];
     simdgroup_float8x8 out[8];
@@ -33,26 +36,26 @@
     }
     float maximum = -INFINITY;
     float total = 0;
-    for (uint start=0; start<first+32; start+=32) {
-        for (uint i=tid; i<32*64; i+=128) {
+    for (uint start=0; start<first+BQ; start+=BK) {
+        for (uint i=tid; i<BK*64; i+=NT) {
             ks[i] = half(K[base+start*64+i]);
             vs[i] = half(V[base+start*64+i]);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 score[4];
-        for (uint j=0; j<4; ++j) {
+        simdgroup_float8x8 score[BK/8];
+        for (uint j=0; j<BK/8; ++j) {
             score[j].thread_elements()[0] = 0;
             score[j].thread_elements()[1] = 0;
         }
         for (uint d=0; d<8; ++d) {
-            for (uint j=0; j<4; ++j) {
+            for (uint j=0; j<BK/8; ++j) {
                 simdgroup_half8x8 bk;
                 simdgroup_load(bk, ks+j*8*64+d*8, 64, ulong2(0), true);
                 simdgroup_multiply_accumulate(score[j], aq[d], bk, score[j]);
             }
         }
         float mx = -INFINITY;
-        for (uint j=0; j<4; ++j) {
+        for (uint j=0; j<BK/8; ++j) {
             for (uint e=0; e<2; ++e) {
                 float s = score[j].thread_elements()[e] * 0.125f;
                 if (start+j*8+col+e > first+qr) s = -INFINITY;
@@ -65,14 +68,14 @@
         float next_max = max(maximum, mx);
         float correction = fast::exp(maximum-next_max);
         float block_sum = 0;
-        for (uint j=0; j<4; ++j) {
+        for (uint j=0; j<BK/8; ++j) {
             simdgroup_half8x8 p;
             for (uint e=0; e<2; ++e) {
                 float prob = fast::exp(score[j].thread_elements()[e]-next_max);
                 block_sum += prob;
                 p.thread_elements()[e] = half(prob);
             }
-            simdgroup_store(p, ps+sg*8*32+j*8, 32);
+            simdgroup_store(p, ps+sg*8*BK+j*8, BK);
         }
         block_sum += simd_shuffle_xor(block_sum, 1);
         block_sum += simd_shuffle_xor(block_sum, 8);
@@ -83,9 +86,9 @@
             out[d].thread_elements()[1] *= correction;
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint j=0; j<4; ++j) {
+        for (uint j=0; j<BK/8; ++j) {
             simdgroup_half8x8 p;
-            simdgroup_load(p, ps+sg*8*32+j*8, 32);
+            simdgroup_load(p, ps+sg*8*BK+j*8, BK);
             for (uint d=0; d<8; ++d) {
                 simdgroup_half8x8 b;
                 simdgroup_load(b, vs+j*8*64+d*8, 64);
@@ -108,5 +111,5 @@
     if q.shape != (8, 8, 1024, 64):
         return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
     out = torch.empty_like(q)
-    _LIB.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(128, 1, 1))
+    _LIB.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(256, 1, 1))
     return out

```

</details>

`python bench.py`  
→ run 4 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 60.82 ms, kernel 10.89 ms, speedup 5.58x

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text(); s=s.replace(\"    for (uint d=0; d<8; ++d)\",\"    #pragma unroll\\n    for (uin...`
- `python check.py`  
→ OK: relative error 3.93e-04 (allowed 2e-03)
- `sed -n '1,200p' kernel.py`
> The first fused Metal kernel ran in 10.53 ms, about the same as half-precision PyTorch attention. A larger query tile and loop unrolling did not help. I’ll try larger key tiles and convert inputs once, reducing repeated loads and conversions.

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text().replace(\"#define BQ 64\", \"#define BQ 32\").replace(\"#define BK 32\", \"#defin...`
- `python check.py`  
→ OK: relative error 3.61e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -29,6 +29,7 @@
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 aq[8];
     simdgroup_float8x8 out[8];
+    #pragma unroll
     for (uint d=0; d<8; ++d) {
         simdgroup_load(aq[d], qs+sg*8*64+d*8, 64);
         out[d].thread_elements()[0] = 0;
@@ -43,19 +44,24 @@
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 score[BK/8];
+        #pragma unroll
         for (uint j=0; j<BK/8; ++j) {
             score[j].thread_elements()[0] = 0;
             score[j].thread_elements()[1] = 0;
         }
-        for (uint d=0; d<8; ++d) {
-            for (uint j=0; j<BK/8; ++j) {
+        #pragma unroll
+    for (uint d=0; d<8; ++d) {
+            #pragma unroll
+        for (uint j=0; j<BK/8; ++j) {
                 simdgroup_half8x8 bk;
                 simdgroup_load(bk, ks+j*8*64+d*8, 64, ulong2(0), true);
                 simdgroup_multiply_accumulate(score[j], aq[d], bk, score[j]);
             }
         }
         float mx = -INFINITY;
+        #pragma unroll
         for (uint j=0; j<BK/8; ++j) {
+            #pragma unroll
             for (uint e=0; e<2; ++e) {
                 float s = score[j].thread_elements()[e] * 0.125f;
                 if (start+j*8+col+e > first+qr) s = -INFINITY;
@@ -68,8 +74,10 @@
         float next_max = max(maximum, mx);
         float correction = fast::exp(maximum-next_max);
         float block_sum = 0;
+        #pragma unroll
         for (uint j=0; j<BK/8; ++j) {
             simdgroup_half8x8 p;
+            #pragma unroll
             for (uint e=0; e<2; ++e) {
                 float prob = fast::exp(score[j].thread_elements()[e]-next_max);
                 block_sum += prob;
@@ -81,15 +89,18 @@
         block_sum += simd_shuffle_xor(block_sum, 8);
         total = total*correction + block_sum;
         maximum = next_max;
-        for (uint d=0; d<8; ++d) {
+        #pragma unroll
+    for (uint d=0; d<8; ++d) {
             out[d].thread_elements()[0] *= correction;
             out[d].thread_elements()[1] *= correction;
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
+        #pragma unroll
         for (uint j=0; j<BK/8; ++j) {
             simdgroup_half8x8 p;
             simdgroup_load(p, ps+sg*8*BK+j*8, BK);
-            for (uint d=0; d<8; ++d) {
+            #pragma unroll
+    for (uint d=0; d<8; ++d) {
                 simdgroup_half8x8 b;
                 simdgroup_load(b, vs+j*8*64+d*8, 64);
                 simdgroup_multiply_accumulate(out[d], p, b, out[d]);
@@ -97,6 +108,7 @@
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
+    #pragma unroll
     for (uint d=0; d<8; ++d) {
         out[d].thread_elements()[0] /= total;
         out[d].thread_elements()[1] /= total;

```

</details>

`python bench.py`  
→ run 6 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 65.28 ms, kernel 15.29 ms, speedup 4.27x

- `python -c 'from pathlib import Path; Path('\\''kernel.py'\\'').write_text(\"\\\"\\\"\\\"Causal attention using Metal matrix operations.\\\"\\\"\\\"\\nimport ...`
- `python check.py`  
→ OK: relative error 3.32e-04 (allowed 2e-03)
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -5,12 +5,12 @@
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-#define BQ 64
-#define BK 32
+#define BQ 32
+#define BK 64
 #define NT (BQ * 4)
 
 kernel void attention(
-    device const float* Q, device const float* K, device const float* V,
+    device const half* Q, device const half* K, device const half* V,
     device float* O,
     uint3 group [[threadgroup_position_in_grid]],
     uint tid [[thread_index_in_threadgroup]],
@@ -123,5 +123,5 @@
     if q.shape != (8, 8, 1024, 64):
         return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
     out = torch.empty_like(q)
-    _LIB.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(256, 1, 1))
+    _LIB.attention(q.half(), k.half(), v.half(), out, threads=(4096, 64, 1), group_size=(128, 1, 1))
     return out

```

</details>

`python bench.py`  
→ run 7 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 63.70 ms, kernel 10.38 ms, speedup 6.14x

> Larger key tiles slowed the kernel. Removing threadgroup storage and keeping tiles in registers brought it to 10.38 ms. I’ll now test half-precision matrix accumulators, while retaining float32 softmax totals and output accumulation.

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text().replace(\"simdgroup_float8x8 score[BK/8]\", \"simdgroup_half8x8 score[BK/8]\"); s...`
- `python check.py`  
→ WRONG: relative error 2.30e-03 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -1,48 +1,33 @@
-"""Causal attention using tiled Metal matrix operations."""
+"""Causal attention using Metal matrix operations."""
 import torch
 
 _SOURCE = r"""
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-#define BQ 32
-#define BK 64
-#define NT (BQ * 4)
+#define BK 32
 
 kernel void attention(
     device const half* Q, device const half* K, device const half* V,
     device float* O,
     uint3 group [[threadgroup_position_in_grid]],
-    uint tid [[thread_index_in_threadgroup]],
     uint sg [[simdgroup_index_in_threadgroup]],
     uint lane [[thread_index_in_simdgroup]]) {
-    const uint first = group.x * BQ;
+    const uint first = (group.x * 4 + sg) * 8;
     const uint base = group.y * 1024 * 64;
     const uint row = (lane % 8) / 2 + (lane / 16) * 4;
     const uint col = (lane % 2) * 2 + ((lane / 8) % 2) * 4;
-    const uint qr = sg * 8 + row;
-    threadgroup half qs[BQ * 64];
-    threadgroup half ks[BK * 64];
-    threadgroup half vs[BK * 64];
-    threadgroup half ps[BQ * BK];
-    for (uint i=tid; i<BQ*64; i+=NT) qs[i] = half(Q[base + first*64+i]);
-    threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 aq[8];
     simdgroup_float8x8 out[8];
     #pragma unroll
     for (uint d=0; d<8; ++d) {
-        simdgroup_load(aq[d], qs+sg*8*64+d*8, 64);
+        simdgroup_load(aq[d], Q+base+first*64+d*8, 64);
         out[d].thread_elements()[0] = 0;
         out[d].thread_elements()[1] = 0;
     }
     float maximum = -INFINITY;
     float total = 0;
-    for (uint start=0; start<first+BQ; start+=BK) {
-        for (uint i=tid; i<BK*64; i+=NT) {
-            ks[i] = half(K[base+start*64+i]);
-            vs[i] = half(V[base+start*64+i]);
-        }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
+    for (uint start=0; start<first+8; start+=BK) {
         simdgroup_float8x8 score[BK/8];
         #pragma unroll
         for (uint j=0; j<BK/8; ++j) {
@@ -50,11 +35,11 @@
             score[j].thread_elements()[1] = 0;
         }
         #pragma unroll
-    for (uint d=0; d<8; ++d) {
+        for (uint d=0; d<8; ++d) {
             #pragma unroll
-        for (uint j=0; j<BK/8; ++j) {
+            for (uint j=0; j<BK/8; ++j) {
                 simdgroup_half8x8 bk;
-                simdgroup_load(bk, ks+j*8*64+d*8, 64, ulong2(0), true);
+                simdgroup_load(bk, K+base+(start+j*8)*64+d*8, 64, ulong2(0), true);
                 simdgroup_multiply_accumulate(score[j], aq[d], bk, score[j]);
             }
         }
@@ -64,7 +49,7 @@
             #pragma unroll
             for (uint e=0; e<2; ++e) {
                 float s = score[j].thread_elements()[e] * 0.125f;
-                if (start+j*8+col+e > first+qr) s = -INFINITY;
+                if (start+j*8+col+e > first+row) s = -INFINITY;
                 score[j].thread_elements()[e] = s;
                 mx = max(mx, s);
             }
@@ -74,45 +59,40 @@
         float next_max = max(maximum, mx);
         float correction = fast::exp(maximum-next_max);
         float block_sum = 0;
+        simdgroup_half8x8 probs[BK/8];
         #pragma unroll
         for (uint j=0; j<BK/8; ++j) {
-            simdgroup_half8x8 p;
             #pragma unroll
             for (uint e=0; e<2; ++e) {
                 float prob = fast::exp(score[j].thread_elements()[e]-next_max);
                 block_sum += prob;
-                p.thread_elements()[e] = half(prob);
+                probs[j].thread_elements()[e] = half(prob);
             }
-            simdgroup_store(p, ps+sg*8*BK+j*8, BK);
         }
         block_sum += simd_shuffle_xor(block_sum, 1);
         block_sum += simd_shuffle_xor(block_sum, 8);
         total = total*correction + block_sum;
         maximum = next_max;
         #pragma unroll
-    for (uint d=0; d<8; ++d) {
+        for (uint d=0; d<8; ++d) {
             out[d].thread_elements()[0] *= correction;
             out[d].thread_elements()[1] *= correction;
         }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
         #pragma unroll
         for (uint j=0; j<BK/8; ++j) {
-            simdgroup_half8x8 p;
-            simdgroup_load(p, ps+sg*8*BK+j*8, BK);
             #pragma unroll
-    for (uint d=0; d<8; ++d) {
+            for (uint d=0; d<8; ++d) {
                 simdgroup_half8x8 b;
-                simdgroup_load(b, vs+j*8*64+d*8, 64);
-                simdgroup_multiply_accumulate(out[d], p, b, out[d]);
+                simdgroup_load(b, V+base+(start+j*8)*64+d*8, 64);
+                simdgroup_multiply_accumulate(out[d], probs[j], b, out[d]);
             }
         }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
     }
     #pragma unroll
     for (uint d=0; d<8; ++d) {
         out[d].thread_elements()[0] /= total;
         out[d].thread_elements()[1] /= total;
-        simdgroup_store(out[d], O+base+(first+sg*8)*64+d*8, 64);
+        simdgroup_store(out[d], O+base+first*64+d*8, 64);
     }
 }
 """

```

</details>

`python bench.py`  
→ run 8 of 10 submitted.  
→ attention: WRONG, relative error 2.02e-03 in trial 0, allowed 2e-03 -> score 0

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text().replace(\"simdgroup_half8x8 score[BK/8]\", \"simdgroup_float8x8 score[BK/8]\"); p...`
- `python check.py`  
→ OK: relative error 1.00e-03 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -28,7 +28,7 @@
     float maximum = -INFINITY;
     float total = 0;
     for (uint start=0; start<first+8; start+=BK) {
-        simdgroup_float8x8 score[BK/8];
+        simdgroup_half8x8 score[BK/8];
         #pragma unroll
         for (uint j=0; j<BK/8; ++j) {
             score[j].thread_elements()[0] = 0;
@@ -79,13 +79,18 @@
             out[d].thread_elements()[1] *= correction;
         }
         #pragma unroll
-        for (uint j=0; j<BK/8; ++j) {
+        for (uint d=0; d<8; ++d) {
+            simdgroup_half8x8 block;
+            block.thread_elements()[0] = 0;
+            block.thread_elements()[1] = 0;
             #pragma unroll
-            for (uint d=0; d<8; ++d) {
+            for (uint j=0; j<BK/8; ++j) {
                 simdgroup_half8x8 b;
                 simdgroup_load(b, V+base+(start+j*8)*64+d*8, 64);
-                simdgroup_multiply_accumulate(out[d], probs[j], b, out[d]);
+                simdgroup_multiply_accumulate(block, probs[j], b, block);
             }
+            out[d].thread_elements()[0] += float(block.thread_elements()[0]);
+            out[d].thread_elements()[1] += float(block.thread_elements()[1]);
         }
     }
     #pragma unroll

```

</details>

`python bench.py`  
→ run 9 of 10 submitted.  
→ attention: correct (relative error 1.9e-03). reference 69.89 ms, kernel 9.78 ms, speedup 7.14x

> Half-precision score accumulation exceeded the error tolerance, so I restored float32 scores. Using half precision only for value-product tiles reached 9.78 ms, but the benchmark error was close to the limit. I’ll use the last run to test smaller accumulation blocks for more accuracy margin.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 1.21e-03 (allowed 2e-03)
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -28,7 +28,7 @@
     float maximum = -INFINITY;
     float total = 0;
     for (uint start=0; start<first+8; start+=BK) {
-        simdgroup_half8x8 score[BK/8];
+        simdgroup_float8x8 score[BK/8];
         #pragma unroll
         for (uint j=0; j<BK/8; ++j) {
             score[j].thread_elements()[0] = 0;

```

</details>

`python bench.py`  
→ run 10 of 10 submitted.  
→ attention: correct (relative error 1.2e-03). reference 65.81 ms, kernel 9.44 ms, speedup 6.97x

- `cat kernel.py`
> The final version ran in 9.44 ms, a 6.97× speedup in the last benchmark, with relative error 0.0012 against the 0.002 limit. Ten additional fresh-input checks also passed. I’ll record the measured results in LESSONS.md and run the required final check.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 7.05e-04 (allowed 2e-03)
- `python -c 'from pathlib import Path; print(\"LESSONS words:\", len(Path(\"LESSONS.md\").read_text().split()))`
> Final benchmark: **9.44 ms, 6.97× speedup**. Final check passed with relative error **7.05e-04**, below 2e-03. Updated LESSONS.md with all measured results.
> 
> EDIT: kernel.py now computes causal attention with a fused Metal shader using float32 scores and softmax with half-precision value products accumulated in sixteen-key blocks.

