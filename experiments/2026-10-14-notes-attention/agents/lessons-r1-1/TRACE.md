# Agent `lessons-r1-1`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what is still true from the notes you were given. At most 300 words, in words, no code.

11 min, 80,227 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 17.011 ms | 4.24x |
| run2 | 10.64 ms | 6.43x |
| run3 | 8.04 ms | 7.28x |
| run4 | 7.039 ms | 8.23x |
| run5 | 7.526 ms | 8.31x |
| run6 | 10.61 ms | 6.16x |
| run7 | 8.552 ms | 7.52x |
| run8 | 7.431 ms | 9.55x |
| run9 | 8.805 ms | 7.25x |
| run10 | 7.768 ms | 8.27x |
| final | 8.279 ms | 7.96x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the task files and prior lessons, then test faster attention implementations within the ten-run budget.

- `pwd && rg --files -g '*`
### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -1,10 +1,7 @@
 """Make kernel() faster. It must return the same values as reference() in reference.py."""
 import torch
+import torch.nn.functional as F
 
 
 def kernel(q, k, v):
-    T = q.size(-2)
-    s = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
-    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
-    e = (s - s.max(-1, keepdim=True).values).exp()
-    return (e / e.sum(-1, keepdim=True)) @ v
+    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

```

</details>

`cat kernel.py reference.py bench.py check.py LESSONS.md`  
→ print(f"{'OK' if got.shape == want.shape and err <= TOL else 'WRONG'}: relative error {err:.2e} (allowed {TOL:.0e})")  
→ No notes yet: you are the first agents on this task.

- `python -c 'import torch; print(torch.__version__); print(hasattr(torch.mps,\"compile_shader\")); print(torch.mps.compile_shader.__doc__)`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 1.57e-07 (allowed 2e-03)
### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -4,4 +4,4 @@
 
 
 def kernel(q, k, v):
-    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
+    return F.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=True).float()

```

</details>

`python bench.py`  
→  |  __new__(*args, **kwargs) class method of pybind11_builtins.pybind11_object  
→  |      Create and return a new object.  See help(type) for accurate signature.

### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -1,7 +1,118 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
+"""Causal attention using tiled Metal matrix operations."""
 import torch
 import torch.nn.functional as F
 
+_BM = 32
+_BN = 64
+_SOURCE = r'''
+#include <metal_stdlib>
+#include <metal_simdgroup_matrix>
+using namespace metal;
+constant uint BM = __BM__;
+constant uint BN = __BN__;
+kernel void attention(device const float* Q [[buffer(0)]],
+                      device const float* K [[buffer(1)]],
+                      device const float* V [[buffer(2)]],
+                      device float* Y [[buffer(3)]],
+                      uint3 group [[threadgroup_position_in_grid]],
+                      uint tid [[thread_index_in_threadgroup]],
+                      ushort sg [[simdgroup_index_in_threadgroup]],
+                      ushort lane [[thread_index_in_simdgroup]]) {
+    threadgroup half qs[BM * 64];
+    threadgroup half ks[BN * 64];
+    threadgroup half vs[BN * 64];
+    uint base = group.z * 1024 * 64;
+    uint qstart = group.y * BM;
+    for (uint i = tid; i < BM * 64; i += BM * 4)
+        qs[i] = half(Q[base + qstart * 64 + i]);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    simdgroup_half8x8 qa[8];
+    simdgroup_float8x8 out[8];
+    #pragma unroll
+    for (uint d = 0; d < 8; ++d) {
+        simdgroup_load(qa[d], qs + sg * 8 * 64 + d * 8, 64);
+        out[d] = simdgroup_float8x8(0.0f);
+    }
+    uint row = ((lane >> 1) & 3) + (lane >> 4) * 4;
+    uint col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
+    uint qrow = qstart + sg * 8 + row;
+    float m = -INFINITY, total = 0;
+    for (uint kb = 0; kb < qstart + BM; kb += BN) {
+        for (uint i = tid; i < BN * 64; i += BM * 4) {
+            ks[i] = half(K[base + kb * 64 + i]);
+            vs[i] = half(V[base + kb * 64 + i]);
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        simdgroup_float8x8 scores[BN / 8];
+        #pragma unroll
+        for (uint c = 0; c < BN / 8; ++c)
+            scores[c] = simdgroup_float8x8(0.0f);
+        #pragma unroll
+        for (uint d = 0; d < 8; ++d) {
+            #pragma unroll
+            for (uint c = 0; c < BN / 8; ++c) {
+                simdgroup_half8x8 kt;
+                simdgroup_load(kt, ks + c * 8 * 64 + d * 8, 64, ulong2(0), true);
+                simdgroup_multiply_accumulate(scores[c], qa[d], kt, scores[c]);
+            }
+        }
+        float mx = -INFINITY;
+        #pragma unroll
+        for (uint c = 0; c < BN / 8; ++c) {
+            #pragma unroll
+            for (uint j = 0; j < 2; ++j) {
+                float s = scores[c].thread_elements()[j] * 0.125f;
+                if (kb + c * 8 + col + j > qrow) s = -INFINITY;
+                scores[c].thread_elements()[j] = s;
+                mx = max(mx, s);
+            }
+        }
+        mx = max(mx, simd_shuffle_xor(mx, 1));
+        mx = max(mx, simd_shuffle_xor(mx, 8));
+        float nm = max(m, mx);
+        float correction = fast::exp(m - nm);
+        float sum = 0;
+        simdgroup_half8x8 probs[BN / 8];
+        #pragma unroll
+        for (uint c = 0; c < BN / 8; ++c) {
+            #pragma unroll
+            for (uint j = 0; j < 2; ++j) {
+                float p = fast::exp(scores[c].thread_elements()[j] - nm);
+                probs[c].thread_elements()[j] = half(p);
+                sum += p;
+            }
+        }
+        sum += simd_shuffle_xor(sum, 1);
+        sum += simd_shuffle_xor(sum, 8);
+        total = total * correction + sum;
+        m = nm;
+        #pragma unroll
+        for (uint d = 0; d < 8; ++d) {
+            out[d].thread_elements()[0] *= correction;
+            out[d].thread_elements()[1] *= correction;
+            #pragma unroll
+            for (uint c = 0; c < BN / 8; ++c) {
+                simdgroup_half8x8 vt;
+                simdgroup_load(vt, vs + c * 8 * 64 + d * 8, 64);
+                simdgroup_multiply_accumulate(out[d], probs[c], vt, out[d]);
+            }
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+    }
+    #pragma unroll
+    for (uint d = 0; d < 8; ++d) {
+        out[d].thread_elements()[0] /= total;
+        out[d].thread_elements()[1] /= total;
+        simdgroup_store(out[d], Y + base + (qstart + sg * 8) * 64 + d * 8, 64);
+    }
+}
+'''
+_LIB = torch.mps.compile_shader(_SOURCE.replace('__BM__', str(_BM)).replace('__BN__', str(_BN)))
+
 
 def kernel(q, k, v):
-    return F.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=True).float()
+    if q.shape != (8, 8, 1024, 64) or not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
+        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
+    y = torch.empty_like(q)
+    _LIB.attention(q, k, v, y, threads=(_BM * 4, 1024 // _BM, 64), group_size=(_BM * 4, 1, 1))
+    return y

```

</details>

`python bench.py --wait`  
→ attention: correct (relative error 2.0e-07). reference 72.17 ms, kernel 17.01 ms, speedup 4.24x

> PyTorch’s built-in causal attention passed the check and ran in 17.01 ms, a 4.24× speedup in the first benchmark. I’ll test half precision next and check its error against the allowed tolerance.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 6.81e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -2,7 +2,7 @@
 import torch
 import torch.nn.functional as F
 
-_BM = 32
+_BM = 64
 _BN = 64
 _SOURCE = r'''
 #include <metal_stdlib>

```

</details>

`python bench.py`  
→ run 2 of 10 submitted.  
→ attention: correct (relative error 8.6e-04). reference 68.43 ms, kernel 10.64 ms, speedup 6.43x

- `python -c 'import torch; s=\"#include <metal_stdlib>\\n#include <metal_simdgroup_matrix>\\nusing namespace metal;\\nkernel void probe(device float* o,uint t ...`
> Half precision also passed and reduced the time to 10.64 ms. I’ll now try a Metal kernel that computes attention in tiles, using half precision for matrix inputs and float32 for softmax and accumulation.

- `cat > kernel.py <<'PY` (exited 1 in 485ms:)
- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.31e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -3,7 +3,7 @@
 import torch.nn.functional as F
 
 _BM = 64
-_BN = 64
+_BN = 32
 _SOURCE = r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 58.53 ms, kernel 8.04 ms, speedup 7.28x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.90e-04 (allowed 2e-03)
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -2,8 +2,8 @@
 import torch
 import torch.nn.functional as F
 
-_BM = 64
-_BN = 32
+_BM = 128
+_BN = 64
 _SOURCE = r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>

```

</details>

`python bench.py`  
→ run 4 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 57.91 ms, kernel 7.04 ms, speedup 8.23x

> The Metal kernel passed and ran in 8.04 ms. Increasing its query tile from 32 to 64 rows improved that to 7.04 ms. I’m testing tile sizes to find the best balance between reuse and GPU occupancy.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.45e-04 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -2,8 +2,8 @@
 import torch
 import torch.nn.functional as F
 
-_BM = 128
-_BN = 64
+_BM = 64
+_BN = 32
 _SOURCE = r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
@@ -23,26 +23,32 @@
     threadgroup half vs[BN * 64];
     uint base = group.z * 1024 * 64;
     uint qstart = group.y * BM;
-    for (uint i = tid; i < BM * 64; i += BM * 4)
+    for (uint i = tid; i < BM * 64; i += BM * 2)
         qs[i] = half(Q[base + qstart * 64 + i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    simdgroup_half8x8 qa[8];
-    simdgroup_float8x8 out[8];
+    simdgroup_half8x8 qa[2][8];
+    simdgroup_float8x8 out[2][8];
+    #pragma unroll
+    for (uint r = 0; r < 2; ++r) {
     #pragma unroll
     for (uint d = 0; d < 8; ++d) {
-        simdgroup_load(qa[d], qs + sg * 8 * 64 + d * 8, 64);
-        out[d] = simdgroup_float8x8(0.0f);
+        simdgroup_load(qa[r][d], qs + (sg * 16 + r * 8) * 64 + d * 8, 64);
+        out[r][d] = simdgroup_float8x8(0.0f);
+    }
     }
     uint row = ((lane >> 1) & 3) + (lane >> 4) * 4;
     uint col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
-    uint qrow = qstart + sg * 8 + row;
-    float m = -INFINITY, total = 0;
+    float m[2] = {-INFINITY, -INFINITY};
+    float total[2] = {0, 0};
     for (uint kb = 0; kb < qstart + BM; kb += BN) {
-        for (uint i = tid; i < BN * 64; i += BM * 4) {
+        for (uint i = tid; i < BN * 64; i += BM * 2) {
             ks[i] = half(K[base + kb * 64 + i]);
             vs[i] = half(V[base + kb * 64 + i]);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
+        #pragma unroll
+        for (uint r = 0; r < 2; ++r) {
+        uint qrow = qstart + sg * 16 + r * 8 + row;
         simdgroup_float8x8 scores[BN / 8];
         #pragma unroll
         for (uint c = 0; c < BN / 8; ++c)
@@ -53,7 +59,7 @@
             for (uint c = 0; c < BN / 8; ++c) {
                 simdgroup_half8x8 kt;
                 simdgroup_load(kt, ks + c * 8 * 64 + d * 8, 64, ulong2(0), true);
-                simdgroup_multiply_accumulate(scores[c], qa[d], kt, scores[c]);
+                simdgroup_multiply_accumulate(scores[c], qa[r][d], kt, scores[c]);
             }
         }
         float mx = -INFINITY;
@@ -69,8 +75,8 @@
         }
         mx = max(mx, simd_shuffle_xor(mx, 1));
         mx = max(mx, simd_shuffle_xor(mx, 8));
-        float nm = max(m, mx);
-        float correction = fast::exp(m - nm);
+        float nm = max(m[r], mx);
+        float correction = fast::exp(m[r] - nm);
         float sum = 0;
         simdgroup_half8x8 probs[BN / 8];
         #pragma unroll
@@ -84,26 +90,30 @@
         }
         sum += simd_shuffle_xor(sum, 1);
         sum += simd_shuffle_xor(sum, 8);
-        total = total * correction + sum;
-        m = nm;
+        total[r] = total[r] * correction + sum;
+        m[r] = nm;
         #pragma unroll
         for (uint d = 0; d < 8; ++d) {
-            out[d].thread_elements()[0] *= correction;
-            out[d].thread_elements()[1] *= correction;
+            out[r][d].thread_elements()[0] *= correction;
+            out[r][d].thread_elements()[1] *= correction;
             #pragma unroll
             for (uint c = 0; c < BN / 8; ++c) {
                 simdgroup_half8x8 vt;
                 simdgroup_load(vt, vs + c * 8 * 64 + d * 8, 64);
-                simdgroup_multiply_accumulate(out[d], probs[c], vt, out[d]);
+                simdgroup_multiply_accumulate(out[r][d], probs[c], vt, out[r][d]);
             }
+        }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
     #pragma unroll
+    for (uint r = 0; r < 2; ++r) {
+    #pragma unroll
     for (uint d = 0; d < 8; ++d) {
-        out[d].thread_elements()[0] /= total;
-        out[d].thread_elements()[1] /= total;
-        simdgroup_store(out[d], Y + base + (qstart + sg * 8) * 64 + d * 8, 64);
+        out[r][d].thread_elements()[0] /= total[r];
+        out[r][d].thread_elements()[1] /= total[r];
+        simdgroup_store(out[r][d], Y + base + (qstart + sg * 16 + r * 8) * 64 + d * 8, 64);
+    }
     }
 }
 '''
@@ -114,5 +124,5 @@
     if q.shape != (8, 8, 1024, 64) or not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
         return F.scaled_dot_product_attention(q, k, v, is_causal=True)
     y = torch.empty_like(q)
-    _LIB.attention(q, k, v, y, threads=(_BM * 4, 1024 // _BM, 64), group_size=(_BM * 4, 1, 1))
+    _LIB.attention(q, k, v, y, threads=(_BM * 2, 1024 // _BM, 64), group_size=(_BM * 2, 1, 1))
     return y

```

</details>

`python bench.py`  
→ run 5 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 62.50 ms, kernel 7.53 ms, speedup 8.31x

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 6.43e-04 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -3,7 +3,7 @@
 import torch.nn.functional as F
 
 _BM = 64
-_BN = 32
+_BN = 64
 _SOURCE = r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
@@ -23,32 +23,26 @@
     threadgroup half vs[BN * 64];
     uint base = group.z * 1024 * 64;
     uint qstart = group.y * BM;
-    for (uint i = tid; i < BM * 64; i += BM * 2)
-        qs[i] = half(Q[base + qstart * 64 + i]);
+    for (uint i = tid; i < BM * 16; i += BM * 4)
+        reinterpret_cast<threadgroup half4*>(qs)[i] = half4(reinterpret_cast<device const float4*>(Q)[base / 4 + qstart * 16 + i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    simdgroup_half8x8 qa[2][8];
-    simdgroup_float8x8 out[2][8];
-    #pragma unroll
-    for (uint r = 0; r < 2; ++r) {
+    simdgroup_half8x8 qa[8];
+    simdgroup_float8x8 out[8];
     #pragma unroll
     for (uint d = 0; d < 8; ++d) {
-        simdgroup_load(qa[r][d], qs + (sg * 16 + r * 8) * 64 + d * 8, 64);
-        out[r][d] = simdgroup_float8x8(0.0f);
-    }
+        simdgroup_load(qa[d], qs + sg * 8 * 64 + d * 8, 64);
+        out[d] = simdgroup_float8x8(0.0f);
     }
     uint row = ((lane >> 1) & 3) + (lane >> 4) * 4;
     uint col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
-    float m[2] = {-INFINITY, -INFINITY};
-    float total[2] = {0, 0};
+    uint qrow = qstart + sg * 8 + row;
+    float m = -INFINITY, total = 0;
     for (uint kb = 0; kb < qstart + BM; kb += BN) {
-        for (uint i = tid; i < BN * 64; i += BM * 2) {
-            ks[i] = half(K[base + kb * 64 + i]);
-            vs[i] = half(V[base + kb * 64 + i]);
+        for (uint i = tid; i < BN * 16; i += BM * 4) {
+            reinterpret_cast<threadgroup half4*>(ks)[i] = half4(reinterpret_cast<device const float4*>(K)[base / 4 + kb * 16 + i]);
+            reinterpret_cast<threadgroup half4*>(vs)[i] = half4(reinterpret_cast<device const float4*>(V)[base / 4 + kb * 16 + i]);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        #pragma unroll
-        for (uint r = 0; r < 2; ++r) {
-        uint qrow = qstart + sg * 16 + r * 8 + row;
         simdgroup_float8x8 scores[BN / 8];
         #pragma unroll
         for (uint c = 0; c < BN / 8; ++c)
@@ -59,7 +53,7 @@
             for (uint c = 0; c < BN / 8; ++c) {
                 simdgroup_half8x8 kt;
                 simdgroup_load(kt, ks + c * 8 * 64 + d * 8, 64, ulong2(0), true);
-                simdgroup_multiply_accumulate(scores[c], qa[r][d], kt, scores[c]);
+                simdgroup_multiply_accumulate(scores[c], qa[d], kt, scores[c]);
             }
         }
         float mx = -INFINITY;
@@ -75,8 +69,8 @@
         }
         mx = max(mx, simd_shuffle_xor(mx, 1));
         mx = max(mx, simd_shuffle_xor(mx, 8));
-        float nm = max(m[r], mx);
-        float correction = fast::exp(m[r] - nm);
+        float nm = max(m, mx);
+        float correction = fast::exp(m - nm);
         float sum = 0;
         simdgroup_half8x8 probs[BN / 8];
         #pragma unroll
@@ -90,30 +84,26 @@
         }
         sum += simd_shuffle_xor(sum, 1);
         sum += simd_shuffle_xor(sum, 8);
-        total[r] = total[r] * correction + sum;
-        m[r] = nm;
+        total = total * correction + sum;
+        m = nm;
         #pragma unroll
         for (uint d = 0; d < 8; ++d) {
-            out[r][d].thread_elements()[0] *= correction;
-            out[r][d].thread_elements()[1] *= correction;
+            out[d].thread_elements()[0] *= correction;
+            out[d].thread_elements()[1] *= correction;
             #pragma unroll
             for (uint c = 0; c < BN / 8; ++c) {
                 simdgroup_half8x8 vt;
                 simdgroup_load(vt, vs + c * 8 * 64 + d * 8, 64);
-                simdgroup_multiply_accumulate(out[r][d], probs[c], vt, out[r][d]);
+                simdgroup_multiply_accumulate(out[d], probs[c], vt, out[d]);
             }
-        }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
     #pragma unroll
-    for (uint r = 0; r < 2; ++r) {
-    #pragma unroll
     for (uint d = 0; d < 8; ++d) {
-        out[r][d].thread_elements()[0] /= total[r];
-        out[r][d].thread_elements()[1] /= total[r];
-        simdgroup_store(out[r][d], Y + base + (qstart + sg * 16 + r * 8) * 64 + d * 8, 64);
-    }
+        out[d].thread_elements()[0] /= total;
+        out[d].thread_elements()[1] /= total;
+        simdgroup_store(out[d], Y + base + (qstart + sg * 8) * 64 + d * 8, 64);
     }
 }
 '''
@@ -124,5 +114,5 @@
     if q.shape != (8, 8, 1024, 64) or not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
         return F.scaled_dot_product_attention(q, k, v, is_causal=True)
     y = torch.empty_like(q)
-    _LIB.attention(q, k, v, y, threads=(_BM * 2, 1024 // _BM, 64), group_size=(_BM * 2, 1, 1))
+    _LIB.attention(q, k, v, y, threads=(_BM * 4, 1024 // _BM, 64), group_size=(_BM * 4, 1, 1))
     return y

```

</details>

`python bench.py`  
→ run 6 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 65.39 ms, kernel 10.61 ms, speedup 6.16x

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 5.30e-04 (allowed 2e-03)
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -11,8 +11,8 @@
 constant uint BM = __BM__;
 constant uint BN = __BN__;
 kernel void attention(device const float* Q [[buffer(0)]],
-                      device const float* K [[buffer(1)]],
-                      device const float* V [[buffer(2)]],
+                      device const half* K [[buffer(1)]],
+                      device const half* V [[buffer(2)]],
                       device float* Y [[buffer(3)]],
                       uint3 group [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
@@ -23,8 +23,8 @@
     threadgroup half vs[BN * 64];
     uint base = group.z * 1024 * 64;
     uint qstart = group.y * BM;
-    for (uint i = tid; i < BM * 16; i += BM * 4)
-        reinterpret_cast<threadgroup half4*>(qs)[i] = half4(reinterpret_cast<device const float4*>(Q)[base / 4 + qstart * 16 + i]);
+    for (uint i = tid; i < BM * 64; i += BM * 4)
+        qs[i] = half(Q[base + qstart * 64 + i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qa[8];
     simdgroup_float8x8 out[8];
@@ -38,9 +38,9 @@
     uint qrow = qstart + sg * 8 + row;
     float m = -INFINITY, total = 0;
     for (uint kb = 0; kb < qstart + BM; kb += BN) {
-        for (uint i = tid; i < BN * 16; i += BM * 4) {
-            reinterpret_cast<threadgroup half4*>(ks)[i] = half4(reinterpret_cast<device const float4*>(K)[base / 4 + kb * 16 + i]);
-            reinterpret_cast<threadgroup half4*>(vs)[i] = half4(reinterpret_cast<device const float4*>(V)[base / 4 + kb * 16 + i]);
+        for (uint i = tid; i < BN * 64; i += BM * 4) {
+            ks[i] = half(K[base + kb * 64 + i]);
+            vs[i] = half(V[base + kb * 64 + i]);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 scores[BN / 8];
@@ -114,5 +114,5 @@
     if q.shape != (8, 8, 1024, 64) or not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
         return F.scaled_dot_product_attention(q, k, v, is_causal=True)
     y = torch.empty_like(q)
-    _LIB.attention(q, k, v, y, threads=(_BM * 4, 1024 // _BM, 64), group_size=(_BM * 4, 1, 1))
+    _LIB.attention(q, k.half(), v.half(), y, threads=(_BM * 4, 1024 // _BM, 64), group_size=(_BM * 4, 1, 1))
     return y

```

</details>

`python bench.py`  
→ run 7 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 64.30 ms, kernel 8.55 ms, speedup 7.52x

> The 128-row tile and the version with more work per SIMD group were slower. The 64×64 tile remains fastest at 7.04 ms. I’m testing vectorized loads next to reduce data-loading instructions.

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.17e-04 (allowed 2e-03)
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -11,16 +11,17 @@
 constant uint BM = __BM__;
 constant uint BN = __BN__;
 kernel void attention(device const float* Q [[buffer(0)]],
-                      device const half* K [[buffer(1)]],
-                      device const half* V [[buffer(2)]],
+                      device const float* K [[buffer(1)]],
+                      device const float* V [[buffer(2)]],
                       device float* Y [[buffer(3)]],
                       uint3 group [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
-    threadgroup half qs[BM * 64];
-    threadgroup half ks[BN * 64];
-    threadgroup half vs[BN * 64];
+    threadgroup half storage[BN * 128];
+    threadgroup half* qs = storage;
+    threadgroup half* ks = storage;
+    threadgroup half* vs = storage + BN * 64;
     uint base = group.z * 1024 * 64;
     uint qstart = group.y * BM;
     for (uint i = tid; i < BM * 64; i += BM * 4)
@@ -33,6 +34,7 @@
         simdgroup_load(qa[d], qs + sg * 8 * 64 + d * 8, 64);
         out[d] = simdgroup_float8x8(0.0f);
     }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     uint row = ((lane >> 1) & 3) + (lane >> 4) * 4;
     uint col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
     uint qrow = qstart + sg * 8 + row;
@@ -114,5 +116,5 @@
     if q.shape != (8, 8, 1024, 64) or not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
         return F.scaled_dot_product_attention(q, k, v, is_causal=True)
     y = torch.empty_like(q)
-    _LIB.attention(q, k.half(), v.half(), y, threads=(_BM * 4, 1024 // _BM, 64), group_size=(_BM * 4, 1, 1))
+    _LIB.attention(q, k, v, y, threads=(_BM * 4, 1024 // _BM, 64), group_size=(_BM * 4, 1, 1))
     return y

```

</details>

`python bench.py`  
→ run 8 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 70.95 ms, kernel 7.43 ms, speedup 9.55x

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 1.48e-03 (allowed 2e-03)
- `python -c 'import torch; from kernel import kernel; from reference import inputs, reference; torch.set_grad_enabled(False); worst=0; errs=[]; `
> Half precision for output accumulation failed on some fresh inputs, so I’ve rejected it and will keep float32 accumulation. Vectorized loads measured 7.43 ms; the fastest version is still 7.04 ms. I have two benchmark runs left for testing input conversion and confirming the final choice.

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.38e-04 (allowed 2e-03)
### Bench run 11

`python bench.py`  
→ run 9 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 63.81 ms, kernel 8.80 ms, speedup 7.25x

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.49e-04 (allowed 2e-03)
### Bench run 12

`python bench.py`  
→ run 10 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 64.28 ms, kernel 7.77 ms, speedup 8.27x

> All ten benchmark runs are complete. The fastest version was the 64×64 Metal kernel with separate shared buffers and float32 accumulation, at 7.04 ms. I’ll restore it, update the lessons, and run the final correctness check.

- `cat kernel.py`
- edited `solution.py`
- edited `solution.py`
- `python -c 'import torch; from kernel import kernel; from reference import inputs, reference; torch.set_grad_enabled(False); errs=[];`
- `python check.py`  
→ OK: relative error 3.58e-04 (allowed 2e-03)
> Fastest measured: **7.04 ms, 8.23× speedup**. Final check passed with relative error **0.000358**, below 0.002. Updated LESSONS.md.
> 
> EDIT: kernel.py computes tiled causal attention in Metal with half precision matrix inputs and float32 accumulation.

