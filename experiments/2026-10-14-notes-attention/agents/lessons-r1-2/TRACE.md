# Agent `lessons-r1-2`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what is still true from the notes you were given. At most 300 words, in words, no code.

9 min, 93,156 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 16.928 ms | 4.24x |
| run2 | 10.57 ms | 6.35x |
| run3 | 6.946 ms | 10.13x |
| run4 | 5.803 ms | 10.24x |
| run5 | 5.259 ms | 11.15x |
| run6 | 4.99 ms | 12.01x |
| run7 | 7.414 ms | 8.89x |
| run8 | 5.982 ms | 11.36x |
| run9 | 6.056 ms | 12.73x |
| run10 | 6.394 ms | 10.83x |
| final | 5.71 ms | 12.09x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the task files and earlier lessons, then test faster attention kernels within the 10-run timing budget.

- `pwd && rg --files -g '"'!__pycache__/**`
### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -1,10 +1,6 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
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

- `cat > kernel.py <<'PY` (exited 1 in 521ms:)
- edited `solution.py`
- `python check.py`  
→ OK: relative error 9.39e-08 (allowed 2e-03)
### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -3,4 +3,4 @@
 
 
 def kernel(q, k, v):
-    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
+    return F.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=True).float()

```

</details>

`python bench.py`  
→         >>> x = torch.zeros(16, device="mps")  
→         >>> lib.full(x, 3.14)

- `python -c 'import torch; l=torch.mps.compile_shader(\"kernel void full(device float* out, uint idx [[thread_position_in_grid]]) { out[idx] = 0; }\"); print(d...`
> PyTorch’s built-in causal attention passed the check and measured 16.93 ms, a 4.24× speedup. I’ll next test half-precision matrix operations and a fused Metal kernel against the accuracy tolerance.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 6.81e-04 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -1,6 +1,81 @@
 import torch
 import torch.nn.functional as F
 
+_source = r'''
+#include <metal_stdlib>
+#include <metal_simdgroup_matrix>
+using namespace metal;
+
+kernel void attention(device const half* Q, device const half* K,
+                      device const half* V, device float* O,
+                      uint3 group [[threadgroup_position_in_grid]],
+                      uint tid [[thread_index_in_threadgroup]],
+                      uint lane [[thread_index_in_simdgroup]],
+                      uint sg [[simdgroup_index_in_threadgroup]]) {
+    const uint start = group.x * 32;
+    const uint base = group.y * 1024 * 64;
+    Q += base; K += base; V += base; O += base;
+    threadgroup float scores[32*32];
+    threadgroup half probs[32*32];
+    threadgroup float oldscale[32];
+    threadgroup float maxima[32];
+    threadgroup float sums[32];
+    threadgroup float result[32*64];
+    for (uint i=tid; i<32*64; i+=256) result[i]=0;
+    if (tid<32) { maxima[tid]=-INFINITY; sums[tid]=0; }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    for (uint key=0; key<start+32; key+=32) {
+        for (uint rt=0; rt<2; ++rt) {
+            uint r=(sg/4+rt*2)*8, c=(sg%4)*8;
+            simdgroup_float8x8 acc(0.0f);
+            for(uint d=0;d<64;d+=8) {
+                simdgroup_half8x8 a,b;
+                simdgroup_load(a,Q+(start+r)*64+d,64);
+                simdgroup_load(b,K+(key+c)*64+d,64,ulong2(0),true);
+                simdgroup_multiply_accumulate(acc,a,b,acc);
+            }
+            simdgroup_store(acc,scores+r*32+c,32);
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for(uint ri=0;ri<4;++ri) {
+            uint r=sg*4+ri;
+            float s=key+lane<=start+r ? scores[r*32+lane]*0.125f : -INFINITY;
+            float m=max(maxima[r],simd_max(s));
+            float f=fast::exp(maxima[r]-m);
+            float p=fast::exp(s-m);
+            float z=simd_sum(p);
+            probs[r*32+lane]=half(p);
+            if(lane==0) { maxima[r]=m; sums[r]=sums[r]*f+z; oldscale[r]=f; }
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for(uint i=tid;i<32*64;i+=256) result[i]*=oldscale[i/64];
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for(uint rt=0;rt<2;++rt) {
+            uint r=(sg/4+rt*2)*8;
+            for(uint ct=0;ct<2;++ct) {
+                uint c=(sg%4+ct*4)*8;
+                simdgroup_float8x8 acc;
+                simdgroup_load(acc,result+r*64+c,64);
+                for(uint d=0;d<32;d+=8) {
+                    simdgroup_half8x8 a,b;
+                    simdgroup_load(a,probs+r*32+d,32);
+                    simdgroup_load(b,V+(key+d)*64+c,64);
+                    simdgroup_multiply_accumulate(acc,a,b,acc);
+                }
+                simdgroup_store(acc,result+r*64+c,64);
+            }
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+    }
+    for(uint i=tid;i<32*64;i+=256) O[start*64+i]=result[i]/sums[i/64];
+}
+'''
+_lib = torch.mps.compile_shader(_source)
 
 def kernel(q, k, v):
-    return F.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=True).float()
+    if q.shape != (8, 8, 1024, 64):
+        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
+    out = torch.empty_like(q)
+    _lib.attention(q.half(), k.half(), v.half(), out,
+                   threads=(32*256,64,1), group_size=(256,1,1))
+    return out

```

</details>

`python bench.py`  
→ run 2 of 10 submitted.  
→ attention: correct (relative error 8.6e-04). reference 67.15 ms, kernel 10.57 ms, speedup 6.35x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.03e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -15,8 +15,8 @@
     const uint start = group.x * 32;
     const uint base = group.y * 1024 * 64;
     Q += base; K += base; V += base; O += base;
-    threadgroup float scores[32*32];
-    threadgroup half probs[32*32];
+    threadgroup float scores[32*64];
+    threadgroup half probs[32*64];
     threadgroup float oldscale[32];
     threadgroup float maxima[32];
     threadgroup float sums[32];
@@ -24,9 +24,10 @@
     for (uint i=tid; i<32*64; i+=256) result[i]=0;
     if (tid<32) { maxima[tid]=-INFINITY; sums[tid]=0; }
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    for (uint key=0; key<start+32; key+=32) {
+    for (uint key=0; key<start+32; key+=64) {
         for (uint rt=0; rt<2; ++rt) {
-            uint r=(sg/4+rt*2)*8, c=(sg%4)*8;
+          for (uint ct=0; ct<2; ++ct) {
+            uint r=(sg/4+rt*2)*8, c=(sg%4+ct*4)*8;
             simdgroup_float8x8 acc(0.0f);
             for(uint d=0;d<64;d+=8) {
                 simdgroup_half8x8 a,b;
@@ -34,17 +35,21 @@
                 simdgroup_load(b,K+(key+c)*64+d,64,ulong2(0),true);
                 simdgroup_multiply_accumulate(acc,a,b,acc);
             }
-            simdgroup_store(acc,scores+r*32+c,32);
+            simdgroup_store(acc,scores+r*64+c,64);
+          }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for(uint ri=0;ri<4;++ri) {
             uint r=sg*4+ri;
-            float s=key+lane<=start+r ? scores[r*32+lane]*0.125f : -INFINITY;
-            float m=max(maxima[r],simd_max(s));
+            float s=key+lane<=start+r ? scores[r*64+lane]*0.125f : -INFINITY;
+            float s2=key+lane+32<=start+r ? scores[r*64+lane+32]*0.125f : -INFINITY;
+            float m=max(maxima[r],simd_max(max(s,s2)));
             float f=fast::exp(maxima[r]-m);
             float p=fast::exp(s-m);
-            float z=simd_sum(p);
-            probs[r*32+lane]=half(p);
+            float p2=fast::exp(s2-m);
+            float z=simd_sum(p+p2);
+            probs[r*64+lane]=half(p);
+            probs[r*64+lane+32]=half(p2);
             if(lane==0) { maxima[r]=m; sums[r]=sums[r]*f+z; oldscale[r]=f; }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
@@ -56,9 +61,9 @@
                 uint c=(sg%4+ct*4)*8;
                 simdgroup_float8x8 acc;
                 simdgroup_load(acc,result+r*64+c,64);
-                for(uint d=0;d<32;d+=8) {
+                for(uint d=0;d<64;d+=8) {
                     simdgroup_half8x8 a,b;
-                    simdgroup_load(a,probs+r*32+d,32);
+                    simdgroup_load(a,probs+r*64+d,64);
                     simdgroup_load(b,V+(key+d)*64+c,64);
                     simdgroup_multiply_accumulate(acc,a,b,acc);
                 }

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 70.37 ms, kernel 6.95 ms, speedup 10.13x

> Half-precision built-in attention measured 10.57 ms and passed. The first custom Metal kernel also passed, with a smaller error; it computes attention in tiles without creating the full attention matrix. I’m measuring that version and testing larger tiles to reduce synchronization.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.86e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -20,8 +20,9 @@
     threadgroup float oldscale[32];
     threadgroup float maxima[32];
     threadgroup float sums[32];
-    threadgroup float result[32*64];
-    for (uint i=tid; i<32*64; i+=256) result[i]=0;
+    simdgroup_float8x8 results[4];
+    for(uint i=0;i<4;++i) results[i]=simdgroup_float8x8(0.0f);
+    const uint row=(lane/2)%4+(lane/16)*4;
     if (tid<32) { maxima[tid]=-INFINITY; sums[tid]=0; }
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for (uint key=0; key<start+32; key+=64) {
@@ -53,26 +54,32 @@
             if(lane==0) { maxima[r]=m; sums[r]=sums[r]*f+z; oldscale[r]=f; }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for(uint i=tid;i<32*64;i+=256) result[i]*=oldscale[i/64];
-        threadgroup_barrier(mem_flags::mem_threadgroup);
         for(uint rt=0;rt<2;++rt) {
             uint r=(sg/4+rt*2)*8;
             for(uint ct=0;ct<2;++ct) {
                 uint c=(sg%4+ct*4)*8;
-                simdgroup_float8x8 acc;
-                simdgroup_load(acc,result+r*64+c,64);
+                uint ix=rt*2+ct;
+                results[ix].thread_elements()[0]*=oldscale[r+row];
+                results[ix].thread_elements()[1]*=oldscale[r+row];
                 for(uint d=0;d<64;d+=8) {
                     simdgroup_half8x8 a,b;
                     simdgroup_load(a,probs+r*64+d,64);
                     simdgroup_load(b,V+(key+d)*64+c,64);
-                    simdgroup_multiply_accumulate(acc,a,b,acc);
+                    simdgroup_multiply_accumulate(results[ix],a,b,results[ix]);
                 }
-                simdgroup_store(acc,result+r*64+c,64);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
-    for(uint i=tid;i<32*64;i+=256) O[start*64+i]=result[i]/sums[i/64];
+    for(uint rt=0;rt<2;++rt) {
+        uint r=(sg/4+rt*2)*8;
+        for(uint ct=0;ct<2;++ct) {
+            uint c=(sg%4+ct*4)*8, ix=rt*2+ct;
+            results[ix].thread_elements()[0]/=sums[r+row];
+            results[ix].thread_elements()[1]/=sums[r+row];
+            simdgroup_store(results[ix],O+(start+r)*64+c,64);
+        }
+    }
 }
 '''
 _lib = torch.mps.compile_shader(_source)

```

</details>

`python bench.py`  
→ run 4 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 59.41 ms, kernel 5.80 ms, speedup 10.24x

- `python -c 'import torch; lib=torch.mps.compile_shader(\"#include <metal_stdlib>\\n#include <metal_simdgroup_matrix>\\nusing namespace metal;\\nkernel void ma...`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.76e-04 (allowed 2e-03)
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -12,21 +12,21 @@
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
-    const uint start = group.x * 32;
+    const uint start = group.x * 64;
     const uint base = group.y * 1024 * 64;
     Q += base; K += base; V += base; O += base;
-    threadgroup float scores[32*64];
-    threadgroup half probs[32*64];
-    threadgroup float oldscale[32];
-    threadgroup float maxima[32];
-    threadgroup float sums[32];
-    simdgroup_float8x8 results[4];
-    for(uint i=0;i<4;++i) results[i]=simdgroup_float8x8(0.0f);
+    threadgroup float scores[64*64];
+    threadgroup half probs[64*64];
+    threadgroup float oldscale[64];
+    threadgroup float maxima[64];
+    threadgroup float sums[64];
+    simdgroup_float8x8 results[8];
+    for(uint i=0;i<8;++i) results[i]=simdgroup_float8x8(0.0f);
     const uint row=(lane/2)%4+(lane/16)*4;
-    if (tid<32) { maxima[tid]=-INFINITY; sums[tid]=0; }
+    if (tid<64) { maxima[tid]=-INFINITY; sums[tid]=0; }
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    for (uint key=0; key<start+32; key+=64) {
-        for (uint rt=0; rt<2; ++rt) {
+    for (uint key=0; key<start+64; key+=64) {
+        for (uint rt=0; rt<4; ++rt) {
           for (uint ct=0; ct<2; ++ct) {
             uint r=(sg/4+rt*2)*8, c=(sg%4+ct*4)*8;
             simdgroup_float8x8 acc(0.0f);
@@ -40,8 +40,8 @@
           }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for(uint ri=0;ri<4;++ri) {
-            uint r=sg*4+ri;
+        for(uint ri=0;ri<8;++ri) {
+            uint r=sg*8+ri;
             float s=key+lane<=start+r ? scores[r*64+lane]*0.125f : -INFINITY;
             float s2=key+lane+32<=start+r ? scores[r*64+lane+32]*0.125f : -INFINITY;
             float m=max(maxima[r],simd_max(max(s,s2)));
@@ -54,7 +54,7 @@
             if(lane==0) { maxima[r]=m; sums[r]=sums[r]*f+z; oldscale[r]=f; }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for(uint rt=0;rt<2;++rt) {
+        for(uint rt=0;rt<4;++rt) {
             uint r=(sg/4+rt*2)*8;
             for(uint ct=0;ct<2;++ct) {
                 uint c=(sg%4+ct*4)*8;
@@ -71,7 +71,7 @@
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
-    for(uint rt=0;rt<2;++rt) {
+    for(uint rt=0;rt<4;++rt) {
         uint r=(sg/4+rt*2)*8;
         for(uint ct=0;ct<2;++ct) {
             uint c=(sg%4+ct*4)*8, ix=rt*2+ct;
@@ -89,5 +89,5 @@
         return F.scaled_dot_product_attention(q, k, v, is_causal=True)
     out = torch.empty_like(q)
     _lib.attention(q.half(), k.half(), v.half(), out,
-                   threads=(32*256,64,1), group_size=(256,1,1))
+                   threads=(16*256,64,1), group_size=(256,1,1))
     return out

```

</details>

`python bench.py`  
→ run 5 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 58.65 ms, kernel 5.26 ms, speedup 11.15x

> The custom kernel measured 6.95 ms with 32-key tiles and 5.80 ms with 64-key tiles. Both passed. I’m testing keeping partial outputs in GPU registers, then increasing the query tile size to reuse more key and value data.

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.59e-04 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -26,18 +26,26 @@
     if (tid<64) { maxima[tid]=-INFINITY; sums[tid]=0; }
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for (uint key=0; key<start+64; key+=64) {
-        for (uint rt=0; rt<4; ++rt) {
-          for (uint ct=0; ct<2; ++ct) {
+        simdgroup_float8x8 sc[8];
+        for(uint i=0;i<8;++i) sc[i]=simdgroup_float8x8(0.0f);
+        for(uint d=0;d<64;d+=8) {
+            simdgroup_half8x8 a[4], b[2];
+            for(uint rt=0;rt<4;++rt) {
+                uint r=(sg/4+rt*2)*8;
+                simdgroup_load(a[rt],Q+(start+r)*64+d,64);
+            }
+            for(uint ct=0;ct<2;++ct) {
+                uint c=(sg%4+ct*4)*8;
+                simdgroup_load(b[ct],K+(key+c)*64+d,64,ulong2(0),true);
+            }
+            for(uint rt=0;rt<4;++rt) for(uint ct=0;ct<2;++ct) {
+                uint ix=rt*2+ct;
+                simdgroup_multiply_accumulate(sc[ix],a[rt],b[ct],sc[ix]);
+            }
+        }
+        for(uint rt=0;rt<4;++rt) for(uint ct=0;ct<2;++ct) {
             uint r=(sg/4+rt*2)*8, c=(sg%4+ct*4)*8;
-            simdgroup_float8x8 acc(0.0f);
-            for(uint d=0;d<64;d+=8) {
-                simdgroup_half8x8 a,b;
-                simdgroup_load(a,Q+(start+r)*64+d,64);
-                simdgroup_load(b,K+(key+c)*64+d,64,ulong2(0),true);
-                simdgroup_multiply_accumulate(acc,a,b,acc);
-            }
-            simdgroup_store(acc,scores+r*64+c,64);
-          }
+            simdgroup_store(sc[rt*2+ct],scores+r*64+c,64);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for(uint ri=0;ri<8;++ri) {
@@ -57,16 +65,24 @@
         for(uint rt=0;rt<4;++rt) {
             uint r=(sg/4+rt*2)*8;
             for(uint ct=0;ct<2;++ct) {
-                uint c=(sg%4+ct*4)*8;
                 uint ix=rt*2+ct;
                 results[ix].thread_elements()[0]*=oldscale[r+row];
                 results[ix].thread_elements()[1]*=oldscale[r+row];
-                for(uint d=0;d<64;d+=8) {
-                    simdgroup_half8x8 a,b;
-                    simdgroup_load(a,probs+r*64+d,64);
-                    simdgroup_load(b,V+(key+d)*64+c,64);
-                    simdgroup_multiply_accumulate(results[ix],a,b,results[ix]);
-                }
+            }
+        }
+        for(uint d=0;d<64;d+=8) {
+            simdgroup_half8x8 a[4], b[2];
+            for(uint rt=0;rt<4;++rt) {
+                uint r=(sg/4+rt*2)*8;
+                simdgroup_load(a[rt],probs+r*64+d,64);
+            }
+            for(uint ct=0;ct<2;++ct) {
+                uint c=(sg%4+ct*4)*8;
+                simdgroup_load(b[ct],V+(key+d)*64+c,64);
+            }
+            for(uint rt=0;rt<4;++rt) for(uint ct=0;ct<2;++ct) {
+                uint ix=rt*2+ct;
+                simdgroup_multiply_accumulate(results[ix],a[rt],b[ct],results[ix]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);

```

</details>

`python bench.py`  
→ run 6 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 59.93 ms, kernel 4.99 ms, speedup 12.01x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.27e-04 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -15,8 +15,8 @@
     const uint start = group.x * 64;
     const uint base = group.y * 1024 * 64;
     Q += base; K += base; V += base; O += base;
-    threadgroup float scores[64*64];
-    threadgroup half probs[64*64];
+    threadgroup half scores[64*128];
+    threadgroup half* probs = scores;
     threadgroup float oldscale[64];
     threadgroup float maxima[64];
     threadgroup float sums[64];
@@ -25,64 +25,57 @@
     const uint row=(lane/2)%4+(lane/16)*4;
     if (tid<64) { maxima[tid]=-INFINITY; sums[tid]=0; }
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    for (uint key=0; key<start+64; key+=64) {
-        simdgroup_float8x8 sc[8];
-        for(uint i=0;i<8;++i) sc[i]=simdgroup_float8x8(0.0f);
-        for(uint d=0;d<64;d+=8) {
-            simdgroup_half8x8 a[4], b[2];
-            for(uint rt=0;rt<4;++rt) {
-                uint r=(sg/4+rt*2)*8;
-                simdgroup_load(a[rt],Q+(start+r)*64+d,64);
+    for (uint key=0; key<start+64; key+=128) {
+        for (uint rt=0; rt<4; ++rt) {
+          for (uint ct=0; ct<4; ++ct) {
+            uint r=(sg/4+rt*2)*8, c=(sg%4+ct*4)*8;
+            simdgroup_float8x8 acc(0.0f);
+            for(uint d=0;d<64;d+=8) {
+                simdgroup_half8x8 a,b;
+                simdgroup_load(a,Q+(start+r)*64+d,64);
+                simdgroup_load(b,K+(key+c)*64+d,64,ulong2(0),true);
+                simdgroup_multiply_accumulate(acc,a,b,acc);
             }
-            for(uint ct=0;ct<2;++ct) {
-                uint c=(sg%4+ct*4)*8;
-                simdgroup_load(b[ct],K+(key+c)*64+d,64,ulong2(0),true);
-            }
-            for(uint rt=0;rt<4;++rt) for(uint ct=0;ct<2;++ct) {
-                uint ix=rt*2+ct;
-                simdgroup_multiply_accumulate(sc[ix],a[rt],b[ct],sc[ix]);
-            }
-        }
-        for(uint rt=0;rt<4;++rt) for(uint ct=0;ct<2;++ct) {
-            uint r=(sg/4+rt*2)*8, c=(sg%4+ct*4)*8;
-            simdgroup_store(sc[rt*2+ct],scores+r*64+c,64);
+            simdgroup_half8x8 hs;
+            hs.thread_elements()[0]=half(acc.thread_elements()[0]);
+            hs.thread_elements()[1]=half(acc.thread_elements()[1]);
+            simdgroup_store(hs,scores+r*128+c,128);
+          }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for(uint ri=0;ri<8;++ri) {
             uint r=sg*8+ri;
-            float s=key+lane<=start+r ? scores[r*64+lane]*0.125f : -INFINITY;
-            float s2=key+lane+32<=start+r ? scores[r*64+lane+32]*0.125f : -INFINITY;
-            float m=max(maxima[r],simd_max(max(s,s2)));
+            float s=key+lane<=start+r ? float(scores[r*128+lane])*0.125f : -INFINITY;
+            float s2=key+lane+32<=start+r ? float(scores[r*128+lane+32])*0.125f : -INFINITY;
+            float s3=key+lane+64<=start+r ? float(scores[r*128+lane+64])*0.125f : -INFINITY;
+            float s4=key+lane+96<=start+r ? float(scores[r*128+lane+96])*0.125f : -INFINITY;
+            float m=max(maxima[r],simd_max(max(max(s,s2),max(s3,s4))));
             float f=fast::exp(maxima[r]-m);
             float p=fast::exp(s-m);
             float p2=fast::exp(s2-m);
-            float z=simd_sum(p+p2);
-            probs[r*64+lane]=half(p);
-            probs[r*64+lane+32]=half(p2);
+            float p3=fast::exp(s3-m);
+            float p4=fast::exp(s4-m);
+            float z=simd_sum(p+p2+p3+p4);
+            probs[r*128+lane]=half(p);
+            probs[r*128+lane+32]=half(p2);
+            probs[r*128+lane+64]=half(p3);
+            probs[r*128+lane+96]=half(p4);
             if(lane==0) { maxima[r]=m; sums[r]=sums[r]*f+z; oldscale[r]=f; }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for(uint rt=0;rt<4;++rt) {
             uint r=(sg/4+rt*2)*8;
             for(uint ct=0;ct<2;++ct) {
+                uint c=(sg%4+ct*4)*8;
                 uint ix=rt*2+ct;
                 results[ix].thread_elements()[0]*=oldscale[r+row];
                 results[ix].thread_elements()[1]*=oldscale[r+row];
-            }
-        }
-        for(uint d=0;d<64;d+=8) {
-            simdgroup_half8x8 a[4], b[2];
-            for(uint rt=0;rt<4;++rt) {
-                uint r=(sg/4+rt*2)*8;
-                simdgroup_load(a[rt],probs+r*64+d,64);
-            }
-            for(uint ct=0;ct<2;++ct) {
-                uint c=(sg%4+ct*4)*8;
-                simdgroup_load(b[ct],V+(key+d)*64+c,64);
-            }
-            for(uint rt=0;rt<4;++rt) for(uint ct=0;ct<2;++ct) {
-                uint ix=rt*2+ct;
-                simdgroup_multiply_accumulate(results[ix],a[rt],b[ct],results[ix]);
+                for(uint d=0;d<128;d+=8) {
+                    simdgroup_half8x8 a,b;
+                    simdgroup_load(a,probs+r*128+d,128);
+                    simdgroup_load(b,V+(key+d)*64+c,64);
+                    simdgroup_multiply_accumulate(results[ix],a,b,results[ix]);
+                }
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);

```

</details>

`python bench.py`  
→ run 7 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 65.91 ms, kernel 7.41 ms, speedup 8.89x

> The 64-query tile measured 4.99 ms. A version that shared more matrix loads slowed to 7.41 ms, so I’m reverting it. With three timing runs left, I’ll test larger tiles using half-precision score storage and keep the fastest passing version.

- `cp queue/run6.py kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.65e-04 (allowed 2e-03)
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -15,7 +15,7 @@
     const uint start = group.x * 64;
     const uint base = group.y * 1024 * 64;
     Q += base; K += base; V += base; O += base;
-    threadgroup half scores[64*128];
+    threadgroup half scores[64*64];
     threadgroup half* probs = scores;
     threadgroup float oldscale[64];
     threadgroup float maxima[64];
@@ -25,9 +25,9 @@
     const uint row=(lane/2)%4+(lane/16)*4;
     if (tid<64) { maxima[tid]=-INFINITY; sums[tid]=0; }
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    for (uint key=0; key<start+64; key+=128) {
+    for (uint key=0; key<start+64; key+=64) {
         for (uint rt=0; rt<4; ++rt) {
-          for (uint ct=0; ct<4; ++ct) {
+          for (uint ct=0; ct<2; ++ct) {
             uint r=(sg/4+rt*2)*8, c=(sg%4+ct*4)*8;
             simdgroup_float8x8 acc(0.0f);
             for(uint d=0;d<64;d+=8) {
@@ -39,27 +39,21 @@
             simdgroup_half8x8 hs;
             hs.thread_elements()[0]=half(acc.thread_elements()[0]);
             hs.thread_elements()[1]=half(acc.thread_elements()[1]);
-            simdgroup_store(hs,scores+r*128+c,128);
+            simdgroup_store(hs,scores+r*64+c,64);
           }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for(uint ri=0;ri<8;++ri) {
             uint r=sg*8+ri;
-            float s=key+lane<=start+r ? float(scores[r*128+lane])*0.125f : -INFINITY;
-            float s2=key+lane+32<=start+r ? float(scores[r*128+lane+32])*0.125f : -INFINITY;
-            float s3=key+lane+64<=start+r ? float(scores[r*128+lane+64])*0.125f : -INFINITY;
-            float s4=key+lane+96<=start+r ? float(scores[r*128+lane+96])*0.125f : -INFINITY;
-            float m=max(maxima[r],simd_max(max(max(s,s2),max(s3,s4))));
+            float s=key+lane<=start+r ? float(scores[r*64+lane])*0.125f : -INFINITY;
+            float s2=key+lane+32<=start+r ? float(scores[r*64+lane+32])*0.125f : -INFINITY;
+            float m=max(maxima[r],simd_max(max(s,s2)));
             float f=fast::exp(maxima[r]-m);
             float p=fast::exp(s-m);
             float p2=fast::exp(s2-m);
-            float p3=fast::exp(s3-m);
-            float p4=fast::exp(s4-m);
-            float z=simd_sum(p+p2+p3+p4);
-            probs[r*128+lane]=half(p);
-            probs[r*128+lane+32]=half(p2);
-            probs[r*128+lane+64]=half(p3);
-            probs[r*128+lane+96]=half(p4);
+            float z=simd_sum(p+p2);
+            probs[r*64+lane]=half(p);
+            probs[r*64+lane+32]=half(p2);
             if(lane==0) { maxima[r]=m; sums[r]=sums[r]*f+z; oldscale[r]=f; }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
@@ -70,9 +64,9 @@
                 uint ix=rt*2+ct;
                 results[ix].thread_elements()[0]*=oldscale[r+row];
                 results[ix].thread_elements()[1]*=oldscale[r+row];
-                for(uint d=0;d<128;d+=8) {
+                for(uint d=0;d<64;d+=8) {
                     simdgroup_half8x8 a,b;
-                    simdgroup_load(a,probs+r*128+d,128);
+                    simdgroup_load(a,probs+r*64+d,64);
                     simdgroup_load(b,V+(key+d)*64+c,64);
                     simdgroup_multiply_accumulate(results[ix],a,b,results[ix]);
                 }

```

</details>

`python bench.py`  
→ run 8 of 10 submitted.  
→ attention: correct (relative error 6.8e-04). reference 67.97 ms, kernel 5.98 ms, speedup 11.36x

- `cp queue/run6.py kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 5.10e-04 (allowed 2e-03)
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -12,21 +12,21 @@
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
-    const uint start = group.x * 64;
+    const uint start = group.x * 128;
     const uint base = group.y * 1024 * 64;
     Q += base; K += base; V += base; O += base;
-    threadgroup half scores[64*64];
+    threadgroup half scores[128*64];
     threadgroup half* probs = scores;
-    threadgroup float oldscale[64];
-    threadgroup float maxima[64];
-    threadgroup float sums[64];
-    simdgroup_float8x8 results[8];
-    for(uint i=0;i<8;++i) results[i]=simdgroup_float8x8(0.0f);
+    threadgroup float oldscale[128];
+    threadgroup float maxima[128];
+    threadgroup float sums[128];
+    simdgroup_float8x8 results[16];
+    for(uint i=0;i<16;++i) results[i]=simdgroup_float8x8(0.0f);
     const uint row=(lane/2)%4+(lane/16)*4;
-    if (tid<64) { maxima[tid]=-INFINITY; sums[tid]=0; }
+    if (tid<128) { maxima[tid]=-INFINITY; sums[tid]=0; }
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    for (uint key=0; key<start+64; key+=64) {
-        for (uint rt=0; rt<4; ++rt) {
+    for (uint key=0; key<start+128; key+=64) {
+        for (uint rt=0; rt<8; ++rt) {
           for (uint ct=0; ct<2; ++ct) {
             uint r=(sg/4+rt*2)*8, c=(sg%4+ct*4)*8;
             simdgroup_float8x8 acc(0.0f);
@@ -43,8 +43,8 @@
           }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for(uint ri=0;ri<8;++ri) {
-            uint r=sg*8+ri;
+        for(uint ri=0;ri<16;++ri) {
+            uint r=sg*16+ri;
             float s=key+lane<=start+r ? float(scores[r*64+lane])*0.125f : -INFINITY;
             float s2=key+lane+32<=start+r ? float(scores[r*64+lane+32])*0.125f : -INFINITY;
             float m=max(maxima[r],simd_max(max(s,s2)));
@@ -57,7 +57,7 @@
             if(lane==0) { maxima[r]=m; sums[r]=sums[r]*f+z; oldscale[r]=f; }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for(uint rt=0;rt<4;++rt) {
+        for(uint rt=0;rt<8;++rt) {
             uint r=(sg/4+rt*2)*8;
             for(uint ct=0;ct<2;++ct) {
                 uint c=(sg%4+ct*4)*8;
@@ -74,7 +74,7 @@
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
-    for(uint rt=0;rt<4;++rt) {
+    for(uint rt=0;rt<8;++rt) {
         uint r=(sg/4+rt*2)*8;
         for(uint ct=0;ct<2;++ct) {
             uint c=(sg%4+ct*4)*8, ix=rt*2+ct;
@@ -92,5 +92,5 @@
         return F.scaled_dot_product_attention(q, k, v, is_causal=True)
     out = torch.empty_like(q)
     _lib.attention(q.half(), k.half(), v.half(), out,
-                   threads=(16*256,64,1), group_size=(256,1,1))
+                   threads=(8*256,64,1), group_size=(256,1,1))
     return out

```

</details>

`python bench.py`  
→ run 9 of 10 submitted.  
→ attention: correct (relative error 6.8e-04). reference 77.12 ms, kernel 6.06 ms, speedup 12.74x (the GPU was busy during this run)

> The 128-key tile passed but measured 5.98 ms, slower than the best 4.99 ms result. The half-score 64-key version measured 6.06 ms, though the benchmark reported a busy GPU. I’ll use the final timing run to test a 128-query tile.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.62e-04 (allowed 2e-03)
### Bench run 11

`python bench.py`  
→ run 10 of 10 submitted.  
→ attention: correct (relative error 6.8e-04). reference 69.26 ms, kernel 6.39 ms, speedup 10.83x

- `rg . results`
> The 128-query tile measured 6.39 ms, so I’m restoring the fastest version: 64-query by 64-key tiles with float32 scores and accumulation. It measured 4.99 ms, a 12.01× speedup. I’ll record the results in LESSONS.md and run the final correctness check.

- `cp queue/run6.py kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.35e-04 (allowed 2e-03)
- edited `solution.py`
- `wc -w LESSONS.md`
> Fastest measured: **4.99 ms, 12.01× speedup**. Final check passed with relative error **0.000435** against the **0.002** tolerance. Updated LESSONS.md with all ten results.
> 
> EDIT: Final kernel.py computes tiled causal attention in Metal using half-precision matrix operands and float32 scores, accumulation, softmax state, and output.

