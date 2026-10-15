# Agent `lessons-r3-4`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what is still true from the notes you were given. At most 300 words, in words, no code.

13 min, 76,186 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 6.429 ms | 10.4x |
| run2 | 7.241 ms | 9.49x |
| run3 | 3.744 ms | 17.76x |
| run4 | 3.874 ms | 18.31x |
| run5 | 3.967 ms | 17.79x |
| run6 | 4.12 ms | 17.4x |
| run7 | 3.896 ms | 17.66x |
| run8 | 3.651 ms | 18.71x |
| run9 | 3.941 ms | 17.87x |
| run10 | 3.971 ms | 17.8x |
| final | 3.803 ms | 17.71x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the reference and prior lessons, then tune the attention kernel within the ten benchmark runs and check the final version.

### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -1,10 +1,114 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
 import torch
 
 
+def _source():
+    return r'''
+#include <metal_stdlib>
+#include <metal_simdgroup_matrix>
+using namespace metal;
+kernel void attention(device const float4* Q [[buffer(0)]],
+                      device const float4* K [[buffer(1)]],
+                      device const float4* V [[buffer(2)]],
+                      device float* O [[buffer(3)]],
+                      uint3 group [[threadgroup_position_in_grid]],
+                      uint tid [[thread_index_in_threadgroup]],
+                      uint sg [[simdgroup_index_in_threadgroup]],
+                      uint lane [[thread_index_in_simdgroup]]) {
+    const uint base = group.y * 1024 * 64;
+    const uint qs = group.x * 64;
+    const uint row = ((lane >> 1) & 3) + ((lane >> 2) & 4);
+    const uint col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
+    const uint qr = qs + sg * 8 + row;
+    threadgroup half keys[4096];
+    threadgroup half vals[4096];
+    for (uint i = tid; i < 1024; i += 256) {
+        ((threadgroup half4*)keys)[i] = half4(Q[base/4 + qs*16 + i]);
+    }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    simdgroup_half8x8 q[8];
+    simdgroup_float8x8 out[8];
+    #pragma unroll
+    for (uint d=0; d<8; ++d) {
+        simdgroup_load(q[d], keys + sg*8*64 + d*8, 64);
+        out[d] = simdgroup_float8x8(0.0f);
+    }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    float running_max = -INFINITY;
+    float denominator = 0.0f;
+    for (uint ks=0; ks<=qs; ks+=64) {
+        for (uint i=tid; i<1024; i+=256) {
+            ((threadgroup half4*)keys)[i] = half4(K[base/4 + ks*16 + i]);
+            ((threadgroup half4*)vals)[i] = half4(V[base/4 + ks*16 + i]);
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        simdgroup_float8x8 scores[8];
+        #pragma unroll
+        for (uint j=0; j<8; ++j) scores[j] = simdgroup_float8x8(0.0f);
+        #pragma unroll
+        for (uint d=0; d<8; ++d) {
+            #pragma unroll
+            for (uint j=0; j<8; ++j) {
+                simdgroup_half8x8 km;
+                simdgroup_load(km, keys + j*8*64 + d*8, 64, ulong2(0), true);
+                simdgroup_multiply_accumulate(scores[j], q[d], km, scores[j]);
+            }
+        }
+        float mx = -INFINITY;
+        #pragma unroll
+        for (uint j=0; j<8; ++j) {
+            #pragma unroll
+            for (uint e=0; e<2; ++e) {
+                float s = scores[j].thread_elements()[e] * 0.125f;
+                if (ks==qs && ks+j*8+col+e>qr) s=-INFINITY;
+                scores[j].thread_elements()[e]=s;
+                mx=max(mx,s);
+            }
+        }
+        mx = max(mx, simd_shuffle_xor(mx,1));
+        mx = max(mx, simd_shuffle_xor(mx,8));
+        float nm = max(running_max,mx);
+        float alpha = exp(running_max-nm);
+        float sum = 0.0f;
+        simdgroup_half8x8 p[8];
+        #pragma unroll
+        for (uint j=0; j<8; ++j) {
+            #pragma unroll
+            for (uint e=0; e<2; ++e) {
+                float z=exp(scores[j].thread_elements()[e]-nm);
+                p[j].thread_elements()[e]=half(z);
+                sum+=z;
+            }
+        }
+        sum += simd_shuffle_xor(sum,1);
+        sum += simd_shuffle_xor(sum,8);
+        denominator=denominator*alpha+sum;
+        running_max=nm;
+        #pragma unroll
+        for (uint d=0; d<8; ++d) {
+            out[d].thread_elements()[0]*=alpha;
+            out[d].thread_elements()[1]*=alpha;
+            #pragma unroll
+            for (uint j=0; j<8; ++j) {
+                simdgroup_half8x8 vm;
+                simdgroup_load(vm, vals+j*8*64+d*8,64);
+                simdgroup_multiply_accumulate(out[d],p[j],vm,out[d]);
+            }
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+    }
+    #pragma unroll
+    for (uint d=0; d<8; ++d) {
+        O[base+qr*64+d*8+col]=out[d].thread_elements()[0]/denominator;
+        O[base+qr*64+d*8+col+1]=out[d].thread_elements()[1]/denominator;
+    }
+}
+'''
+
+
+_lib = torch.mps.compile_shader(_source())
+
+
 def kernel(q, k, v):
-    T = q.size(-2)
-    s = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
-    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
-    e = (s - s.max(-1, keepdim=True).values).exp()
-    return (e / e.sum(-1, keepdim=True)) @ v
+    out = torch.empty_like(q)
+    _lib.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(256, 1, 1))
+    return out

```

</details>

`pwd && rg --files -g 'AGENTS.md' -g 'kernel.py' -g 'reference.py' -g 'bench.py' -g 'check.py' -g 'LESSONS.md`  
→ bench.py  
→ reference.py

### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -88,10 +88,15 @@
             out[d].thread_elements()[0]*=alpha;
             out[d].thread_elements()[1]*=alpha;
             #pragma unroll
-            for (uint j=0; j<8; ++j) {
-                simdgroup_half8x8 vm;
-                simdgroup_load(vm, vals+j*8*64+d*8,64);
-                simdgroup_multiply_accumulate(out[d],p[j],vm,out[d]);
+            for (uint j=0; j<8; j+=2) {
+                simdgroup_half8x8 vm0, vm1;
+                simdgroup_half8x8 partial(half(0.0f));
+                simdgroup_load(vm0, vals+j*8*64+d*8,64);
+                simdgroup_load(vm1, vals+(j+1)*8*64+d*8,64);
+                simdgroup_multiply_accumulate(partial,p[j],vm0,partial);
+                simdgroup_multiply_accumulate(partial,p[j+1],vm1,partial);
+                out[d].thread_elements()[0] += float(partial.thread_elements()[0]);
+                out[d].thread_elements()[1] += float(partial.thread_elements()[1]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);

```

</details>

`cat kernel.py reference.py LESSONS.md bench.py check.py`  
→ print(f"{'OK' if got.shape == want.shape and err <= TOL else 'WRONG'}: relative error {err:.2e} (allowed {TOL:.0e})")  
→ 2026-10-14T13:10:41.090102Z ERROR codex_models_manager::manager: failed to refresh available models: request timed out

- `python - <<'PY` (exited 1 in 0ms:)
- `ls -la`
- `python -c 'import torch; print(torch.__version__); print(torch.mps.compile_shader.__doc__)`
> The earlier rounds found that fused Metal attention was much faster than the PyTorch implementation. I’ll start with their strongest approach: register-held scores, unrolled matrix operations, and float32 accumulation.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.74e-04 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -1,8 +1,9 @@
+import re
 import torch
 
 
 def _source():
-    return r'''
+    source = r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
@@ -43,7 +44,7 @@
         threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 scores[8];
         #pragma unroll
-        for (uint j=0; j<8; ++j) scores[j] = simdgroup_float8x8(0.0f);
+        for (uint j=0; j<8; ++j) { scores[j] = simdgroup_float8x8(0.0f); }
         #pragma unroll
         for (uint d=0; d<8; ++d) {
             #pragma unroll
@@ -88,15 +89,10 @@
             out[d].thread_elements()[0]*=alpha;
             out[d].thread_elements()[1]*=alpha;
             #pragma unroll
-            for (uint j=0; j<8; j+=2) {
-                simdgroup_half8x8 vm0, vm1;
-                simdgroup_half8x8 partial(half(0.0f));
-                simdgroup_load(vm0, vals+j*8*64+d*8,64);
-                simdgroup_load(vm1, vals+(j+1)*8*64+d*8,64);
-                simdgroup_multiply_accumulate(partial,p[j],vm0,partial);
-                simdgroup_multiply_accumulate(partial,p[j+1],vm1,partial);
-                out[d].thread_elements()[0] += float(partial.thread_elements()[0]);
-                out[d].thread_elements()[1] += float(partial.thread_elements()[1]);
+            for (uint j=0; j<8; ++j) {
+                simdgroup_half8x8 vm;
+                simdgroup_load(vm, vals+j*8*64+d*8,64);
+                simdgroup_multiply_accumulate(out[d],p[j],vm,out[d]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
@@ -108,6 +104,19 @@
     }
 }
 '''
+    pattern = re.compile(r'#pragma unroll\s+for \(uint (\w+)=(\d+); \1<(\d+); \+\+\1\) \{')
+    while matches := list(pattern.finditer(source)):
+        match = matches[-1]
+        end = match.end()
+        depth = 1
+        while depth:
+            depth += (source[end] == '{') - (source[end] == '}')
+            end += 1
+        body = source[match.end():end-1]
+        expanded = '\n'.join('{' + re.sub(r'\b'+match[1]+r'\b', str(i), body) + '}'
+                             for i in range(int(match[2]), int(match[3])))
+        source = source[:match.start()] + expanded + source[end:]
+    return source
 
 
 _lib = torch.mps.compile_shader(_source())

```

</details>

`python bench.py`  
→ run 1 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 66.89 ms, kernel 6.43 ms, speedup 10.40x

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 8.58e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -22,18 +22,15 @@
     const uint qr = qs + sg * 8 + row;
     threadgroup half keys[4096];
     threadgroup half vals[4096];
-    for (uint i = tid; i < 1024; i += 256) {
-        ((threadgroup half4*)keys)[i] = half4(Q[base/4 + qs*16 + i]);
-    }
-    threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 q[8];
     simdgroup_float8x8 out[8];
     #pragma unroll
     for (uint d=0; d<8; ++d) {
-        simdgroup_load(q[d], keys + sg*8*64 + d*8, 64);
+        float2 pair = ((device const float2*)Q)[(base + qr*64 + d*8 + col)/2];
+        q[d].thread_elements()[0] = half(pair.x);
+        q[d].thread_elements()[1] = half(pair.y);
         out[d] = simdgroup_float8x8(0.0f);
     }
-    threadgroup_barrier(mem_flags::mem_threadgroup);
     float running_max = -INFINITY;
     float denominator = 0.0f;
     for (uint ks=0; ks<=qs; ks+=64) {

```

</details>

`python bench.py`  
→ run 2 of 10 submitted.  
→ attention: correct (relative error 1.2e-03). reference 68.72 ms, kernel 7.24 ms, speedup 9.49x

> The first fused version passed and took 6.43 ms, a 10.40× speedup. Half-precision value partial sums were slower at 7.24 ms. I’ll now expand the fixed matrix loops explicitly, which helped in earlier rounds.
> 2026-10-14T13:13:10.613041Z ERROR codex_core::tools::router: error=apply_patch verification failed: invalid patch: multiple operations target ./kernel.py

- `cat kernel.py`
- edited `solution.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.04e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -22,15 +22,18 @@
     const uint qr = qs + sg * 8 + row;
     threadgroup half keys[4096];
     threadgroup half vals[4096];
+    for (uint i = tid; i < 1024; i += 256) {
+        ((threadgroup half4*)keys)[i] = half4(Q[base/4 + qs*16 + i]);
+    }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 q[8];
     simdgroup_float8x8 out[8];
     #pragma unroll
     for (uint d=0; d<8; ++d) {
-        float2 pair = ((device const float2*)Q)[(base + qr*64 + d*8 + col)/2];
-        q[d].thread_elements()[0] = half(pair.x);
-        q[d].thread_elements()[1] = half(pair.y);
+        simdgroup_load(q[d], keys + sg*8*64 + d*8, 64);
         out[d] = simdgroup_float8x8(0.0f);
     }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     float running_max = -INFINITY;
     float denominator = 0.0f;
     for (uint ks=0; ks<=qs; ks+=64) {
@@ -86,10 +89,15 @@
             out[d].thread_elements()[0]*=alpha;
             out[d].thread_elements()[1]*=alpha;
             #pragma unroll
-            for (uint j=0; j<8; ++j) {
-                simdgroup_half8x8 vm;
-                simdgroup_load(vm, vals+j*8*64+d*8,64);
-                simdgroup_multiply_accumulate(out[d],p[j],vm,out[d]);
+            for (uint j=0; j<8; j+=2) {
+                simdgroup_half8x8 vm0, vm1;
+                simdgroup_half8x8 partial(half(0.0f));
+                simdgroup_load(vm0, vals+j*8*64+d*8,64);
+                simdgroup_load(vm1, vals+(j+1)*8*64+d*8,64);
+                simdgroup_multiply_accumulate(partial,p[j],vm0,partial);
+                simdgroup_multiply_accumulate(partial,p[j+1],vm1,partial);
+                out[d].thread_elements()[0] += float(partial.thread_elements()[0]);
+                out[d].thread_elements()[1] += float(partial.thread_elements()[1]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
@@ -101,7 +109,7 @@
     }
 }
 '''
-    pattern = re.compile(r'#pragma unroll\s+for \(uint (\w+)=(\d+); \1<(\d+); \+\+\1\) \{')
+    pattern = re.compile(r'#pragma unroll\s+for \(uint (\w+)=(\d+); \1<(\d+); (\+\+\1|\1\+=(\d+))\) \{')
     while matches := list(pattern.finditer(source)):
         match = matches[-1]
         end = match.end()
@@ -111,7 +119,7 @@
             end += 1
         body = source[match.end():end-1]
         expanded = '\n'.join('{' + re.sub(r'\b'+match[1]+r'\b', str(i), body) + '}'
-                             for i in range(int(match[2]), int(match[3])))
+                             for i in range(int(match[2]), int(match[3]), int(match[5] or 1)))
         source = source[:match.start()] + expanded + source[end:]
     return source
 

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 66.48 ms, kernel 3.74 ms, speedup 17.76x

- `ls results`
- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 5.57e-04 (allowed 2e-03)
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -16,28 +16,25 @@
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
     const uint base = group.y * 1024 * 64;
-    const uint qs = group.x * 64;
+    const uint qs = group.x * 128;
     const uint row = ((lane >> 1) & 3) + ((lane >> 2) & 4);
     const uint col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
     const uint qr = qs + sg * 8 + row;
     threadgroup half keys[4096];
     threadgroup half vals[4096];
-    for (uint i = tid; i < 1024; i += 256) {
-        ((threadgroup half4*)keys)[i] = half4(Q[base/4 + qs*16 + i]);
-    }
-    threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 q[8];
     simdgroup_float8x8 out[8];
     #pragma unroll
     for (uint d=0; d<8; ++d) {
-        simdgroup_load(q[d], keys + sg*8*64 + d*8, 64);
+        float2 pair = ((device const float2*)Q)[(base + qr*64 + d*8 + col)/2];
+        q[d].thread_elements()[0] = half(pair.x);
+        q[d].thread_elements()[1] = half(pair.y);
         out[d] = simdgroup_float8x8(0.0f);
     }
-    threadgroup_barrier(mem_flags::mem_threadgroup);
     float running_max = -INFINITY;
     float denominator = 0.0f;
-    for (uint ks=0; ks<=qs; ks+=64) {
-        for (uint i=tid; i<1024; i+=256) {
+    for (uint ks=0; ks<=qs+64; ks+=64) {
+        for (uint i=tid; i<1024; i+=512) {
             ((threadgroup half4*)keys)[i] = half4(K[base/4 + ks*16 + i]);
             ((threadgroup half4*)vals)[i] = half4(V[base/4 + ks*16 + i]);
         }
@@ -60,7 +57,7 @@
             #pragma unroll
             for (uint e=0; e<2; ++e) {
                 float s = scores[j].thread_elements()[e] * 0.125f;
-                if (ks==qs && ks+j*8+col+e>qr) s=-INFINITY;
+                if (ks+64>qs && ks+j*8+col+e>qr) s=-INFINITY;
                 scores[j].thread_elements()[e]=s;
                 mx=max(mx,s);
             }
@@ -89,15 +86,10 @@
             out[d].thread_elements()[0]*=alpha;
             out[d].thread_elements()[1]*=alpha;
             #pragma unroll
-            for (uint j=0; j<8; j+=2) {
-                simdgroup_half8x8 vm0, vm1;
-                simdgroup_half8x8 partial(half(0.0f));
-                simdgroup_load(vm0, vals+j*8*64+d*8,64);
-                simdgroup_load(vm1, vals+(j+1)*8*64+d*8,64);
-                simdgroup_multiply_accumulate(partial,p[j],vm0,partial);
-                simdgroup_multiply_accumulate(partial,p[j+1],vm1,partial);
-                out[d].thread_elements()[0] += float(partial.thread_elements()[0]);
-                out[d].thread_elements()[1] += float(partial.thread_elements()[1]);
+            for (uint j=0; j<8; ++j) {
+                simdgroup_half8x8 vm;
+                simdgroup_load(vm, vals+j*8*64+d*8,64);
+                simdgroup_multiply_accumulate(out[d],p[j],vm,out[d]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
@@ -109,7 +101,7 @@
     }
 }
 '''
-    pattern = re.compile(r'#pragma unroll\s+for \(uint (\w+)=(\d+); \1<(\d+); (\+\+\1|\1\+=(\d+))\) \{')
+    pattern = re.compile(r'#pragma unroll\s+for \(uint (\w+)=(\d+); \1<(\d+); \+\+\1\) \{')
     while matches := list(pattern.finditer(source)):
         match = matches[-1]
         end = match.end()
@@ -119,7 +111,7 @@
             end += 1
         body = source[match.end():end-1]
         expanded = '\n'.join('{' + re.sub(r'\b'+match[1]+r'\b', str(i), body) + '}'
-                             for i in range(int(match[2]), int(match[3]), int(match[5] or 1)))
+                             for i in range(int(match[2]), int(match[3])))
         source = source[:match.start()] + expanded + source[end:]
     return source
 
@@ -129,5 +121,5 @@
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _lib.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(256, 1, 1))
+    _lib.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(512, 1, 1))
     return out

```

</details>

`python bench.py`  
→ run 4 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 70.94 ms, kernel 3.87 ms, speedup 18.31x

- `cat kernel.py`
- edited `solution.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 7.49e-04 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -16,36 +16,39 @@
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
     const uint base = group.y * 1024 * 64;
-    const uint qs = group.x * 128;
+    const uint qs = group.x * 64;
     const uint row = ((lane >> 1) & 3) + ((lane >> 2) & 4);
     const uint col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
     const uint qr = qs + sg * 8 + row;
     threadgroup half keys[4096];
-    threadgroup half vals[4096];
+    threadgroup half vals[2048];
+    for (uint i = tid; i < 1024; i += 256) {
+        ((threadgroup half4*)keys)[i] = half4(Q[base/4 + qs*16 + i]);
+    }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 q[8];
     simdgroup_float8x8 out[8];
     #pragma unroll
     for (uint d=0; d<8; ++d) {
-        float2 pair = ((device const float2*)Q)[(base + qr*64 + d*8 + col)/2];
-        q[d].thread_elements()[0] = half(pair.x);
-        q[d].thread_elements()[1] = half(pair.y);
+        simdgroup_load(q[d], keys + sg*8*64 + d*8, 64);
         out[d] = simdgroup_float8x8(0.0f);
     }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     float running_max = -INFINITY;
     float denominator = 0.0f;
-    for (uint ks=0; ks<=qs+64; ks+=64) {
-        for (uint i=tid; i<1024; i+=512) {
+    for (uint ks=0; ks<=qs+32; ks+=32) {
+        for (uint i=tid; i<512; i+=256) {
             ((threadgroup half4*)keys)[i] = half4(K[base/4 + ks*16 + i]);
             ((threadgroup half4*)vals)[i] = half4(V[base/4 + ks*16 + i]);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 scores[8];
+        simdgroup_float8x8 scores[4];
         #pragma unroll
-        for (uint j=0; j<8; ++j) { scores[j] = simdgroup_float8x8(0.0f); }
+        for (uint j=0; j<4; ++j) { scores[j] = simdgroup_float8x8(0.0f); }
         #pragma unroll
         for (uint d=0; d<8; ++d) {
             #pragma unroll
-            for (uint j=0; j<8; ++j) {
+            for (uint j=0; j<4; ++j) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km, keys + j*8*64 + d*8, 64, ulong2(0), true);
                 simdgroup_multiply_accumulate(scores[j], q[d], km, scores[j]);
@@ -53,11 +56,11 @@
         }
         float mx = -INFINITY;
         #pragma unroll
-        for (uint j=0; j<8; ++j) {
+        for (uint j=0; j<4; ++j) {
             #pragma unroll
             for (uint e=0; e<2; ++e) {
                 float s = scores[j].thread_elements()[e] * 0.125f;
-                if (ks+64>qs && ks+j*8+col+e>qr) s=-INFINITY;
+                if (ks+32>qs && ks+j*8+col+e>qr) s=-INFINITY;
                 scores[j].thread_elements()[e]=s;
                 mx=max(mx,s);
             }
@@ -67,9 +70,9 @@
         float nm = max(running_max,mx);
         float alpha = exp(running_max-nm);
         float sum = 0.0f;
-        simdgroup_half8x8 p[8];
+        simdgroup_half8x8 p[4];
         #pragma unroll
-        for (uint j=0; j<8; ++j) {
+        for (uint j=0; j<4; ++j) {
             #pragma unroll
             for (uint e=0; e<2; ++e) {
                 float z=exp(scores[j].thread_elements()[e]-nm);
@@ -86,7 +89,7 @@
             out[d].thread_elements()[0]*=alpha;
             out[d].thread_elements()[1]*=alpha;
             #pragma unroll
-            for (uint j=0; j<8; ++j) {
+            for (uint j=0; j<4; ++j) {
                 simdgroup_half8x8 vm;
                 simdgroup_load(vm, vals+j*8*64+d*8,64);
                 simdgroup_multiply_accumulate(out[d],p[j],vm,out[d]);
@@ -121,5 +124,5 @@
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _lib.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(512, 1, 1))
+    _lib.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(256, 1, 1))
     return out

```

</details>

`python bench.py`  
→ run 5 of 10 submitted.  
→ attention: correct (relative error 1.2e-03). reference 70.56 ms, kernel 3.97 ms, speedup 17.79x

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.34e-04 (allowed 2e-03)
> The 3.74 ms version is still the fastest. Direct query loads took 3.87 ms, and explicitly unrolled half-precision partial sums took 3.97 ms. The larger query tile is still being measured; I’ll also test a smaller key tile.

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.40e-04 (allowed 2e-03)
- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.30e-04 (allowed 2e-03)
> The tile changes also regressed: 128 query rows took 4.12 ms, and 32-key tiles took 3.90 ms. I’m keeping the 64-by-64 tile and using the remaining runs to test base-two exponentials and a single diagonal-mask branch.

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.65e-04 (allowed 2e-03)
- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.74e-04 (allowed 2e-03)
> Base-two exponentials improved the best time to 3.65 ms, an 18.71× speedup. Moving the mask into a separate branch was slower. The tenth run is testing Metal’s fast exponential; after that I’ll keep the fastest version, check fresh inputs, and rewrite the lessons.

- `cat kernel.py`
- `cat kernel.py`
- edited `solution.py`
- `python -c 'import torch; from kernel import kernel; from reference import inputs, reference, TOL; errors=[]`
> I’ve kept the 3.65 ms version; the fast exponential was slower at 3.97 ms. All 30 fresh-input checks passed, with a maximum relative error of 0.000595 against the 0.002 limit. I’ll finish the lessons and run the final check.

- `cat LESSONS.md`
- edited `solution.py`
- `wc -w LESSONS.md`
- `python check.py`  
→ OK: relative error 3.42e-04 (allowed 2e-03)
> Best benchmark: **3.65 ms, 18.71× speedup**. All 30 fresh-input checks passed; final `check.py` passed with error 0.000342. Rewrote `LESSONS.md` in 278 words.
> 
> EDIT: kernel.py now uses fused Metal causal attention with explicitly unrolled matrix operations, stable base-two softmax, and float32 accumulation.

