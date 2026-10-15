# Agent `lessons-r3-1`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what is still true from the notes you were given. At most 300 words, in words, no code.

13 min, 65,388 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 3.683 ms | 17.27x |
| run2 | 3.813 ms | 17.86x |
| run3 | 3.768 ms | 18.0x |
| run4 | 3.902 ms | 18.0x |
| run5 | 4.63 ms | 16.43x |
| run6 | 4.081 ms | 17.21x |
| run7 | 3.885 ms | 17.52x |
| run8 | 3.69 ms | 17.67x |
| run9 | 3.707 ms | 18.27x |
| run10 | 3.861 ms | 17.45x |
| final | 3.919 ms | 17.34x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the reference and earlier lessons, then use the benchmark runs to compare implementations and check the final result.

### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -1,10 +1,94 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
 import torch
 
 
+def _source():
+    src = r'''
+#include <metal_stdlib>
+#include <metal_simdgroup_matrix>
+using namespace metal;
+kernel void attention(device const float* Q [[buffer(0)]],
+                      device const float* K [[buffer(1)]],
+                      device const float* V [[buffer(2)]],
+                      device float* O [[buffer(3)]],
+                      uint3 group [[threadgroup_position_in_grid]],
+                      uint tid [[thread_index_in_threadgroup]],
+                      uint lane [[thread_index_in_simdgroup]],
+                      uint sg [[simdgroup_index_in_threadgroup]]) {
+    threadgroup half keys[4096];
+    threadgroup half values[4096];
+    uint base = group.y * 1024 * 64;
+    uint first = group.x * 64;
+    uint row = ((lane & 7) >> 1) + ((lane >> 4) << 2);
+    uint col = (lane & 1) * 2 + ((lane & 8) >> 1);
+    uint qr = first + sg * 8 + row;
+    simdgroup_half8x8 qm[8];
+    simdgroup_float8x8 out[8];
+    for (uint z=tid; z<1024; z+=256) {
+        ((threadgroup half4*)keys)[z] = half4(((device const float4*)(Q+base+first*64))[z]);
+    }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    __INIT__
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    float maximum = -INFINITY;
+    float denom = 0.0f;
+    for (uint start=0; start<=first; start+=64) {
+        for (uint z=tid; z<1024; z+=256) {
+            ((threadgroup half4*)keys)[z] = half4(((device const float4*)(K+base+start*64))[z]);
+            ((threadgroup half4*)values)[z] = half4(((device const float4*)(V+base+start*64))[z]);
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        simdgroup_float8x8 scores[8];
+        DOT
+        float tilemax = -INFINITY;
+        SCALE
+        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
+        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
+        float nm = max(maximum, tilemax);
+        float alpha = exp(maximum - nm);
+        float sum = 0.0f;
+        simdgroup_half8x8 probs[8];
+        PROBS
+        sum += simd_shuffle_xor(sum, 1);
+        sum += simd_shuffle_xor(sum, 8);
+        denom = denom * alpha + sum;
+        maximum = nm;
+        VALUE
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+    }
+    STORE
+}
+'''
+    src = src.replace('__INIT__', '\n'.join(f'simdgroup_load(qm[{d}], keys+sg*8*64+{d*8}, 64); out[{d}] = simdgroup_float8x8(0.0f);' for d in range(8)))
+    dot = []
+    for j in range(8):
+        dot.append(f'scores[{j}] = simdgroup_float8x8(0.0f);')
+        for d in range(8):
+            dot.append(f'{{ simdgroup_half8x8 km; simdgroup_load(km, keys+{j*8*64+d*8}, 64, ulong2(0), true); simdgroup_multiply_accumulate(scores[{j}], qm[{d}], km, scores[{j}]); }}')
+    src = src.replace('DOT', '\n'.join(dot))
+    scale = []
+    for j in range(8):
+        for e in range(2):
+            scale.append(f'{{ float x = scores[{j}].thread_elements()[{e}] * 0.125f; if (start == first && start+{j*8}+col+{e} > qr) x = -INFINITY; scores[{j}].thread_elements()[{e}] = x; tilemax = max(tilemax,x); }}')
+    src = src.replace('SCALE', '\n'.join(scale))
+    probs = []
+    for j in range(8):
+        for e in range(2):
+            probs.append(f'{{ float p = exp(scores[{j}].thread_elements()[{e}] - nm); sum += p; probs[{j}].thread_elements()[{e}] = half(p); }}')
+    src = src.replace('PROBS', '\n'.join(probs))
+    vals = []
+    for d in range(8):
+        vals.append(f'out[{d}].thread_elements()[0] *= alpha; out[{d}].thread_elements()[1] *= alpha;')
+        for j in range(8):
+            vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{j*8*64+d*8},64); simdgroup_multiply_accumulate(out[{d}], probs[{j}], vm, out[{d}]); }}')
+    src = src.replace('VALUE', '\n'.join(vals))
+    src = src.replace('STORE', '\n'.join(f'O[base+qr*64+{d*8}+col+{e}] = out[{d}].thread_elements()[{e}] / denom;' for d in range(8) for e in range(2)))
+    return src
+
+
+_shader = torch.mps.compile_shader(_source())
+
+
 def kernel(q, k, v):
-    T = q.size(-2)
-    s = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
-    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
-    e = (s - s.max(-1, keepdim=True).values).exp()
-    return (e / e.sum(-1, keepdim=True)) @ v
+    out = torch.empty_like(q)
+    _shader.attention(q, k, v, out, threads=(16*256, 64, 1), group_size=(256, 1, 1))
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
@@ -78,8 +78,11 @@
     vals = []
     for d in range(8):
         vals.append(f'out[{d}].thread_elements()[0] *= alpha; out[{d}].thread_elements()[1] *= alpha;')
-        for j in range(8):
-            vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{j*8*64+d*8},64); simdgroup_multiply_accumulate(out[{d}], probs[{j}], vm, out[{d}]); }}')
+        for j in range(0, 8, 2):
+            vals.append('{ simdgroup_half8x8 partial(half(0.0f));')
+            for z in range(j, j + 2):
+                vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{z*8*64+d*8},64); simdgroup_multiply_accumulate(partial, probs[{z}], vm, partial); }}')
+            vals.append(f'out[{d}].thread_elements()[0] += float(partial.thread_elements()[0]); out[{d}].thread_elements()[1] += float(partial.thread_elements()[1]); }}')
     src = src.replace('VALUE', '\n'.join(vals))
     src = src.replace('STORE', '\n'.join(f'O[base+qr*64+{d*8}+col+{e}] = out[{d}].thread_elements()[{e}] / denom;' for d in range(8) for e in range(2)))
     return src

```

</details>

`cat kernel.py reference.py LESSONS.md bench.py check.py`  
→     err = (got.float() - want).abs().max().item() / want.abs().max().item()  
→ print(f"{'OK' if got.shape == want.shape and err <= TOL else 'WRONG'}: relative error {err:.2e} (allowed {TOL:.0e})")

- `python -c 'import torch; print(torch.__version__); print(torch.mps.compile_shader.__doc__)`
- `rg --files -g '"'!queue/**'"' -g '"'!results/**`
> The earlier lessons favor fused Metal attention with explicitly unrolled matrix operations. I’ll build on that: keep scores and softmax in registers, use half precision matrix inputs, and retain float32 accumulators.

- `cat > kernel.py <<'PY` (exited 1 in 485ms:)
- edited `solution.py`
- `python check.py`  
→ simdgroup_load(qm[7], keys+sg*8*64+56, 64); out[7] = simdgroup_float8x8(0.0f);Y;  
→                                                                               ^
- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text().replace(\"    INIT\", \"    __INIT__\").replace(\"replace(\\x27INIT\\x27\", \"rep...`
- `python check.py`  
→ OK: relative error 4.96e-04 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -1,4 +1,8 @@
 import torch
+
+
+_QSIZE = 64
+_KSIZE = 32
 
 
 def _source():
@@ -14,16 +18,16 @@
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup half keys[4096];
-    threadgroup half values[4096];
+    threadgroup half keys[__KEY_STORAGE__];
+    threadgroup half values[__KEY_STORAGE__];
     uint base = group.y * 1024 * 64;
-    uint first = group.x * 64;
+    uint first = group.x * __QSIZE__;
     uint row = ((lane & 7) >> 1) + ((lane >> 4) << 2);
     uint col = (lane & 1) * 2 + ((lane & 8) >> 1);
     uint qr = first + sg * 8 + row;
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 out[8];
-    for (uint z=tid; z<1024; z+=256) {
+    for (uint z=tid; z<__QUERY_VECTORS__; z+=__THREADS__) {
         ((threadgroup half4*)keys)[z] = half4(((device const float4*)(Q+base+first*64))[z]);
     }
     threadgroup_barrier(mem_flags::mem_threadgroup);
@@ -31,13 +35,13 @@
     threadgroup_barrier(mem_flags::mem_threadgroup);
     float maximum = -INFINITY;
     float denom = 0.0f;
-    for (uint start=0; start<=first; start+=64) {
-        for (uint z=tid; z<1024; z+=256) {
+    for (uint start=0; start<first+__QSIZE__; start+=__KSIZE__) {
+        for (uint z=tid; z<__KEY_VECTORS__; z+=__THREADS__) {
             ((threadgroup half4*)keys)[z] = half4(((device const float4*)(K+base+start*64))[z]);
             ((threadgroup half4*)values)[z] = half4(((device const float4*)(V+base+start*64))[z]);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 scores[8];
+        simdgroup_float8x8 scores[__KMATS__];
         DOT
         float tilemax = -INFINITY;
         SCALE
@@ -46,7 +50,7 @@
         float nm = max(maximum, tilemax);
         float alpha = exp(maximum - nm);
         float sum = 0.0f;
-        simdgroup_half8x8 probs[8];
+        simdgroup_half8x8 probs[__KMATS__];
         PROBS
         sum += simd_shuffle_xor(sum, 1);
         sum += simd_shuffle_xor(sum, 8);
@@ -60,32 +64,38 @@
 '''
     src = src.replace('__INIT__', '\n'.join(f'simdgroup_load(qm[{d}], keys+sg*8*64+{d*8}, 64); out[{d}] = simdgroup_float8x8(0.0f);' for d in range(8)))
     dot = []
-    for j in range(8):
+    for j in range(_KSIZE // 8):
         dot.append(f'scores[{j}] = simdgroup_float8x8(0.0f);')
         for d in range(8):
             dot.append(f'{{ simdgroup_half8x8 km; simdgroup_load(km, keys+{j*8*64+d*8}, 64, ulong2(0), true); simdgroup_multiply_accumulate(scores[{j}], qm[{d}], km, scores[{j}]); }}')
     src = src.replace('DOT', '\n'.join(dot))
     scale = []
-    for j in range(8):
+    for j in range(_KSIZE // 8):
         for e in range(2):
-            scale.append(f'{{ float x = scores[{j}].thread_elements()[{e}] * 0.125f; if (start == first && start+{j*8}+col+{e} > qr) x = -INFINITY; scores[{j}].thread_elements()[{e}] = x; tilemax = max(tilemax,x); }}')
+            scale.append(f'{{ float x = scores[{j}].thread_elements()[{e}] * 0.125f; if (start + {_KSIZE} > first && start+{j*8}+col+{e} > qr) x = -INFINITY; scores[{j}].thread_elements()[{e}] = x; tilemax = max(tilemax,x); }}')
     src = src.replace('SCALE', '\n'.join(scale))
     probs = []
-    for j in range(8):
+    for j in range(_KSIZE // 8):
         for e in range(2):
             probs.append(f'{{ float p = exp(scores[{j}].thread_elements()[{e}] - nm); sum += p; probs[{j}].thread_elements()[{e}] = half(p); }}')
     src = src.replace('PROBS', '\n'.join(probs))
     vals = []
     for d in range(8):
         vals.append(f'out[{d}].thread_elements()[0] *= alpha; out[{d}].thread_elements()[1] *= alpha;')
-        for j in range(0, 8, 2):
+        for j in range(0, _KSIZE // 8, 2):
             vals.append('{ simdgroup_half8x8 partial(half(0.0f));')
             for z in range(j, j + 2):
                 vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{z*8*64+d*8},64); simdgroup_multiply_accumulate(partial, probs[{z}], vm, partial); }}')
             vals.append(f'out[{d}].thread_elements()[0] += float(partial.thread_elements()[0]); out[{d}].thread_elements()[1] += float(partial.thread_elements()[1]); }}')
     src = src.replace('VALUE', '\n'.join(vals))
     src = src.replace('STORE', '\n'.join(f'O[base+qr*64+{d*8}+col+{e}] = out[{d}].thread_elements()[{e}] / denom;' for d in range(8) for e in range(2)))
-    return src
+    return (src.replace('__QSIZE__', str(_QSIZE))
+            .replace('__KSIZE__', str(_KSIZE))
+            .replace('__KEY_STORAGE__', str(max(_QSIZE, _KSIZE) * 64))
+            .replace('__QUERY_VECTORS__', str(_QSIZE * 16))
+            .replace('__KEY_VECTORS__', str(_KSIZE * 16))
+            .replace('__THREADS__', str(_QSIZE * 4))
+            .replace('__KMATS__', str(_KSIZE // 8)))
 
 
 _shader = torch.mps.compile_shader(_source())
@@ -93,5 +103,5 @@
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _shader.attention(q, k, v, out, threads=(16*256, 64, 1), group_size=(256, 1, 1))
+    _shader.attention(q, k, v, out, threads=(1024 // _QSIZE * (_QSIZE * 4), 64, 1), group_size=(_QSIZE * 4, 1, 1))
     return out

```

</details>

`python bench.py`  
→ run 1 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 63.61 ms, kernel 3.68 ms, speedup 17.27x

- `cat kernel.py`
> The first fused version passes and measures 3.68 ms, a 17.27× speedup in this run. I’ll test short half precision value sums next; the earlier lessons found them faster while keeping error within tolerance.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 6.60e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -27,12 +27,7 @@
     uint qr = first + sg * 8 + row;
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 out[8];
-    for (uint z=tid; z<__QUERY_VECTORS__; z+=__THREADS__) {
-        ((threadgroup half4*)keys)[z] = half4(((device const float4*)(Q+base+first*64))[z]);
-    }
-    threadgroup_barrier(mem_flags::mem_threadgroup);
     __INIT__
-    threadgroup_barrier(mem_flags::mem_threadgroup);
     float maximum = -INFINITY;
     float denom = 0.0f;
     for (uint start=0; start<first+__QSIZE__; start+=__KSIZE__) {
@@ -62,7 +57,7 @@
     STORE
 }
 '''
-    src = src.replace('__INIT__', '\n'.join(f'simdgroup_load(qm[{d}], keys+sg*8*64+{d*8}, 64); out[{d}] = simdgroup_float8x8(0.0f);' for d in range(8)))
+    src = src.replace('__INIT__', '\n'.join(f'qm[{d}].thread_elements()[0] = half(Q[base+qr*64+{d*8}+col]); qm[{d}].thread_elements()[1] = half(Q[base+qr*64+{d*8}+col+1]); out[{d}] = simdgroup_float8x8(0.0f);' for d in range(8)))
     dot = []
     for j in range(_KSIZE // 8):
         dot.append(f'scores[{j}] = simdgroup_float8x8(0.0f);')
@@ -91,7 +86,7 @@
     src = src.replace('STORE', '\n'.join(f'O[base+qr*64+{d*8}+col+{e}] = out[{d}].thread_elements()[{e}] / denom;' for d in range(8) for e in range(2)))
     return (src.replace('__QSIZE__', str(_QSIZE))
             .replace('__KSIZE__', str(_KSIZE))
-            .replace('__KEY_STORAGE__', str(max(_QSIZE, _KSIZE) * 64))
+            .replace('__KEY_STORAGE__', str(_KSIZE * 64))
             .replace('__QUERY_VECTORS__', str(_QSIZE * 16))
             .replace('__KEY_VECTORS__', str(_KSIZE * 16))
             .replace('__THREADS__', str(_QSIZE * 4))

```

</details>

`python bench.py`  
→     _shader.attention(q, k, v, out, threads=(16*256, 64, 1), group_size=(256, 1, 1))  
→     return out

- edited `solution.py`
- `python check.py`  
→ OK: relative error 9.66e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -1,7 +1,7 @@
 import torch
 
 
-_QSIZE = 64
+_QSIZE = 128
 _KSIZE = 32
 
 

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 1.2e-03). reference 67.82 ms, kernel 3.77 ms, speedup 18.00x

> The half precision partial sums measured 3.81 ms versus 3.68 ms for float32 sums, so they haven’t shown a benefit here. I’m testing smaller key tiles and direct query loads next to reduce shared memory use.

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 8.14e-04 (allowed 2e-03)
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -1,8 +1,4 @@
 import torch
-
-
-_QSIZE = 128
-_KSIZE = 32
 
 
 def _source():
@@ -18,25 +14,30 @@
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup half keys[__KEY_STORAGE__];
-    threadgroup half values[__KEY_STORAGE__];
+    threadgroup half keys[4096];
+    threadgroup half values[4096];
     uint base = group.y * 1024 * 64;
-    uint first = group.x * __QSIZE__;
+    uint first = group.x * 64;
     uint row = ((lane & 7) >> 1) + ((lane >> 4) << 2);
     uint col = (lane & 1) * 2 + ((lane & 8) >> 1);
     uint qr = first + sg * 8 + row;
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 out[8];
+    for (uint z=tid; z<1024; z+=256) {
+        ((threadgroup half4*)keys)[z] = half4(((device const float4*)(Q+base+first*64))[z]);
+    }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     __INIT__
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     float maximum = -INFINITY;
     float denom = 0.0f;
-    for (uint start=0; start<first+__QSIZE__; start+=__KSIZE__) {
-        for (uint z=tid; z<__KEY_VECTORS__; z+=__THREADS__) {
+    for (uint start=0; start<=first; start+=64) {
+        for (uint z=tid; z<1024; z+=256) {
             ((threadgroup half4*)keys)[z] = half4(((device const float4*)(K+base+start*64))[z]);
             ((threadgroup half4*)values)[z] = half4(((device const float4*)(V+base+start*64))[z]);
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 scores[__KMATS__];
+        simdgroup_float8x8 scores[8];
         DOT
         float tilemax = -INFINITY;
         SCALE
@@ -45,7 +46,7 @@
         float nm = max(maximum, tilemax);
         float alpha = exp(maximum - nm);
         float sum = 0.0f;
-        simdgroup_half8x8 probs[__KMATS__];
+        simdgroup_half8x8 probs[8];
         PROBS
         sum += simd_shuffle_xor(sum, 1);
         sum += simd_shuffle_xor(sum, 8);
@@ -57,40 +58,35 @@
     STORE
 }
 '''
-    src = src.replace('__INIT__', '\n'.join(f'qm[{d}].thread_elements()[0] = half(Q[base+qr*64+{d*8}+col]); qm[{d}].thread_elements()[1] = half(Q[base+qr*64+{d*8}+col+1]); out[{d}] = simdgroup_float8x8(0.0f);' for d in range(8)))
+    src = src.replace('__INIT__', '\n'.join(f'simdgroup_load(qm[{d}], keys+sg*8*64+{d*8}, 64); out[{d}] = simdgroup_float8x8(0.0f);' for d in range(8)))
     dot = []
-    for j in range(_KSIZE // 8):
+    for j in range(8):
         dot.append(f'scores[{j}] = simdgroup_float8x8(0.0f);')
         for d in range(8):
             dot.append(f'{{ simdgroup_half8x8 km; simdgroup_load(km, keys+{j*8*64+d*8}, 64, ulong2(0), true); simdgroup_multiply_accumulate(scores[{j}], qm[{d}], km, scores[{j}]); }}')
     src = src.replace('DOT', '\n'.join(dot))
-    scale = []
-    for j in range(_KSIZE // 8):
+    scale = ['if (start == first) {']
+    for j in range(8):
         for e in range(2):
-            scale.append(f'{{ float x = scores[{j}].thread_elements()[{e}] * 0.125f; if (start + {_KSIZE} > first && start+{j*8}+col+{e} > qr) x = -INFINITY; scores[{j}].thread_elements()[{e}] = x; tilemax = max(tilemax,x); }}')
+            scale.append(f'if (start+{j*8}+col+{e} > qr) scores[{j}].thread_elements()[{e}] = -INFINITY;')
+    scale.append('}')
+    for j in range(8):
+        for e in range(2):
+            scale.append(f'{{ float x = scores[{j}].thread_elements()[{e}] * 0.125f; scores[{j}].thread_elements()[{e}] = x; tilemax = max(tilemax,x); }}')
     src = src.replace('SCALE', '\n'.join(scale))
     probs = []
-    for j in range(_KSIZE // 8):
+    for j in range(8):
         for e in range(2):
             probs.append(f'{{ float p = exp(scores[{j}].thread_elements()[{e}] - nm); sum += p; probs[{j}].thread_elements()[{e}] = half(p); }}')
     src = src.replace('PROBS', '\n'.join(probs))
     vals = []
     for d in range(8):
         vals.append(f'out[{d}].thread_elements()[0] *= alpha; out[{d}].thread_elements()[1] *= alpha;')
-        for j in range(0, _KSIZE // 8, 2):
-            vals.append('{ simdgroup_half8x8 partial(half(0.0f));')
-            for z in range(j, j + 2):
-                vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{z*8*64+d*8},64); simdgroup_multiply_accumulate(partial, probs[{z}], vm, partial); }}')
-            vals.append(f'out[{d}].thread_elements()[0] += float(partial.thread_elements()[0]); out[{d}].thread_elements()[1] += float(partial.thread_elements()[1]); }}')
+        for j in range(8):
+            vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{j*8*64+d*8},64); simdgroup_multiply_accumulate(out[{d}], probs[{j}], vm, out[{d}]); }}')
     src = src.replace('VALUE', '\n'.join(vals))
     src = src.replace('STORE', '\n'.join(f'O[base+qr*64+{d*8}+col+{e}] = out[{d}].thread_elements()[{e}] / denom;' for d in range(8) for e in range(2)))
-    return (src.replace('__QSIZE__', str(_QSIZE))
-            .replace('__KSIZE__', str(_KSIZE))
-            .replace('__KEY_STORAGE__', str(_KSIZE * 64))
-            .replace('__QUERY_VECTORS__', str(_QSIZE * 16))
-            .replace('__KEY_VECTORS__', str(_KSIZE * 16))
-            .replace('__THREADS__', str(_QSIZE * 4))
-            .replace('__KMATS__', str(_KSIZE // 8)))
+    return src
 
 
 _shader = torch.mps.compile_shader(_source())
@@ -98,5 +94,5 @@
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _shader.attention(q, k, v, out, threads=(1024 // _QSIZE * (_QSIZE * 4), 64, 1), group_size=(_QSIZE * 4, 1, 1))
+    _shader.attention(q, k, v, out, threads=(16*256, 64, 1), group_size=(256, 1, 1))
     return out

```

</details>

`python bench.py`  


- `python check.py`  
→ OK: relative error 7.23e-04 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -24,7 +24,7 @@
     simdgroup_half8x8 qm[8];
     simdgroup_float8x8 out[8];
     for (uint z=tid; z<1024; z+=256) {
-        ((threadgroup half4*)keys)[z] = half4(((device const float4*)(Q+base+first*64))[z]);
+        ((threadgroup half4*)keys)[z] = half4(((device const float4*)(Q+base+first*64))[z] * 0.125f);
     }
     threadgroup_barrier(mem_flags::mem_threadgroup);
     __INIT__
@@ -44,7 +44,7 @@
         tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
         tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
         float nm = max(maximum, tilemax);
-        float alpha = exp(maximum - nm);
+        float alpha = fast::exp(maximum - nm);
         float sum = 0.0f;
         simdgroup_half8x8 probs[8];
         PROBS
@@ -72,12 +72,12 @@
     scale.append('}')
     for j in range(8):
         for e in range(2):
-            scale.append(f'{{ float x = scores[{j}].thread_elements()[{e}] * 0.125f; scores[{j}].thread_elements()[{e}] = x; tilemax = max(tilemax,x); }}')
+            scale.append(f'{{ float x = scores[{j}].thread_elements()[{e}]; tilemax = max(tilemax,x); }}')
     src = src.replace('SCALE', '\n'.join(scale))
     probs = []
     for j in range(8):
         for e in range(2):
-            probs.append(f'{{ float p = exp(scores[{j}].thread_elements()[{e}] - nm); sum += p; probs[{j}].thread_elements()[{e}] = half(p); }}')
+            probs.append(f'{{ float p = fast::exp(scores[{j}].thread_elements()[{e}] - nm); sum += p; probs[{j}].thread_elements()[{e}] = half(p); }}')
     src = src.replace('PROBS', '\n'.join(probs))
     vals = []
     for d in range(8):

```

</details>

`python bench.py`  
→ run 4 of 10 submitted.  
→ attention: correct (relative error 1.2e-03). reference 70.25 ms, kernel 3.90 ms, speedup 18.00x

- `python' -c 'from pathlib import Path; Path('\\''kernel.py'\\'').write_text(\"import torch\\n\\n\\ndef _source():\\n    src = r'\\'''\\'''\\''\\n#include <met...`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 1.85e-04 (allowed 2e-03)
> Smaller key tiles and direct query loads haven’t improved the measured time; the larger query tile took 4.63 ms in a run flagged as GPU-busy. I’m returning to the 64×64 float32 accumulator version and testing a simpler causal-mask branch.

### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -82,8 +82,11 @@
     vals = []
     for d in range(8):
         vals.append(f'out[{d}].thread_elements()[0] *= alpha; out[{d}].thread_elements()[1] *= alpha;')
-        for j in range(8):
-            vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{j*8*64+d*8},64); simdgroup_multiply_accumulate(out[{d}], probs[{j}], vm, out[{d}]); }}')
+        for j in range(0, 8, 4):
+            vals.append('{ simdgroup_half8x8 partial(half(0.0f));')
+            for z in range(j, j + 4):
+                vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{z*8*64+d*8},64); simdgroup_multiply_accumulate(partial, probs[{z}], vm, partial); }}')
+            vals.append(f'out[{d}].thread_elements()[0] += float(partial.thread_elements()[0]); out[{d}].thread_elements()[1] += float(partial.thread_elements()[1]); }}')
     src = src.replace('VALUE', '\n'.join(vals))
     src = src.replace('STORE', '\n'.join(f'O[base+qr*64+{d*8}+col+{e}] = out[{d}].thread_elements()[{e}] / denom;' for d in range(8) for e in range(2)))
     return src

```

</details>

`python bench.py`  
→ run 6 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 70.23 ms, kernel 4.08 ms, speedup 17.21x

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 2.99e-04 (allowed 2e-03)
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -82,9 +82,9 @@
     vals = []
     for d in range(8):
         vals.append(f'out[{d}].thread_elements()[0] *= alpha; out[{d}].thread_elements()[1] *= alpha;')
-        for j in range(0, 8, 4):
+        for j in range(0, 8, 2):
             vals.append('{ simdgroup_half8x8 partial(half(0.0f));')
-            for z in range(j, j + 4):
+            for z in range(j, j + 2):
                 vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{z*8*64+d*8},64); simdgroup_multiply_accumulate(partial, probs[{z}], vm, partial); }}')
             vals.append(f'out[{d}].thread_elements()[0] += float(partial.thread_elements()[0]); out[{d}].thread_elements()[1] += float(partial.thread_elements()[1]); }}')
     src = src.replace('VALUE', '\n'.join(vals))

```

</details>

`python bench.py`  
→     _shader.attention(q, k, v, out, threads=(16*256, 64, 1), group_size=(256, 1, 1))  
→     return out

> The mask branch measured 4.08 ms, which is close to the original after accounting for the reference timing. I’m testing query scaling and faster exponentials, then a slightly longer half precision value sum; I’ll check the latter on fresh inputs for its higher rounding error.

- edited `solution.py`
- `python -c 'import torch; from kernel import kernel; from reference import inputs, reference; errors=[]; torch.set_grad_enabled(False)`
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -65,14 +65,10 @@
         for d in range(8):
             dot.append(f'{{ simdgroup_half8x8 km; simdgroup_load(km, keys+{j*8*64+d*8}, 64, ulong2(0), true); simdgroup_multiply_accumulate(scores[{j}], qm[{d}], km, scores[{j}]); }}')
     src = src.replace('DOT', '\n'.join(dot))
-    scale = ['if (start == first) {']
+    scale = []
     for j in range(8):
         for e in range(2):
-            scale.append(f'if (start+{j*8}+col+{e} > qr) scores[{j}].thread_elements()[{e}] = -INFINITY;')
-    scale.append('}')
-    for j in range(8):
-        for e in range(2):
-            scale.append(f'{{ float x = scores[{j}].thread_elements()[{e}]; tilemax = max(tilemax,x); }}')
+            scale.append(f'{{ float x = scores[{j}].thread_elements()[{e}] ; if (start == first && start+{j*8}+col+{e} > qr) x = -INFINITY; scores[{j}].thread_elements()[{e}] = x; tilemax = max(tilemax,x); }}')
     src = src.replace('SCALE', '\n'.join(scale))
     probs = []
     for j in range(8):
@@ -82,11 +78,8 @@
     vals = []
     for d in range(8):
         vals.append(f'out[{d}].thread_elements()[0] *= alpha; out[{d}].thread_elements()[1] *= alpha;')
-        for j in range(0, 8, 2):
-            vals.append('{ simdgroup_half8x8 partial(half(0.0f));')
-            for z in range(j, j + 2):
-                vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{z*8*64+d*8},64); simdgroup_multiply_accumulate(partial, probs[{z}], vm, partial); }}')
-            vals.append(f'out[{d}].thread_elements()[0] += float(partial.thread_elements()[0]); out[{d}].thread_elements()[1] += float(partial.thread_elements()[1]); }}')
+        for j in range(8):
+            vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{j*8*64+d*8},64); simdgroup_multiply_accumulate(out[{d}], probs[{j}], vm, out[{d}]); }}')
     src = src.replace('VALUE', '\n'.join(vals))
     src = src.replace('STORE', '\n'.join(f'O[base+qr*64+{d*8}+col+{e}] = out[{d}].thread_elements()[{e}] / denom;' for d in range(8) for e in range(2)))
     return src

```

</details>

`python bench.py`  
→     _shader.attention(q, k, v, out, threads=(16*256, 64, 1), group_size=(256, 1, 1))  
→     return out

- `python -c 'import torch; from kernel import kernel; from reference import inputs, reference; errors=[]; torch.set_grad_enabled(False)`
- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text().replace(\"range(0, 8, 4)\", \"range(0, 8, 2)\").replace(\"range(j, j + 4)\", \"ra...`
### Bench run 11

`python bench.py`  
→     _shader.attention(q, k, v, out, threads=(16*256, 64, 1), group_size=(256, 1, 1))  
→     return out

- `python -c 'from pathlib import Path; Path('\\''kernel.py'\\'').write_text(\"import torch\\n\\n\\ndef _source():\\n    src = r'\\'''\\'''\\''\\n#include <meta...`
- `python check.py`  
→ OK: relative error 3.31e-04 (allowed 2e-03)
### Bench run 12

`python bench.py`  
→ run 9 of 10 submitted.  
→ attention: correct (relative error 1.2e-03). reference 67.74 ms, kernel 3.71 ms, speedup 18.27x

- `python -c 'import torch; from kernel import kernel; from reference import inputs, reference; errors=[]; torch.set_grad_enabled(False)`
> The final comparison favors 16-key half precision partial sums: 3.71 ms and 18.27× speedup, versus 3.86 ms for full float32 value accumulation. I’m keeping the 16-key version, checking it on fresh inputs, and rewriting the lessons with all ten results.

- `python -c 'from pathlib import Path; Path('\\''kernel.py'\\'').write_text(\"import torch\\n\\n\\ndef _source():\\n    src = r'\\'''\\'''\\''\\n#include <meta...`
- edited `solution.py`
- `python -c 'import torch; from kernel import kernel; from reference import inputs, reference; errors=[]; torch.set_grad_enabled(False)`
- `cat LESSONS.md`
- `python check.py`  
→ OK: relative error 6.35e-04 (allowed 2e-03)
### Bench run 13

`python -c 'from pathlib import Path; Path('\\''LESSONS.md'\\'').write_text(\"The workload is contiguous float32 causal attention on MPS: eight batches, eight heads, 1024 positions, 64 channels. Only time with bench.py. GPU sharing changes timings; compare absolute times and speedups and treat small differences cautiously.\\n\\nUse a fused Metal shader compiled at import. The best configuration uses 64 queries and 64 keys, eight SIMD groups, and 256 threads. Each SIMD group handles eight rows. Explicitly unroll constant matrix operations. Keep queries, float32 scores, half probabilities, and float32 output in registers. Stage inputs through four-element vector loads into separate shared half key and value buffers. Reuse key storage after loading queries. Use stable float32 online softmax, skip future tiles, and mask individual elements only on diagonal tiles. Matrix rows span four lanes; reductions need exclusive-or shuffle offsets one and eight.\\n\\nThis round used ten benchmarks. Kernel times in milliseconds: float32 value accumulation, 3.68; sixteen-key half partial sums, 3.81; 32-key tiles, 3.77; direct query loads, 3.90; 128-query tiles, 4.63 during GPU contention; separate diagonal masking branch, 4.08; query prescaling and fast exponentials, 3.89; 32-key half partial sums, 3.69; final sixteen-key partial sums with prescaling, 3.71; prescaled float32 value accumulation, 3.86.\\n\\nFinal speedup was 18.27 times against 67.74 milliseconds. Benchmark error was 0.0012. Thirty fresh checks passed with maximum error 0.001304; final check also passed. Thirty-two-key half accumulation reached 0.0019, too close to tolerance. Keep dot products and running outputs float32.\\n\\nEarlier rounds measured built-in attention around 17 milliseconds, or 10.5–11.7 with half inputs. Shared scores and missing unrolling were much slower. Preconversion and larger tiles generally regressed. Earlier optimized fused versions reached 3.88–4.10 milliseconds.\\n\")`  


- `wc -w LESSONS.md`
> Final kernel: **3.71 ms, 18.27× speedup**. Thirty fresh-input checks passed; final `check.py` passed with relative error 0.000635. Rewrote `LESSONS.md` within 300 words.
> 
> EDIT: The final kernel.py uses fused tiled Metal causal attention with register softmax, half precision matrix operands, and float32 running accumulators.

