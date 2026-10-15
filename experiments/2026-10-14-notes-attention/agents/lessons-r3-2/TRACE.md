# Agent `lessons-r3-2`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what is still true from the notes you were given. At most 300 words, in words, no code.

14 min, 85,644 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 3.743 ms | 17.82x |
| run2 | 3.904 ms | 17.07x |
| run3 | 3.77 ms | 18.13x |
| run4 | 3.821 ms | 17.96x |
| run5 | 4.212 ms | 15.97x |
| run6 | 4.021 ms | 17.25x |
| run7 | 3.862 ms | 18.48x |
| run8 | 3.665 ms | 18.53x |
| run9 | 3.973 ms | 17.66x |
| run10 | 4.004 ms | 17.03x |
| final | 3.941 ms | 17.38x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the reference, kernel, and prior lessons, then optimize and benchmark within the 10-run limit. I’ll finish with a correctness check and update the lessons.

### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -1,10 +1,105 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
+"""Fused causal attention for the M4 GPU."""
 import torch
 
 
+def _source():
+    n = 8
+    code = r'''
+#include <metal_stdlib>
+#include <metal_simdgroup_matrix>
+using namespace metal;
+kernel void attention(device const float *q [[buffer(0)]],
+                      device const float *k [[buffer(1)]],
+                      device const float *v [[buffer(2)]],
+                      device float *out [[buffer(3)]],
+                      uint3 group [[threadgroup_position_in_grid]],
+                      uint tid [[thread_index_in_threadgroup]],
+                      uint lane [[thread_index_in_simdgroup]],
+                      uint sg [[simdgroup_index_in_threadgroup]]) {
+    const uint qb = group.x * 64;
+    const uint base = group.y * 1024 * 64;
+    const uint row = ((lane >> 1) & 3) + ((lane >> 2) & 4);
+    const uint col = (lane & 1) * 2 + (lane & 8) / 2;
+    const uint qr = qb + sg * 8 + row;
+    threadgroup half keys[64 * 64];
+    threadgroup half vals[64 * 64];
+    simdgroup_half8x8 Q[8], P[8];
+    simdgroup_float8x8 S[8], O[8];
+    float maximum = -INFINITY, denom = 0.0f;
+'''
+    for d in range(8):
+        code += f'''    {{ float2 x = *((device const float2 *)(q + base + qr * 64 + {d*8} + col));
+    Q[{d}].thread_elements()[0] = half(x.x);
+    Q[{d}].thread_elements()[1] = half(x.y); }}
+    O[{d}] = simdgroup_float8x8(0.0f);
+'''
+    code += '''    for (uint kb = 0; kb <= qb; kb += 64) {
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for (uint i = tid; i < 1024; i += 256) {
+            *((threadgroup half4 *)(keys + i * 4)) = half4(*((device const float4 *)(k + base + kb * 64 + i * 4)));
+            *((threadgroup half4 *)(vals + i * 4)) = half4(*((device const float4 *)(v + base + kb * 64 + i * 4)));
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+'''
+    for j in range(n):
+        code += f'        S[{j}] = simdgroup_float8x8(0.0f);\n'
+    for d in range(8):
+        code += '        { simdgroup_half8x8 K;\n'
+        for j in range(n):
+            code += f'''        simdgroup_load(K, keys + {j*8*64+d*8}, 64, ulong2(0), true);
+        simdgroup_multiply_accumulate(S[{j}], Q[{d}], K, S[{j}]);
+'''
+        code += '        }\n'
+    code += '        float tilemax = -INFINITY;\n'
+    for j in range(n):
+        code += f'''        S[{j}].thread_elements()[0] *= 0.125f;
+        S[{j}].thread_elements()[1] *= 0.125f;
+        if (kb == qb) {{
+            if (kb + {j*8} + col > qr) S[{j}].thread_elements()[0] = -INFINITY;
+            if (kb + {j*8} + col + 1 > qr) S[{j}].thread_elements()[1] = -INFINITY;
+        }}
+        tilemax = max(tilemax, max(S[{j}].thread_elements()[0], S[{j}].thread_elements()[1]));
+'''
+    code += '''        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
+        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
+        float newmax = max(maximum, tilemax);
+        float correction = fast::exp(maximum - newmax);
+        float tilesum = 0.0f;
+'''
+    for j in range(n):
+        code += f'''        {{ float a = fast::exp(S[{j}].thread_elements()[0] - newmax);
+        float b = fast::exp(S[{j}].thread_elements()[1] - newmax);
+        tilesum += a + b;
+        P[{j}].thread_elements()[0] = half(a);
+        P[{j}].thread_elements()[1] = half(b); }}
+'''
+    code += '''        tilesum += simd_shuffle_xor(tilesum, 1);
+        tilesum += simd_shuffle_xor(tilesum, 8);
+        denom = denom * correction + tilesum;
+        maximum = newmax;
+'''
+    for d in range(8):
+        code += f'''        O[{d}].thread_elements()[0] *= correction;
+        O[{d}].thread_elements()[1] *= correction;
+'''
+    for j in range(n):
+        code += '        { simdgroup_half8x8 V;\n'
+        for d in range(8):
+            code += f'''        simdgroup_load(V, vals + {j*8*64+d*8}, 64);
+        simdgroup_multiply_accumulate(O[{d}], P[{j}], V, O[{d}]);
+'''
+        code += '        }\n'
+    code += '    }\n'
+    for d in range(8):
+        code += f'''    *((device float2 *)(out + base + qr * 64 + {d*8} + col)) = float2(O[{d}].thread_elements()[0], O[{d}].thread_elements()[1]) / denom;
+'''
+    return code + '}\n'
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
+    _lib.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
+    return out

```

</details>

`pwd; rg --files -g 'AGENTS.md' -g 'kernel.py' -g 'reference.py' -g 'bench.py' -g 'check.py' -g 'LESSONS.md`  
→ check.py  
→ kernel.py

### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -82,13 +82,16 @@
         code += f'''        O[{d}].thread_elements()[0] *= correction;
         O[{d}].thread_elements()[1] *= correction;
 '''
-    for j in range(n):
-        code += '        { simdgroup_half8x8 V;\n'
-        for d in range(8):
-            code += f'''        simdgroup_load(V, vals + {j*8*64+d*8}, 64);
-        simdgroup_multiply_accumulate(O[{d}], P[{j}], V, O[{d}]);
+    for d in range(8):
+        for j in range(0, n, 2):
+            code += f'''        {{ simdgroup_half8x8 V, R(half(0));
+        simdgroup_load(V, vals + {j*8*64+d*8}, 64);
+        simdgroup_multiply_accumulate(R, P[{j}], V, R);
+        simdgroup_load(V, vals + {(j+1)*8*64+d*8}, 64);
+        simdgroup_multiply_accumulate(R, P[{j+1}], V, R);
+        O[{d}].thread_elements()[0] += float(R.thread_elements()[0]);
+        O[{d}].thread_elements()[1] += float(R.thread_elements()[1]); }}
 '''
-        code += '        }\n'
     code += '    }\n'
     for d in range(8):
         code += f'''    *((device float2 *)(out + base + qr * 64 + {d*8} + col)) = float2(O[{d}].thread_elements()[0], O[{d}].thread_elements()[1]) / denom;

```

</details>

`cat kernel.py reference.py bench.py check.py LESSONS.md`  
→ Use eight SIMD groups, 256 threads, shared half key/value tiles, register queries and probabilities, float32 scores, stable float32 online softmax, and float32 running output. Skip future key tiles. On M4, matrix lanes own two adjacent columns; row reductions use lane shuffle offsets one and eight.  
→ Long half precision accumulation is unsafe. Restrict it to sixteen-key partial sums. Thirty fresh-input checks passed; the final twenty had maximum relative error 0.00114, below 0.002. Benchmark error was 0.0012.

- `python - <<'PY` (exited 1 in 0ms:)
- `python -c 'import torch; print(torch.__version__); print(torch.mps.compile_shader.__doc__)`
> The prior lessons point to fused Metal attention with register-held scores and explicit loop unrolling. I’ll start there, then test tile sizes and accumulation choices.
> 
> 2026-10-14T13:11:14.146375Z ERROR codex_core::tools::router: error=apply_patch verification failed: invalid patch: multiple operations target ./kernel.py

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.18e-04 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -52,8 +52,8 @@
         code += '        }\n'
     code += '        float tilemax = -INFINITY;\n'
     for j in range(n):
-        code += f'''        S[{j}].thread_elements()[0] *= 0.125f;
-        S[{j}].thread_elements()[1] *= 0.125f;
+        code += f'''        S[{j}].thread_elements()[0] *= 0.18033688011112042f;
+        S[{j}].thread_elements()[1] *= 0.18033688011112042f;
         if (kb == qb) {{
             if (kb + {j*8} + col > qr) S[{j}].thread_elements()[0] = -INFINITY;
             if (kb + {j*8} + col + 1 > qr) S[{j}].thread_elements()[1] = -INFINITY;
@@ -63,12 +63,12 @@
     code += '''        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
         tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
         float newmax = max(maximum, tilemax);
-        float correction = fast::exp(maximum - newmax);
+        float correction = fast::exp2(maximum - newmax);
         float tilesum = 0.0f;
 '''
     for j in range(n):
-        code += f'''        {{ float a = fast::exp(S[{j}].thread_elements()[0] - newmax);
-        float b = fast::exp(S[{j}].thread_elements()[1] - newmax);
+        code += f'''        {{ float a = fast::exp2(S[{j}].thread_elements()[0] - newmax);
+        float b = fast::exp2(S[{j}].thread_elements()[1] - newmax);
         tilesum += a + b;
         P[{j}].thread_elements()[0] = half(a);
         P[{j}].thread_elements()[1] = half(b); }}
@@ -82,16 +82,13 @@
         code += f'''        O[{d}].thread_elements()[0] *= correction;
         O[{d}].thread_elements()[1] *= correction;
 '''
-    for d in range(8):
-        for j in range(0, n, 2):
-            code += f'''        {{ simdgroup_half8x8 V, R(half(0));
-        simdgroup_load(V, vals + {j*8*64+d*8}, 64);
-        simdgroup_multiply_accumulate(R, P[{j}], V, R);
-        simdgroup_load(V, vals + {(j+1)*8*64+d*8}, 64);
-        simdgroup_multiply_accumulate(R, P[{j+1}], V, R);
-        O[{d}].thread_elements()[0] += float(R.thread_elements()[0]);
-        O[{d}].thread_elements()[1] += float(R.thread_elements()[1]); }}
+    for j in range(n):
+        code += '        { simdgroup_half8x8 V;\n'
+        for d in range(8):
+            code += f'''        simdgroup_load(V, vals + {j*8*64+d*8}, 64);
+        simdgroup_multiply_accumulate(O[{d}], P[{j}], V, O[{d}]);
 '''
+        code += '        }\n'
     code += '    }\n'
     for d in range(8):
         code += f'''    *((device float2 *)(out + base + qr * 64 + {d*8} + col)) = float2(O[{d}].thread_elements()[0], O[{d}].thread_elements()[1]) / denom;

```

</details>

`python bench.py`  
→ run 1 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 66.70 ms, kernel 3.74 ms, speedup 17.82x

> The first fused version passed at 3.74 ms, a 17.82× speedup. I’ll next test half-precision value products over short spans, while keeping the running output in float32.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 7.17e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -1,9 +1,11 @@
 """Fused causal attention for the M4 GPU."""
 import torch
 
+_QROWS = 64
+_KCOLS = 32
 
 def _source():
-    n = 8
+    n = _KCOLS // 8
     code = r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
@@ -16,15 +18,15 @@
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
-    const uint qb = group.x * 64;
+    const uint qb = group.x * QROWS;
     const uint base = group.y * 1024 * 64;
     const uint row = ((lane >> 1) & 3) + ((lane >> 2) & 4);
     const uint col = (lane & 1) * 2 + (lane & 8) / 2;
     const uint qr = qb + sg * 8 + row;
-    threadgroup half keys[64 * 64];
-    threadgroup half vals[64 * 64];
-    simdgroup_half8x8 Q[8], P[8];
-    simdgroup_float8x8 S[8], O[8];
+    threadgroup half keys[KCOLS * 64];
+    threadgroup half vals[KCOLS * 64];
+    simdgroup_half8x8 Q[8], P[KCOLS / 8];
+    simdgroup_float8x8 S[KCOLS / 8], O[8];
     float maximum = -INFINITY, denom = 0.0f;
 '''
     for d in range(8):
@@ -33,9 +35,9 @@
     Q[{d}].thread_elements()[1] = half(x.y); }}
     O[{d}] = simdgroup_float8x8(0.0f);
 '''
-    code += '''    for (uint kb = 0; kb <= qb; kb += 64) {
+    code += '''    for (uint kb = 0; kb < qb + QROWS; kb += KCOLS) {
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < 1024; i += 256) {
+        for (uint i = tid; i < KCOLS * 16; i += QROWS * 4) {
             *((threadgroup half4 *)(keys + i * 4)) = half4(*((device const float4 *)(k + base + kb * 64 + i * 4)));
             *((threadgroup half4 *)(vals + i * 4)) = half4(*((device const float4 *)(v + base + kb * 64 + i * 4)));
         }
@@ -54,7 +56,7 @@
     for j in range(n):
         code += f'''        S[{j}].thread_elements()[0] *= 0.18033688011112042f;
         S[{j}].thread_elements()[1] *= 0.18033688011112042f;
-        if (kb == qb) {{
+        if (kb + KCOLS > qb) {{
             if (kb + {j*8} + col > qr) S[{j}].thread_elements()[0] = -INFINITY;
             if (kb + {j*8} + col + 1 > qr) S[{j}].thread_elements()[1] = -INFINITY;
         }}
@@ -93,7 +95,7 @@
     for d in range(8):
         code += f'''    *((device float2 *)(out + base + qr * 64 + {d*8} + col)) = float2(O[{d}].thread_elements()[0], O[{d}].thread_elements()[1]) / denom;
 '''
-    return code + '}\n'
+    return (code + '}\n').replace('QROWS', str(_QROWS)).replace('KCOLS', str(_KCOLS))
 
 
 _lib = torch.mps.compile_shader(_source())
@@ -101,5 +103,5 @@
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _lib.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
+    _lib.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(_QROWS * 4, 1, 1))
     return out

```

</details>

`python bench.py`  
→ run 2 of 10 submitted.  
→ attention: correct (relative error 1.2e-03). reference 66.65 ms, kernel 3.90 ms, speedup 17.07x

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=Path(\"queue/run1.py\").read_text(); s=s.replace(\"*= 0.125f\", \"*= 0.18033688011112042f\").re...`
- `python check.py`  
→ OK: relative error 3.91e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -1,8 +1,8 @@
 """Fused causal attention for the M4 GPU."""
 import torch
 
-_QROWS = 64
-_KCOLS = 32
+_QROWS = 128
+_KCOLS = 64
 
 def _source():
     n = _KCOLS // 8

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 68.35 ms, kernel 3.77 ms, speedup 18.13x

> Short half-precision value sums took 3.90 ms and increased error. Float32 accumulation remains the better choice. Base-two exponentials measured similarly at 3.77 ms; I’ll now compare tile sizes.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.10e-04 (allowed 2e-03)
- edited `solution.py`
- `python check.py`  
→ run 4 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 68.64 ms, kernel 3.82 ms, speedup 17.96x
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -1,8 +1,9 @@
 """Fused causal attention for the M4 GPU."""
 import torch
 
-_QROWS = 128
+_QROWS = 64
 _KCOLS = 64
+_STRIDE = 72
 
 def _source():
     n = _KCOLS // 8
@@ -23,8 +24,8 @@
     const uint row = ((lane >> 1) & 3) + ((lane >> 2) & 4);
     const uint col = (lane & 1) * 2 + (lane & 8) / 2;
     const uint qr = qb + sg * 8 + row;
-    threadgroup half keys[KCOLS * 64];
-    threadgroup half vals[KCOLS * 64];
+    threadgroup half keys[KCOLS * STRIDE];
+    threadgroup half *vals = keys;
     simdgroup_half8x8 Q[8], P[KCOLS / 8];
     simdgroup_float8x8 S[KCOLS / 8], O[8];
     float maximum = -INFINITY, denom = 0.0f;
@@ -38,8 +39,7 @@
     code += '''    for (uint kb = 0; kb < qb + QROWS; kb += KCOLS) {
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for (uint i = tid; i < KCOLS * 16; i += QROWS * 4) {
-            *((threadgroup half4 *)(keys + i * 4)) = half4(*((device const float4 *)(k + base + kb * 64 + i * 4)));
-            *((threadgroup half4 *)(vals + i * 4)) = half4(*((device const float4 *)(v + base + kb * 64 + i * 4)));
+            *((threadgroup half4 *)(keys + (i / 16) * STRIDE + (i % 16) * 4)) = half4(*((device const float4 *)(k + base + kb * 64 + i * 4)));
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
 '''
@@ -48,11 +48,17 @@
     for d in range(8):
         code += '        { simdgroup_half8x8 K;\n'
         for j in range(n):
-            code += f'''        simdgroup_load(K, keys + {j*8*64+d*8}, 64, ulong2(0), true);
+            code += f'''        simdgroup_load(K, keys + {j*8*_STRIDE+d*8}, STRIDE, ulong2(0), true);
         simdgroup_multiply_accumulate(S[{j}], Q[{d}], K, S[{j}]);
 '''
         code += '        }\n'
-    code += '        float tilemax = -INFINITY;\n'
+    code += '''        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for (uint i = tid; i < KCOLS * 16; i += QROWS * 4) {
+            *((threadgroup half4 *)(vals + (i / 16) * STRIDE + (i % 16) * 4)) = half4(*((device const float4 *)(v + base + kb * 64 + i * 4)));
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        float tilemax = -INFINITY;
+'''
     for j in range(n):
         code += f'''        S[{j}].thread_elements()[0] *= 0.18033688011112042f;
         S[{j}].thread_elements()[1] *= 0.18033688011112042f;
@@ -87,7 +93,7 @@
     for j in range(n):
         code += '        { simdgroup_half8x8 V;\n'
         for d in range(8):
-            code += f'''        simdgroup_load(V, vals + {j*8*64+d*8}, 64);
+            code += f'''        simdgroup_load(V, vals + {j*8*_STRIDE+d*8}, STRIDE);
         simdgroup_multiply_accumulate(O[{d}], P[{j}], V, O[{d}]);
 '''
         code += '        }\n'
@@ -95,7 +101,7 @@
     for d in range(8):
         code += f'''    *((device float2 *)(out + base + qr * 64 + {d*8} + col)) = float2(O[{d}].thread_elements()[0], O[{d}].thread_elements()[1]) / denom;
 '''
-    return (code + '}\n').replace('QROWS', str(_QROWS)).replace('KCOLS', str(_KCOLS))
+    return (code + '}\n').replace('QROWS', str(_QROWS)).replace('KCOLS', str(_KCOLS)).replace('STRIDE', str(_STRIDE))
 
 
 _lib = torch.mps.compile_shader(_source())

```

</details>

`python bench.py`  
→ run 5 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 67.25 ms, kernel 4.21 ms, speedup 15.97x

> Tile changes have not helped: 32-key tiles took 3.82 ms, and 128-query tiles took 4.21 ms. I’ll next test padded shared-memory rows to reduce possible bank conflicts during matrix loads.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.06e-04 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -2,7 +2,7 @@
 import torch
 
 _QROWS = 64
-_KCOLS = 64
+_KCOLS = 32
 _STRIDE = 72
 
 def _source():
@@ -25,7 +25,7 @@
     const uint col = (lane & 1) * 2 + (lane & 8) / 2;
     const uint qr = qb + sg * 8 + row;
     threadgroup half keys[KCOLS * STRIDE];
-    threadgroup half *vals = keys;
+    threadgroup half vals[KCOLS * STRIDE];
     simdgroup_half8x8 Q[8], P[KCOLS / 8];
     simdgroup_float8x8 S[KCOLS / 8], O[8];
     float maximum = -INFINITY, denom = 0.0f;
@@ -40,6 +40,7 @@
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for (uint i = tid; i < KCOLS * 16; i += QROWS * 4) {
             *((threadgroup half4 *)(keys + (i / 16) * STRIDE + (i % 16) * 4)) = half4(*((device const float4 *)(k + base + kb * 64 + i * 4)));
+            *((threadgroup half4 *)(vals + (i / 16) * STRIDE + (i % 16) * 4)) = half4(*((device const float4 *)(v + base + kb * 64 + i * 4)));
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
 '''
@@ -52,13 +53,7 @@
         simdgroup_multiply_accumulate(S[{j}], Q[{d}], K, S[{j}]);
 '''
         code += '        }\n'
-    code += '''        threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < KCOLS * 16; i += QROWS * 4) {
-            *((threadgroup half4 *)(vals + (i / 16) * STRIDE + (i % 16) * 4)) = half4(*((device const float4 *)(v + base + kb * 64 + i * 4)));
-        }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        float tilemax = -INFINITY;
-'''
+    code += '        float tilemax = -INFINITY;\n'
     for j in range(n):
         code += f'''        S[{j}].thread_elements()[0] *= 0.18033688011112042f;
         S[{j}].thread_elements()[1] *= 0.18033688011112042f;

```

</details>

`python bench.py`  
→ run 6 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 69.35 ms, kernel 4.02 ms, speedup 17.25x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.38e-04 (allowed 2e-03)
- `python -c 'from pathlib import Path; Path(\"kernel.py\").write_text(Path(\"queue/run1.py\").read_text())`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 7.24e-04 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -1,12 +1,9 @@
 """Fused causal attention for the M4 GPU."""
 import torch
 
-_QROWS = 64
-_KCOLS = 32
-_STRIDE = 72
 
 def _source():
-    n = _KCOLS // 8
+    n = 8
     code = r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
@@ -19,15 +16,15 @@
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
-    const uint qb = group.x * QROWS;
+    const uint qb = group.x * 64;
     const uint base = group.y * 1024 * 64;
     const uint row = ((lane >> 1) & 3) + ((lane >> 2) & 4);
     const uint col = (lane & 1) * 2 + (lane & 8) / 2;
     const uint qr = qb + sg * 8 + row;
-    threadgroup half keys[KCOLS * STRIDE];
-    threadgroup half vals[KCOLS * STRIDE];
-    simdgroup_half8x8 Q[8], P[KCOLS / 8];
-    simdgroup_float8x8 S[KCOLS / 8], O[8];
+    threadgroup half keys[64 * 64];
+    threadgroup half vals[64 * 64];
+    simdgroup_half8x8 Q[8], P[8];
+    simdgroup_float8x8 S[8], O[8];
     float maximum = -INFINITY, denom = 0.0f;
 '''
     for d in range(8):
@@ -36,11 +33,11 @@
     Q[{d}].thread_elements()[1] = half(x.y); }}
     O[{d}] = simdgroup_float8x8(0.0f);
 '''
-    code += '''    for (uint kb = 0; kb < qb + QROWS; kb += KCOLS) {
+    code += '''    for (uint kb = 0; kb <= qb; kb += 64) {
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for (uint i = tid; i < KCOLS * 16; i += QROWS * 4) {
-            *((threadgroup half4 *)(keys + (i / 16) * STRIDE + (i % 16) * 4)) = half4(*((device const float4 *)(k + base + kb * 64 + i * 4)));
-            *((threadgroup half4 *)(vals + (i / 16) * STRIDE + (i % 16) * 4)) = half4(*((device const float4 *)(v + base + kb * 64 + i * 4)));
+        for (uint i = tid; i < 1024; i += 256) {
+            *((threadgroup half4 *)(keys + i * 4)) = half4(*((device const float4 *)(k + base + kb * 64 + i * 4)));
+            *((threadgroup half4 *)(vals + i * 4)) = half4(*((device const float4 *)(v + base + kb * 64 + i * 4)));
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
 '''
@@ -49,15 +46,15 @@
     for d in range(8):
         code += '        { simdgroup_half8x8 K;\n'
         for j in range(n):
-            code += f'''        simdgroup_load(K, keys + {j*8*_STRIDE+d*8}, STRIDE, ulong2(0), true);
+            code += f'''        simdgroup_load(K, keys + {j*8*64+d*8}, 64, ulong2(0), true);
         simdgroup_multiply_accumulate(S[{j}], Q[{d}], K, S[{j}]);
 '''
         code += '        }\n'
     code += '        float tilemax = -INFINITY;\n'
     for j in range(n):
-        code += f'''        S[{j}].thread_elements()[0] *= 0.18033688011112042f;
-        S[{j}].thread_elements()[1] *= 0.18033688011112042f;
-        if (kb + KCOLS > qb) {{
+        code += f'''        S[{j}].thread_elements()[0] *= 0.125f;
+        S[{j}].thread_elements()[1] *= 0.125f;
+        if (kb == qb) {{
             if (kb + {j*8} + col > qr) S[{j}].thread_elements()[0] = -INFINITY;
             if (kb + {j*8} + col + 1 > qr) S[{j}].thread_elements()[1] = -INFINITY;
         }}
@@ -66,12 +63,12 @@
     code += '''        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
         tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
         float newmax = max(maximum, tilemax);
-        float correction = fast::exp2(maximum - newmax);
+        float correction = fast::exp(maximum - newmax);
         float tilesum = 0.0f;
 '''
     for j in range(n):
-        code += f'''        {{ float a = fast::exp2(S[{j}].thread_elements()[0] - newmax);
-        float b = fast::exp2(S[{j}].thread_elements()[1] - newmax);
+        code += f'''        {{ float a = fast::exp(S[{j}].thread_elements()[0] - newmax);
+        float b = fast::exp(S[{j}].thread_elements()[1] - newmax);
         tilesum += a + b;
         P[{j}].thread_elements()[0] = half(a);
         P[{j}].thread_elements()[1] = half(b); }}
@@ -85,18 +82,25 @@
         code += f'''        O[{d}].thread_elements()[0] *= correction;
         O[{d}].thread_elements()[1] *= correction;
 '''
-    for j in range(n):
-        code += '        { simdgroup_half8x8 V;\n'
+    for j in range(0, n, 2):
+        code += '        { simdgroup_half8x8 V, R[8];\n'
         for d in range(8):
-            code += f'''        simdgroup_load(V, vals + {j*8*_STRIDE+d*8}, STRIDE);
-        simdgroup_multiply_accumulate(O[{d}], P[{j}], V, O[{d}]);
+            code += f'        R[{d}] = simdgroup_half8x8(half(0));\n'
+        for jj in (j, j + 1):
+            for d in range(8):
+                code += f'''        simdgroup_load(V, vals + {jj*8*64+d*8}, 64);
+        simdgroup_multiply_accumulate(R[{d}], P[{jj}], V, R[{d}]);
+'''
+        for d in range(8):
+            code += f'''        O[{d}].thread_elements()[0] += float(R[{d}].thread_elements()[0]);
+        O[{d}].thread_elements()[1] += float(R[{d}].thread_elements()[1]);
 '''
         code += '        }\n'
     code += '    }\n'
     for d in range(8):
         code += f'''    *((device float2 *)(out + base + qr * 64 + {d*8} + col)) = float2(O[{d}].thread_elements()[0], O[{d}].thread_elements()[1]) / denom;
 '''
-    return (code + '}\n').replace('QROWS', str(_QROWS)).replace('KCOLS', str(_KCOLS)).replace('STRIDE', str(_STRIDE))
+    return code + '}\n'
 
 
 _lib = torch.mps.compile_shader(_source())
@@ -104,5 +108,5 @@
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _lib.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(_QROWS * 4, 1, 1))
+    _lib.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
     return out

```

</details>

`python bench.py`  


- edited `solution.py`
- `python check.py`  
→ OK: relative error 1.24e-03 (allowed 2e-03)
- `python -c 'import subprocess; [subprocess.run([\"python\", \"check.py\"], check=True) for _ in range(10)]`  
→ OK: relative error 9.04e-04 (allowed 2e-03)  
→ OK: relative error 1.53e-03 (allowed 2e-03)
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -43,11 +43,18 @@
 '''
     for j in range(n):
         code += f'        S[{j}] = simdgroup_float8x8(0.0f);\n'
-    for d in range(8):
-        code += '        { simdgroup_half8x8 K;\n'
+    for d in range(0, 8, 2):
+        code += '        { simdgroup_half8x8 K, R[8];\n'
         for j in range(n):
-            code += f'''        simdgroup_load(K, keys + {j*8*64+d*8}, 64, ulong2(0), true);
-        simdgroup_multiply_accumulate(S[{j}], Q[{d}], K, S[{j}]);
+            code += f'        R[{j}] = simdgroup_half8x8(half(0));\n'
+        for dd in (d, d + 1):
+            for j in range(n):
+                code += f'''        simdgroup_load(K, keys + {j*8*64+dd*8}, 64, ulong2(0), true);
+        simdgroup_multiply_accumulate(R[{j}], Q[{dd}], K, R[{j}]);
+'''
+        for j in range(n):
+            code += f'''        S[{j}].thread_elements()[0] += float(R[{j}].thread_elements()[0]);
+        S[{j}].thread_elements()[1] += float(R[{j}].thread_elements()[1]);
 '''
         code += '        }\n'
     code += '        float tilemax = -INFINITY;\n'
@@ -82,18 +89,11 @@
         code += f'''        O[{d}].thread_elements()[0] *= correction;
         O[{d}].thread_elements()[1] *= correction;
 '''
-    for j in range(0, n, 2):
-        code += '        { simdgroup_half8x8 V, R[8];\n'
+    for j in range(n):
+        code += '        { simdgroup_half8x8 V;\n'
         for d in range(8):
-            code += f'        R[{d}] = simdgroup_half8x8(half(0));\n'
-        for jj in (j, j + 1):
-            for d in range(8):
-                code += f'''        simdgroup_load(V, vals + {jj*8*64+d*8}, 64);
-        simdgroup_multiply_accumulate(R[{d}], P[{jj}], V, R[{d}]);
-'''
-        for d in range(8):
-            code += f'''        O[{d}].thread_elements()[0] += float(R[{d}].thread_elements()[0]);
-        O[{d}].thread_elements()[1] += float(R[{d}].thread_elements()[1]);
+            code += f'''        simdgroup_load(V, vals + {j*8*64+d*8}, 64);
+        simdgroup_multiply_accumulate(O[{d}], P[{j}], V, O[{d}]);
 '''
         code += '        }\n'
     code += '    }\n'

```

</details>

`python bench.py`  
→ run 9 of 10 submitted.  
→ attention: correct (relative error 1.6e-03). reference 70.17 ms, kernel 3.97 ms, speedup 17.66x

- `python -c 'from pathlib import Path; Path(\"kernel.py\").write_text(Path(\"queue/run8.py\").read_text())`
- edited `solution.py`
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -3,6 +3,7 @@
 
 
 def _source():
+    # Emit constant matrix operations explicitly to keep matrices in registers.
     n = 8
     code = r'''
 #include <metal_stdlib>
@@ -18,6 +19,7 @@
                       uint sg [[simdgroup_index_in_threadgroup]]) {
     const uint qb = group.x * 64;
     const uint base = group.y * 1024 * 64;
+    // Each matrix lane owns two adjacent columns; four lanes share one row.
     const uint row = ((lane >> 1) & 3) + ((lane >> 2) & 4);
     const uint col = (lane & 1) * 2 + (lane & 8) / 2;
     const uint qr = qb + sg * 8 + row;
@@ -43,18 +45,11 @@
 '''
     for j in range(n):
         code += f'        S[{j}] = simdgroup_float8x8(0.0f);\n'
-    for d in range(0, 8, 2):
-        code += '        { simdgroup_half8x8 K, R[8];\n'
+    for d in range(8):
+        code += '        { simdgroup_half8x8 K;\n'
         for j in range(n):
-            code += f'        R[{j}] = simdgroup_half8x8(half(0));\n'
-        for dd in (d, d + 1):
-            for j in range(n):
-                code += f'''        simdgroup_load(K, keys + {j*8*64+dd*8}, 64, ulong2(0), true);
-        simdgroup_multiply_accumulate(R[{j}], Q[{dd}], K, R[{j}]);
-'''
-        for j in range(n):
-            code += f'''        S[{j}].thread_elements()[0] += float(R[{j}].thread_elements()[0]);
-        S[{j}].thread_elements()[1] += float(R[{j}].thread_elements()[1]);
+            code += f'''        simdgroup_load(K, keys + {j*8*64+d*8}, 64, ulong2(0), true);
+        simdgroup_multiply_accumulate(S[{j}], Q[{d}], K, S[{j}]);
 '''
         code += '        }\n'
     code += '        float tilemax = -INFINITY;\n'
@@ -89,11 +84,19 @@
         code += f'''        O[{d}].thread_elements()[0] *= correction;
         O[{d}].thread_elements()[1] *= correction;
 '''
-    for j in range(n):
-        code += '        { simdgroup_half8x8 V;\n'
+    for j in range(0, n, 2):
+        # Round only sixteen-key value products to half. The running sum is float.
+        code += '        { simdgroup_half8x8 V, R[8];\n'
         for d in range(8):
-            code += f'''        simdgroup_load(V, vals + {j*8*64+d*8}, 64);
-        simdgroup_multiply_accumulate(O[{d}], P[{j}], V, O[{d}]);
+            code += f'        R[{d}] = simdgroup_half8x8(half(0));\n'
+        for jj in (j, j + 1):
+            for d in range(8):
+                code += f'''        simdgroup_load(V, vals + {jj*8*64+d*8}, 64);
+        simdgroup_multiply_accumulate(R[{d}], P[{jj}], V, R[{d}]);
+'''
+        for d in range(8):
+            code += f'''        O[{d}].thread_elements()[0] += float(R[{d}].thread_elements()[0]);
+        O[{d}].thread_elements()[1] += float(R[{d}].thread_elements()[1]);
 '''
         code += '        }\n'
     code += '    }\n'
@@ -103,6 +106,7 @@
     return code + '}\n'
 
 
+# Compile once, outside the timed invocation.
 _lib = torch.mps.compile_shader(_source())
 
 

```

</details>

`python bench.py`  
→ run 10 of 10 submitted.  
→ attention: correct (relative error 1.2e-03). reference 68.21 ms, kernel 4.00 ms, speedup 17.03x

> The half-precision value variant measured 3.66 ms initially but 4.00 ms on repeat. I’m keeping float32 accumulation, which measured 3.74 ms and retained more accuracy margin. I’ll now check it on fresh inputs and record all ten results.

- `python -c 'from pathlib import Path; Path(\"kernel.py\").write_text(Path(\"queue/run1.py\").read_text())`
- edited `solution.py`
- `python -c 'import subprocess; [subprocess.run([\"python\", \"check.py\"], check=True) for _ in range(20)]`  
→ OK: relative error 3.87e-04 (allowed 2e-03)  
→ OK: relative error 3.51e-04 (allowed 2e-03)
### Bench run 11

`python -c 'from pathlib import Path; Path(\"LESSONS.md\").write_text(\"The workload is contiguous float32 causal attention: eight batches, eight heads, 1024 positions, 64 channels. Supplied baseline: 65.6 milliseconds. Time only through bench.py. Shared GPU load varies; small timing differences are uncertain.\\n\\nCompile a fused Metal shader once at import. Use 64 queries and 64 keys per tile, eight SIMD groups, 256 threads. Keep queries, float32 scores, half probabilities, and float32 output accumulators in registers. Explicitly generate unrolled matrix operations. Cooperatively stage separate half key and value buffers with four-element vector loads. Maintain stable float32 online softmax maxima and denominators. Skip future tiles; mask individual elements only on the diagonal tile. Matrix lanes own two adjacent columns; row reductions use exclusive-or shuffle offsets one and eight.\\n\\nThis round used ten benchmarks. Times in milliseconds: full float32 accumulation, 3.74; sixteen-key half value sums with output columns processed separately, 3.90; base-two exponentials, 3.77; 32-key tiles, 3.82; 128-query tiles, 4.21; padded shared storage reused for keys and values, 4.02; separate padded storage with 32-key tiles, 3.86; interleaved sixteen-key half value sums, 3.66; sixteen-channel half dot-product sums, 3.97; repeat interleaved half value sums, 4.00.\\n\\nFinal kernel retains full float32 accumulation: 3.74 milliseconds versus 66.70, speedup 17.82 times. Benchmark error was 0.00061. Twenty fresh checks passed, maximum error 0.000517 against 0.002. Half value variants had benchmark error 0.0012; short half dot products reached 0.0016.\\n\\nEarlier useful findings: built-in attention took approximately 17 milliseconds in float32 and 10.5–11.7 with half inputs. Unrolling register matrix loops and vector staging produced major improvements. Shared score buffers and unrolled larger tiles generally lost. Long half dot-product or output accumulation can exceed tolerance.\\n\"); print(len(Path(\"LESSONS.md\").read_text().split()), \"words\")`  
→ 263 words

- `python check.py`  
→ OK: relative error 3.49e-04 (allowed 2e-03)
> Measured 3.74 ms, a 17.82× speedup. All 21 final correctness checks passed. Updated LESSONS.md with all ten benchmark results.
> 
> EDIT: kernel.py now computes causal attention using a fused tiled Metal shader with stable softmax and float32 accumulation.

