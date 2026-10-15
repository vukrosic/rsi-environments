# Agent `lessons-r3-3`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what is still true from the notes you were given. At most 300 words, in words, no code.

10 min, 63,027 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 3.704 ms | 18.25x |
| run2 | 3.764 ms | 17.16x |
| run3 | 3.756 ms | 17.26x |
| run4 | 4.194 ms | 16.26x |
| run5 | 4.45 ms | 15.35x |
| run6 | 3.737 ms | 18.1x |
| run7 | 3.895 ms | 17.6x |
| run8 | 3.706 ms | 18.29x |
| run9 | 4.098 ms | 16.52x |
| run10 | 3.815 ms | 18.71x |
| final | 3.876 ms | 17.96x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the kernel, reference, and earlier notes, then use the benchmark runs to tune it. I’ll finish with a correctness check and update LESSONS.md.

- `pwd && rg --files -g '"'!__pycache__'"' -g '"'!*.pyc`
### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -1,10 +1,90 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
+"""Fused, tiled causal attention for Apple GPUs."""
 import torch
 
 
+def _source():
+    s = ['''#include <metal_stdlib>
+#include <metal_simdgroup_matrix>
+using namespace metal;
+kernel void attention(device const float* Q, device const float* K,
+                      device const float* V, device float* O,
+                      uint2 block [[threadgroup_position_in_grid]],
+                      uint tid [[thread_index_in_threadgroup]],
+                      uint lane [[thread_index_in_simdgroup]],
+                      uint sg [[simdgroup_index_in_threadgroup]]) {
+    threadgroup half keys[4096];
+    threadgroup half vals[4096];
+    uint base = block.y * 1024 * 64;
+    uint qr = block.x * 64 + sg * 8;
+    uint r = ((lane >> 1) & 3) + ((lane >> 4) * 4);
+    uint c = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
+    simdgroup_half8x8 q[8];
+    simdgroup_float8x8 out[8];
+    float maximum = -INFINITY, denominator = 0.0f;
+''']
+    for d in range(8):
+        s.append(f'''q[{d}].thread_elements()[0] = half(Q[base+(qr+r)*64+{d*8}+c]);
+q[{d}].thread_elements()[1] = half(Q[base+(qr+r)*64+{d*8}+c+1]);
+out[{d}] = simdgroup_float8x8(0.0f);''')
+    s.append('''for (uint start=0; start<=block.x*64; start+=64) {
+    for (uint i=tid; i<1024; i+=256) {
+        *((threadgroup half4*)keys+i) = half4(*((device const float4*)(K+base+start*64)+i));
+        *((threadgroup half4*)vals+i) = half4(*((device const float4*)(V+base+start*64)+i));
+    }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    simdgroup_float8x8 score[8];
+    simdgroup_half8x8 p[8];
+''')
+    for n in range(8):
+        s.append(f'score[{n}] = simdgroup_float8x8(0.0f);')
+    for d in range(8):
+        s.append(f'simdgroup_half8x8 kk{d};')
+        for n in range(8):
+            s.append(f'''simdgroup_load(kk{d}, keys+{n*8*64+d*8},64,ulong2(0),true);
+simdgroup_multiply_accumulate(score[{n}], q[{d}], kk{d}, score[{n}]);''')
+    s.append('float mx = -INFINITY;')
+    for n in range(8):
+        s.append(f'''score[{n}].thread_elements()[0] *= 0.125f;
+score[{n}].thread_elements()[1] *= 0.125f;
+if (start == block.x*64) {{
+ if (start+{n*8}+c > qr+r) score[{n}].thread_elements()[0] = -INFINITY;
+ if (start+{n*8}+c+1 > qr+r) score[{n}].thread_elements()[1] = -INFINITY;
+}}
+mx=max(mx,max(score[{n}].thread_elements()[0],score[{n}].thread_elements()[1]));''')
+    s.append('''mx=max(mx,simd_shuffle_xor(mx,1));
+mx=max(mx,simd_shuffle_xor(mx,8));
+float nextmax=max(maximum,mx);
+float alpha=fast::exp(maximum-nextmax);
+maximum=nextmax;
+float total=0.0f;''')
+    for n in range(8):
+        s.append(f'''float2 prob{n} = fast::exp(float2(score[{n}].thread_elements()[0],score[{n}].thread_elements()[1])-nextmax);
+p[{n}].thread_elements()[0]=half(prob{n}.x);
+p[{n}].thread_elements()[1]=half(prob{n}.y);
+total += prob{n}.x+prob{n}.y;
+out[{n}].thread_elements()[0] *= alpha;
+out[{n}].thread_elements()[1] *= alpha;''')
+    s.append('''total+=simd_shuffle_xor(total,1);
+total+=simd_shuffle_xor(total,8);
+denominator=denominator*alpha+total;''')
+    for n in range(8):
+        s.append(f'simdgroup_half8x8 vv{n};')
+        for d in range(8):
+            s.append(f'''simdgroup_load(vv{n}, vals+{n*8*64+d*8},64);
+simdgroup_multiply_accumulate(out[{d}],p[{n}],vv{n},out[{d}]);''')
+    s.append('threadgroup_barrier(mem_flags::mem_threadgroup);\n}')
+    for d in range(8):
+        s.append(f'''out[{d}].thread_elements()[0] /= denominator;
+out[{d}].thread_elements()[1] /= denominator;
+simdgroup_store(out[{d}],O+base+qr*64+{d*8},64);''')
+    s.append('}')
+    return '\n'.join(s)
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
+    _lib.attention(q, k, v, out, threads=(16*256,64,1), group_size=(256,1,1))
+    return out

```

</details>

`cat kernel.py reference.py bench.py check.py LESSONS.md`  
→ Use eight SIMD groups, 256 threads, shared half key/value tiles, register queries and probabilities, float32 scores, stable float32 online softmax, and float32 running output. Skip future key tiles. On M4, matrix lanes own two adjacent columns; row reductions use lane shuffle offsets one and eight.  
→ Long half precision accumulation is unsafe. Restrict it to sixteen-key partial sums. Thirty fresh-input checks passed; the final twenty had maximum relative error 0.00114, below 0.002. Benchmark error was 0.0012.

- `python -c 'import torch; print(torch.__version__); print(torch.mps.compile_shader.__doc__)`
- `python - <<'PY` (exited 1 in 0ms:)
- `python -c 'import torch; s=torch.mps.compile_shader(\"\"\"#include <metal_stdlib>`
> The notes point to a fused Metal kernel as the fastest approach. I’m implementing register-held attention tiles with float32 accumulation and causal masking, then I’ll check accuracy before timing it.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.84e-04 (allowed 2e-03)
### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -1,5 +1,8 @@
 """Fused, tiled causal attention for Apple GPUs."""
 import torch
+
+
+_HALF_PARTIAL = True
 
 
 def _source():
@@ -67,10 +70,22 @@
     s.append('''total+=simd_shuffle_xor(total,1);
 total+=simd_shuffle_xor(total,8);
 denominator=denominator*alpha+total;''')
-    for n in range(8):
-        s.append(f'simdgroup_half8x8 vv{n};')
-        for d in range(8):
-            s.append(f'''simdgroup_load(vv{n}, vals+{n*8*64+d*8},64);
+    if _HALF_PARTIAL:
+        for n in range(0,8,2):
+            s.append(f'simdgroup_half8x8 vv{n};')
+            for d in range(8):
+                s.append(f'''simdgroup_half8x8 tmp{n}_{d}(half(0));
+simdgroup_load(vv{n}, vals+{n*8*64+d*8},64);
+simdgroup_multiply_accumulate(tmp{n}_{d},p[{n}],vv{n},tmp{n}_{d});
+simdgroup_load(vv{n}, vals+{(n+1)*8*64+d*8},64);
+simdgroup_multiply_accumulate(tmp{n}_{d},p[{n+1}],vv{n},tmp{n}_{d});
+out[{d}].thread_elements()[0] += float(tmp{n}_{d}.thread_elements()[0]);
+out[{d}].thread_elements()[1] += float(tmp{n}_{d}.thread_elements()[1]);''')
+    else:
+        for n in range(8):
+            s.append(f'simdgroup_half8x8 vv{n};')
+            for d in range(8):
+                s.append(f'''simdgroup_load(vv{n}, vals+{n*8*64+d*8},64);
 simdgroup_multiply_accumulate(out[{d}],p[{n}],vv{n},out[{d}]);''')
     s.append('threadgroup_barrier(mem_flags::mem_threadgroup);\n}')
     for d in range(8):

```

</details>

`python bench.py`  
→ run 1 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 67.59 ms, kernel 3.70 ms, speedup 18.25x

> The first fused version passed and measured 3.70 ms, an 18.25× speedup. I’ll now test shorter half-precision value products and different tile sizes to look for further gains.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 5.98e-04 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -2,7 +2,9 @@
 import torch
 
 
-_HALF_PARTIAL = True
+_HALF_PARTIAL = False
+_BQ = 64
+_BK = 32
 
 
 def _source():
@@ -38,15 +40,15 @@
     simdgroup_float8x8 score[8];
     simdgroup_half8x8 p[8];
 ''')
-    for n in range(8):
+    for n in range(_BK // 8):
         s.append(f'score[{n}] = simdgroup_float8x8(0.0f);')
     for d in range(8):
         s.append(f'simdgroup_half8x8 kk{d};')
-        for n in range(8):
+        for n in range(_BK // 8):
             s.append(f'''simdgroup_load(kk{d}, keys+{n*8*64+d*8},64,ulong2(0),true);
 simdgroup_multiply_accumulate(score[{n}], q[{d}], kk{d}, score[{n}]);''')
     s.append('float mx = -INFINITY;')
-    for n in range(8):
+    for n in range(_BK // 8):
         s.append(f'''score[{n}].thread_elements()[0] *= 0.125f;
 score[{n}].thread_elements()[1] *= 0.125f;
 if (start == block.x*64) {{
@@ -60,18 +62,19 @@
 float alpha=fast::exp(maximum-nextmax);
 maximum=nextmax;
 float total=0.0f;''')
-    for n in range(8):
+    for n in range(_BK // 8):
         s.append(f'''float2 prob{n} = fast::exp(float2(score[{n}].thread_elements()[0],score[{n}].thread_elements()[1])-nextmax);
 p[{n}].thread_elements()[0]=half(prob{n}.x);
 p[{n}].thread_elements()[1]=half(prob{n}.y);
-total += prob{n}.x+prob{n}.y;
-out[{n}].thread_elements()[0] *= alpha;
-out[{n}].thread_elements()[1] *= alpha;''')
+total += prob{n}.x+prob{n}.y;''')
+    for d in range(8):
+        s.append(f'''out[{d}].thread_elements()[0] *= alpha;
+out[{d}].thread_elements()[1] *= alpha;''')
     s.append('''total+=simd_shuffle_xor(total,1);
 total+=simd_shuffle_xor(total,8);
 denominator=denominator*alpha+total;''')
     if _HALF_PARTIAL:
-        for n in range(0,8,2):
+        for n in range(0,_BK // 8,2):
             s.append(f'simdgroup_half8x8 vv{n};')
             for d in range(8):
                 s.append(f'''simdgroup_half8x8 tmp{n}_{d}(half(0));
@@ -82,7 +85,7 @@
 out[{d}].thread_elements()[0] += float(tmp{n}_{d}.thread_elements()[0]);
 out[{d}].thread_elements()[1] += float(tmp{n}_{d}.thread_elements()[1]);''')
     else:
-        for n in range(8):
+        for n in range(_BK // 8):
             s.append(f'simdgroup_half8x8 vv{n};')
             for d in range(8):
                 s.append(f'''simdgroup_load(vv{n}, vals+{n*8*64+d*8},64);
@@ -93,7 +96,13 @@
 out[{d}].thread_elements()[1] /= denominator;
 simdgroup_store(out[{d}],O+base+qr*64+{d*8},64);''')
     s.append('}')
-    return '\n'.join(s)
+    source = '\n'.join(s)
+    source = source.replace('keys[4096]', f'keys[{_BK*64}]').replace('vals[4096]', f'vals[{_BK*64}]')
+    source = source.replace('block.x * 64', f'block.x * {_BQ}')
+    source = source.replace('start<=block.x*64; start+=64', f'start<block.x*{_BQ}+{_BQ}; start+={_BK}')
+    source = source.replace('start == block.x*64', f'start+{_BK} > block.x*{_BQ}')
+    source = source.replace('i<1024; i+=256', f'i<{_BK*16}; i+={_BQ*4}')
+    return source
 
 
 _lib = torch.mps.compile_shader(_source())
@@ -101,5 +110,5 @@
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _lib.attention(q, k, v, out, threads=(16*256,64,1), group_size=(256,1,1))
+    _lib.attention(q, k, v, out, threads=(1024//_BQ*(_BQ*4),64,1), group_size=(_BQ*4,1,1))
     return out

```

</details>

`python bench.py`  
→     _lib.attention(q, k, v, out, threads=(16*256,64,1), group_size=(256,1,1))  
→     return out

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.50e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -3,8 +3,8 @@
 
 
 _HALF_PARTIAL = False
-_BQ = 64
-_BK = 32
+_BQ = 128
+_BK = 64
 
 
 def _source():

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 64.81 ms, kernel 3.76 ms, speedup 17.25x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.70e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -3,8 +3,8 @@
 
 
 _HALF_PARTIAL = False
-_BQ = 128
-_BK = 64
+_BQ = 64
+_BK = 128
 
 
 def _source():
@@ -37,8 +37,8 @@
         *((threadgroup half4*)vals+i) = half4(*((device const float4*)(V+base+start*64)+i));
     }
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    simdgroup_float8x8 score[8];
-    simdgroup_half8x8 p[8];
+    simdgroup_float8x8 score[KEY_MATRICES];
+    simdgroup_half8x8 p[KEY_MATRICES];
 ''')
     for n in range(_BK // 8):
         s.append(f'score[{n}] = simdgroup_float8x8(0.0f);')
@@ -97,6 +97,7 @@
 simdgroup_store(out[{d}],O+base+qr*64+{d*8},64);''')
     s.append('}')
     source = '\n'.join(s)
+    source = source.replace('KEY_MATRICES',str(_BK//8))
     source = source.replace('keys[4096]', f'keys[{_BK*64}]').replace('vals[4096]', f'vals[{_BK*64}]')
     source = source.replace('block.x * 64', f'block.x * {_BQ}')
     source = source.replace('start<=block.x*64; start+=64', f'start<block.x*{_BQ}+{_BQ}; start+={_BK}')

```

</details>

`python bench.py`  
→ run 4 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 68.22 ms, kernel 4.19 ms, speedup 16.27x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 6.02e-04 (allowed 2e-03)
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -3,8 +3,9 @@
 
 
 _HALF_PARTIAL = False
+_HALF_SCORE = True
 _BQ = 64
-_BK = 128
+_BK = 64
 
 
 def _source():
@@ -42,10 +43,22 @@
 ''')
     for n in range(_BK // 8):
         s.append(f'score[{n}] = simdgroup_float8x8(0.0f);')
-    for d in range(8):
-        s.append(f'simdgroup_half8x8 kk{d};')
-        for n in range(_BK // 8):
-            s.append(f'''simdgroup_load(kk{d}, keys+{n*8*64+d*8},64,ulong2(0),true);
+    if _HALF_SCORE:
+        for d in range(0,8,2):
+            s.append(f'simdgroup_half8x8 kk{d};')
+            for n in range(_BK // 8):
+                s.append(f'''simdgroup_half8x8 ss{d}_{n}(half(0));
+simdgroup_load(kk{d}, keys+{n*8*64+d*8},64,ulong2(0),true);
+simdgroup_multiply_accumulate(ss{d}_{n},q[{d}],kk{d},ss{d}_{n});
+simdgroup_load(kk{d}, keys+{n*8*64+(d+1)*8},64,ulong2(0),true);
+simdgroup_multiply_accumulate(ss{d}_{n},q[{d+1}],kk{d},ss{d}_{n});
+score[{n}].thread_elements()[0] += float(ss{d}_{n}.thread_elements()[0]);
+score[{n}].thread_elements()[1] += float(ss{d}_{n}.thread_elements()[1]);''')
+    else:
+        for d in range(8):
+            s.append(f'simdgroup_half8x8 kk{d};')
+            for n in range(_BK // 8):
+                s.append(f'''simdgroup_load(kk{d}, keys+{n*8*64+d*8},64,ulong2(0),true);
 simdgroup_multiply_accumulate(score[{n}], q[{d}], kk{d}, score[{n}]);''')
     s.append('float mx = -INFINITY;')
     for n in range(_BK // 8):

```

</details>

`python bench.py`  
→ run 5 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 68.32 ms, kernel 4.45 ms, speedup 15.35x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 7.14e-04 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -3,7 +3,7 @@
 
 
 _HALF_PARTIAL = False
-_HALF_SCORE = True
+_HALF_SCORE = False
 _BQ = 64
 _BK = 64
 
@@ -116,6 +116,8 @@
     source = source.replace('start<=block.x*64; start+=64', f'start<block.x*{_BQ}+{_BQ}; start+={_BK}')
     source = source.replace('start == block.x*64', f'start+{_BK} > block.x*{_BQ}')
     source = source.replace('i<1024; i+=256', f'i<{_BK*16}; i+={_BQ*4}')
+    source = source.replace('0.125f', '0.18033688011112042f').replace('fast::exp(', 'fast::exp2(')
+    source = source.replace('    for (uint i=tid;', '    #pragma unroll\n    for (uint i=tid;')
     return source
 
 

```

</details>

`python bench.py`  
→ run 6 of 10 submitted.  
→ attention: correct (relative error 1.6e-03). reference 67.63 ms, kernel 3.74 ms, speedup 18.10x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.76e-04 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -63,12 +63,15 @@
     s.append('float mx = -INFINITY;')
     for n in range(_BK // 8):
         s.append(f'''score[{n}].thread_elements()[0] *= 0.125f;
-score[{n}].thread_elements()[1] *= 0.125f;
-if (start == block.x*64) {{
+score[{n}].thread_elements()[1] *= 0.125f;''')
+    s.append('if (start == block.x*64) {')
+    for n in range(_BK // 8):
+        s.append(f'''
  if (start+{n*8}+c > qr+r) score[{n}].thread_elements()[0] = -INFINITY;
- if (start+{n*8}+c+1 > qr+r) score[{n}].thread_elements()[1] = -INFINITY;
-}}
-mx=max(mx,max(score[{n}].thread_elements()[0],score[{n}].thread_elements()[1]));''')
+ if (start+{n*8}+c+1 > qr+r) score[{n}].thread_elements()[1] = -INFINITY;''')
+    s.append('}')
+    for n in range(_BK // 8):
+        s.append(f'mx=max(mx,max(score[{n}].thread_elements()[0],score[{n}].thread_elements()[1]));')
     s.append('''mx=max(mx,simd_shuffle_xor(mx,1));
 mx=max(mx,simd_shuffle_xor(mx,8));
 float nextmax=max(maximum,mx);

```

</details>

`python bench.py`  
→ OK: relative error 4.53e-04 (allowed 2e-03)

### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -4,6 +4,7 @@
 
 _HALF_PARTIAL = False
 _HALF_SCORE = False
+_REUSE_KV = True
 _BQ = 64
 _BK = 64
 
@@ -89,6 +90,12 @@
     s.append('''total+=simd_shuffle_xor(total,1);
 total+=simd_shuffle_xor(total,8);
 denominator=denominator*alpha+total;''')
+    if _REUSE_KV:
+        s.append('''threadgroup_barrier(mem_flags::mem_threadgroup);
+for (uint i=tid; i<1024; i+=256) {
+    *((threadgroup half4*)keys+i) = half4(*((device const float4*)(V+base+start*64)+i));
+}
+threadgroup_barrier(mem_flags::mem_threadgroup);''')
     if _HALF_PARTIAL:
         for n in range(0,_BK // 8,2):
             s.append(f'simdgroup_half8x8 vv{n};')
@@ -119,8 +126,10 @@
     source = source.replace('start<=block.x*64; start+=64', f'start<block.x*{_BQ}+{_BQ}; start+={_BK}')
     source = source.replace('start == block.x*64', f'start+{_BK} > block.x*{_BQ}')
     source = source.replace('i<1024; i+=256', f'i<{_BK*16}; i+={_BQ*4}')
-    source = source.replace('0.125f', '0.18033688011112042f').replace('fast::exp(', 'fast::exp2(')
-    source = source.replace('    for (uint i=tid;', '    #pragma unroll\n    for (uint i=tid;')
+    if _REUSE_KV:
+        source = source.replace(f'threadgroup half vals[{_BK*64}];','')
+        source = source.replace('        *((threadgroup half4*)vals+i) = half4(*((device const float4*)(V+base+start*64)+i));','')
+        source = source.replace('vals+', 'keys+')
     return source
 
 

```

</details>

`python bench.py`  
→ run 7 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 68.55 ms, kernel 3.89 ms, speedup 17.60x

> The first version still leads at 3.70 ms. Short half-precision score partials took 3.74 ms but had higher error; base-two exponentials and staging-loop unrolling took 3.89 ms. I’ll test one shared-memory variant, then recheck the fastest candidate.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.30e-04 (allowed 2e-03)
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -2,14 +2,8 @@
 import torch
 
 
-_HALF_PARTIAL = False
-_HALF_SCORE = False
-_REUSE_KV = True
-_BQ = 64
-_BK = 64
-
-
 def _source():
+    # Generate constant matrix operations so the compiler keeps tiles in registers.
     s = ['''#include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
@@ -39,39 +33,27 @@
         *((threadgroup half4*)vals+i) = half4(*((device const float4*)(V+base+start*64)+i));
     }
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    simdgroup_float8x8 score[KEY_MATRICES];
-    simdgroup_half8x8 p[KEY_MATRICES];
+    simdgroup_float8x8 score[8];
+    simdgroup_half8x8 p[8];
 ''')
-    for n in range(_BK // 8):
+    for n in range(8):
         s.append(f'score[{n}] = simdgroup_float8x8(0.0f);')
-    if _HALF_SCORE:
-        for d in range(0,8,2):
-            s.append(f'simdgroup_half8x8 kk{d};')
-            for n in range(_BK // 8):
-                s.append(f'''simdgroup_half8x8 ss{d}_{n}(half(0));
-simdgroup_load(kk{d}, keys+{n*8*64+d*8},64,ulong2(0),true);
-simdgroup_multiply_accumulate(ss{d}_{n},q[{d}],kk{d},ss{d}_{n});
-simdgroup_load(kk{d}, keys+{n*8*64+(d+1)*8},64,ulong2(0),true);
-simdgroup_multiply_accumulate(ss{d}_{n},q[{d+1}],kk{d},ss{d}_{n});
-score[{n}].thread_elements()[0] += float(ss{d}_{n}.thread_elements()[0]);
-score[{n}].thread_elements()[1] += float(ss{d}_{n}.thread_elements()[1]);''')
-    else:
-        for d in range(8):
-            s.append(f'simdgroup_half8x8 kk{d};')
-            for n in range(_BK // 8):
-                s.append(f'''simdgroup_load(kk{d}, keys+{n*8*64+d*8},64,ulong2(0),true);
+    for d in range(8):
+        s.append(f'simdgroup_half8x8 kk{d};')
+        for n in range(8):
+            s.append(f'''simdgroup_load(kk{d}, keys+{n*8*64+d*8},64,ulong2(0),true);
 simdgroup_multiply_accumulate(score[{n}], q[{d}], kk{d}, score[{n}]);''')
     s.append('float mx = -INFINITY;')
-    for n in range(_BK // 8):
+    for n in range(8):
         s.append(f'''score[{n}].thread_elements()[0] *= 0.125f;
 score[{n}].thread_elements()[1] *= 0.125f;''')
     s.append('if (start == block.x*64) {')
-    for n in range(_BK // 8):
+    for n in range(8):
         s.append(f'''
  if (start+{n*8}+c > qr+r) score[{n}].thread_elements()[0] = -INFINITY;
  if (start+{n*8}+c+1 > qr+r) score[{n}].thread_elements()[1] = -INFINITY;''')
     s.append('}')
-    for n in range(_BK // 8):
+    for n in range(8):
         s.append(f'mx=max(mx,max(score[{n}].thread_elements()[0],score[{n}].thread_elements()[1]));')
     s.append('''mx=max(mx,simd_shuffle_xor(mx,1));
 mx=max(mx,simd_shuffle_xor(mx,8));
@@ -79,7 +61,7 @@
 float alpha=fast::exp(maximum-nextmax);
 maximum=nextmax;
 float total=0.0f;''')
-    for n in range(_BK // 8):
+    for n in range(8):
         s.append(f'''float2 prob{n} = fast::exp(float2(score[{n}].thread_elements()[0],score[{n}].thread_elements()[1])-nextmax);
 p[{n}].thread_elements()[0]=half(prob{n}.x);
 p[{n}].thread_elements()[1]=half(prob{n}.y);
@@ -90,28 +72,10 @@
     s.append('''total+=simd_shuffle_xor(total,1);
 total+=simd_shuffle_xor(total,8);
 denominator=denominator*alpha+total;''')
-    if _REUSE_KV:
-        s.append('''threadgroup_barrier(mem_flags::mem_threadgroup);
-for (uint i=tid; i<1024; i+=256) {
-    *((threadgroup half4*)keys+i) = half4(*((device const float4*)(V+base+start*64)+i));
-}
-threadgroup_barrier(mem_flags::mem_threadgroup);''')
-    if _HALF_PARTIAL:
-        for n in range(0,_BK // 8,2):
-            s.append(f'simdgroup_half8x8 vv{n};')
-            for d in range(8):
-                s.append(f'''simdgroup_half8x8 tmp{n}_{d}(half(0));
-simdgroup_load(vv{n}, vals+{n*8*64+d*8},64);
-simdgroup_multiply_accumulate(tmp{n}_{d},p[{n}],vv{n},tmp{n}_{d});
-simdgroup_load(vv{n}, vals+{(n+1)*8*64+d*8},64);
-simdgroup_multiply_accumulate(tmp{n}_{d},p[{n+1}],vv{n},tmp{n}_{d});
-out[{d}].thread_elements()[0] += float(tmp{n}_{d}.thread_elements()[0]);
-out[{d}].thread_elements()[1] += float(tmp{n}_{d}.thread_elements()[1]);''')
-    else:
-        for n in range(_BK // 8):
-            s.append(f'simdgroup_half8x8 vv{n};')
-            for d in range(8):
-                s.append(f'''simdgroup_load(vv{n}, vals+{n*8*64+d*8},64);
+    for n in range(8):
+        s.append(f'simdgroup_half8x8 vv{n};')
+        for d in range(8):
+            s.append(f'''simdgroup_load(vv{n}, vals+{n*8*64+d*8},64);
 simdgroup_multiply_accumulate(out[{d}],p[{n}],vv{n},out[{d}]);''')
     s.append('threadgroup_barrier(mem_flags::mem_threadgroup);\n}')
     for d in range(8):
@@ -119,18 +83,7 @@
 out[{d}].thread_elements()[1] /= denominator;
 simdgroup_store(out[{d}],O+base+qr*64+{d*8},64);''')
     s.append('}')
-    source = '\n'.join(s)
-    source = source.replace('KEY_MATRICES',str(_BK//8))
-    source = source.replace('keys[4096]', f'keys[{_BK*64}]').replace('vals[4096]', f'vals[{_BK*64}]')
-    source = source.replace('block.x * 64', f'block.x * {_BQ}')
-    source = source.replace('start<=block.x*64; start+=64', f'start<block.x*{_BQ}+{_BQ}; start+={_BK}')
-    source = source.replace('start == block.x*64', f'start+{_BK} > block.x*{_BQ}')
-    source = source.replace('i<1024; i+=256', f'i<{_BK*16}; i+={_BQ*4}')
-    if _REUSE_KV:
-        source = source.replace(f'threadgroup half vals[{_BK*64}];','')
-        source = source.replace('        *((threadgroup half4*)vals+i) = half4(*((device const float4*)(V+base+start*64)+i));','')
-        source = source.replace('vals+', 'keys+')
-    return source
+    return '\n'.join(s)
 
 
 _lib = torch.mps.compile_shader(_source())
@@ -138,5 +91,5 @@
 
 def kernel(q, k, v):
     out = torch.empty_like(q)
-    _lib.attention(q, k, v, out, threads=(1024//_BQ*(_BQ*4),64,1), group_size=(_BQ*4,1,1))
+    _lib.attention(q, k, v, out, threads=(4096,64,1), group_size=(256,1,1))
     return out

```

</details>

`python bench.py`  
→     _lib.attention(q, k, v, out, threads=(1024//_BQ*(_BQ*4),64,1), group_size=(_BQ*4,1,1))  
→     return out

- edited `solution.py`
- `python check.py`  
→ OK: relative error 2.92e-04 (allowed 2e-03)
### Bench run 11

`python bench.py`  
→     _lib.attention(q, k, v, out, threads=(4096,64,1), group_size=(256,1,1))  
→     return out

> All ten benchmark runs passed. I’m keeping the fastest measured version at 3.70 ms (18.25× speedup). I’ll now check it on fresh inputs and update the notes with the results.

- `cp queue/run1.py kernel.py`
- `python -c 'import torch; from kernel import kernel; from reference import inputs, reference, TOL; errors=[]`
### Bench run 12

`python -c 'from pathlib import Path; notes=\"\"\"The workload is contiguous float32 causal attention on MPS: eight batches, eight heads, 1024 positions, and 64 channels. GPU contention varies reference times; compare absolute kernel times and treat small differences cautiously. Use only bench.py for timing.`  
→ LESSONS.md words: 279

- `python check.py`  
→ OK: relative error 3.45e-04 (allowed 2e-03)
> Kernel measured **3.70 ms**, an **18.25× speedup**. Twenty fresh-input checks and the final check.py passed. LESSONS.md was updated.
> 
> EDIT: Final kernel.py uses fused tiled Metal causal attention with register-held matrices and float32 accumulation.

