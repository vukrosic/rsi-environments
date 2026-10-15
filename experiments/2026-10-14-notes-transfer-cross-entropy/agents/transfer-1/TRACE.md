# Agent `transfer-1`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who optimized a different kernel (causal attention) on this same Mac. Read it before your first bench run.

11 min, 47,899 tokens. Unchanged code: 18.5 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 1.935 ms | 8.38x |
| run2 | 1.705 ms | 7.6x |
| run3 | 2.791 ms | 6.14x |
| run4 | 2.14 ms | 7.14x |
| run5 | 2.994 ms | 5.73x |
| run6 | 2.229 ms | 6.63x |
| run7 | 1.808 ms | 7.44x |
| run8 | 1.975 ms | 7.53x |
| run9 | 2.359 ms | 7.37x |
| run10 | 2.149 ms | 7.07x |
| final | 2.287 ms | 7.78x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the reference and local notes, then optimize the loss calculation and use the benchmark runs to compare implementations.

- `pwd && rg --files .`
### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -1,8 +1,72 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
+"""Fused, stable cross entropy for contiguous MPS logits."""
 import torch
+
+_shader = torch.mps.compile_shader(r"""
+#include <metal_stdlib>
+using namespace metal;
+
+kernel void row_loss(device const float4* logits,
+                     device const uint2* labels,
+                     device float* losses,
+                     uint row [[threadgroup_position_in_grid]],
+                     uint tid [[thread_index_in_threadgroup]],
+                     uint lane [[thread_index_in_simdgroup]],
+                     uint sg [[simdgroup_index_in_threadgroup]]) {
+    float4 v[4];
+    float mx = -INFINITY;
+    #pragma unroll
+    for (uint i = 0; i < 4; ++i) {
+        v[i] = logits[row * 1024 + tid + i * 256];
+        mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
+    }
+    threadgroup float scratch[8];
+    mx = simd_max(mx);
+    if (lane == 0) scratch[sg] = mx;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    mx = simd_max(lane < 8 ? scratch[lane] : -INFINITY);
+    float sum = 0.0f;
+    #pragma unroll
+    for (uint i = 0; i < 4; ++i) {
+        float4 e = exp(v[i] - mx);
+        sum += (e.x + e.y) + (e.z + e.w);
+    }
+    sum = simd_sum(sum);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if (lane == 0) scratch[sg] = sum;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    sum = simd_sum(lane < 8 ? scratch[lane] : 0.0f);
+    if (tid == 0) {
+        device const float* scalars = reinterpret_cast<device const float*>(logits);
+        losses[row] = mx + log(sum) - scalars[row * 4096 + labels[row].x];
+    }
+}
+
+kernel void mean_loss(device const float* losses,
+                      device float* out,
+                      constant uint& rows,
+                      uint tid [[thread_index_in_threadgroup]],
+                      uint lane [[thread_index_in_simdgroup]],
+                      uint sg [[simdgroup_index_in_threadgroup]]) {
+    float sum = 0.0f;
+    for (uint i = tid; i < rows; i += 256) sum += losses[i];
+    sum = simd_sum(sum);
+    threadgroup float scratch[8];
+    if (lane == 0) scratch[sg] = sum;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if (sg == 0) {
+        sum = simd_sum(lane < 8 ? scratch[lane] : 0.0f);
+        if (lane == 0) out[0] = sum / float(rows);
+    }
+}
+""")
 
 
 def kernel(logits, y):
-    m = logits.max(-1, keepdim=True).values
-    lse = (logits - m).exp().sum(-1).log() + m.squeeze(-1)
-    return (lse - logits.gather(-1, y[:, None]).squeeze(-1)).mean()
+    if logits.shape[1] != 4096 or not logits.is_contiguous() or not y.is_contiguous():
+        return (torch.logsumexp(logits, -1) - logits.gather(-1, y[:, None]).squeeze(-1)).mean()
+    rows = logits.shape[0]
+    losses = torch.empty((rows,), device=logits.device, dtype=torch.float32)
+    out = torch.empty((), device=logits.device, dtype=torch.float32)
+    _shader.row_loss(logits, y, losses, threads=rows * 256, group_size=256)
+    _shader.mean_loss(losses, out, rows, threads=256, group_size=256)
+    return out

```

</details>

`cat kernel.py reference.py bench.py check.py LESSONS.md`  
→ Explicitly expand fixed matrix loops before compilation. Compiler unroll hints alone measured 6.43 milliseconds; literal expansion measured 3.74. Other measurements: half sixteen-key partial sums with hints, 7.24; direct query loads, 3.87; expanded half partial sums, 3.97; 128-query tiles, 4.12; 32-key tiles, 3.90; separate diagonal-mask branch, 3.94; fast exponential intrinsic, 3.97.  
→ Earlier rounds measured built-in attention around seventeen milliseconds in float32 and 10.5 to 11.7 with half inputs. Fused register implementations previously reached 4.10 and 3.88 milliseconds. Shared score buffers, insufficient unrolling, and separate input conversion often regressed. Half dot-product accumulation and long half output accumulation can exceed tolerance.

- `python -c 'import torch; print(torch.__version__); print(torch.mps.compile_shader.__doc__)`
- `python -c 'import torch; lib = torch.mps.compile_shader(\"kernel void full(device float* out, uint idx [[thread_position_in_grid]]) { out[idx] = 1; }\"); pri...`
- `python -c 'import torch; lib = torch.mps.compile_shader(\"kernel void full(device float* out, uint idx [[thread_position_in_grid]]) { out[idx] = 1; }\"); x=t...`
> The reference creates several large intermediate tensors. I’ll try a fused Metal shader that reads each logit once, computes the loss for each row, and uses a second small shader to calculate the mean.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 0.00e+00 (allowed 2e-03)
### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -12,21 +12,21 @@
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
-    float4 v[4];
+    float4 v[8];
     float mx = -INFINITY;
     #pragma unroll
-    for (uint i = 0; i < 4; ++i) {
-        v[i] = logits[row * 1024 + tid + i * 256];
+    for (uint i = 0; i < 8; ++i) {
+        v[i] = logits[row * 1024 + tid + i * 128];
         mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
     }
-    threadgroup float scratch[8];
+    threadgroup float scratch[4];
     mx = simd_max(mx);
     if (lane == 0) scratch[sg] = mx;
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    mx = simd_max(lane < 8 ? scratch[lane] : -INFINITY);
+    mx = simd_max(lane < 4 ? scratch[lane] : -INFINITY);
     float sum = 0.0f;
     #pragma unroll
-    for (uint i = 0; i < 4; ++i) {
+    for (uint i = 0; i < 8; ++i) {
         float4 e = exp(v[i] - mx);
         sum += (e.x + e.y) + (e.z + e.w);
     }
@@ -34,7 +34,7 @@
     threadgroup_barrier(mem_flags::mem_threadgroup);
     if (lane == 0) scratch[sg] = sum;
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    sum = simd_sum(lane < 8 ? scratch[lane] : 0.0f);
+    sum = simd_sum(lane < 4 ? scratch[lane] : 0.0f);
     if (tid == 0) {
         device const float* scalars = reinterpret_cast<device const float*>(logits);
         losses[row] = mx + log(sum) - scalars[row * 4096 + labels[row].x];
@@ -67,6 +67,6 @@
     rows = logits.shape[0]
     losses = torch.empty((rows,), device=logits.device, dtype=torch.float32)
     out = torch.empty((), device=logits.device, dtype=torch.float32)
-    _shader.row_loss(logits, y, losses, threads=rows * 256, group_size=256)
+    _shader.row_loss(logits, y, losses, threads=rows * 128, group_size=128)
     _shader.mean_loss(losses, out, rows, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=1200, check=True)`  
→ run 1 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 16.21 ms, kernel 1.94 ms, speedup 8.38x

> The fused version passes the check and runs in 1.94 ms, an 8.38× speedup in the first benchmark. I’ll now compare threadgroup sizes to see whether they improve GPU throughput.

- `cat kernel.py`
- `python - <<'PY`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 0.00e+00 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -8,34 +8,27 @@
 kernel void row_loss(device const float4* logits,
                      device const uint2* labels,
                      device float* losses,
-                     uint row [[threadgroup_position_in_grid]],
+                     uint group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
-    float4 v[8];
+    uint row = group * 8 + sg;
+    float4 v[32];
     float mx = -INFINITY;
     #pragma unroll
-    for (uint i = 0; i < 8; ++i) {
-        v[i] = logits[row * 1024 + tid + i * 128];
+    for (uint i = 0; i < 32; ++i) {
+        v[i] = logits[row * 1024 + lane + i * 32];
         mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
     }
-    threadgroup float scratch[4];
     mx = simd_max(mx);
-    if (lane == 0) scratch[sg] = mx;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    mx = simd_max(lane < 4 ? scratch[lane] : -INFINITY);
     float sum = 0.0f;
     #pragma unroll
-    for (uint i = 0; i < 8; ++i) {
+    for (uint i = 0; i < 32; ++i) {
         float4 e = exp(v[i] - mx);
         sum += (e.x + e.y) + (e.z + e.w);
     }
     sum = simd_sum(sum);
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    if (lane == 0) scratch[sg] = sum;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    sum = simd_sum(lane < 4 ? scratch[lane] : 0.0f);
-    if (tid == 0) {
+    if (lane == 0) {
         device const float* scalars = reinterpret_cast<device const float*>(logits);
         losses[row] = mx + log(sum) - scalars[row * 4096 + labels[row].x];
     }
@@ -67,6 +60,6 @@
     rows = logits.shape[0]
     losses = torch.empty((rows,), device=logits.device, dtype=torch.float32)
     out = torch.empty((), device=logits.device, dtype=torch.float32)
-    _shader.row_loss(logits, y, losses, threads=rows * 128, group_size=128)
+    _shader.row_loss(logits, y, losses, threads=rows * 32, group_size=256)
     _shader.mean_loss(losses, out, rows, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=1200, check=True)`  
→ run 2 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 12.95 ms, kernel 1.70 ms, speedup 7.60x

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 7.54e-08 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -8,27 +8,34 @@
 kernel void row_loss(device const float4* logits,
                      device const uint2* labels,
                      device float* losses,
-                     uint group [[threadgroup_position_in_grid]],
+                     uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
-    uint row = group * 8 + sg;
-    float4 v[32];
+    float4 v[16];
     float mx = -INFINITY;
     #pragma unroll
-    for (uint i = 0; i < 32; ++i) {
-        v[i] = logits[row * 1024 + lane + i * 32];
+    for (uint i = 0; i < 16; ++i) {
+        v[i] = logits[row * 1024 + tid + i * 64];
         mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
     }
+    threadgroup float scratch[2];
     mx = simd_max(mx);
+    if (lane == 0) scratch[sg] = mx;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    mx = simd_max(lane < 2 ? scratch[lane] : -INFINITY);
     float sum = 0.0f;
     #pragma unroll
-    for (uint i = 0; i < 32; ++i) {
+    for (uint i = 0; i < 16; ++i) {
         float4 e = exp(v[i] - mx);
         sum += (e.x + e.y) + (e.z + e.w);
     }
     sum = simd_sum(sum);
-    if (lane == 0) {
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if (lane == 0) scratch[sg] = sum;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    sum = simd_sum(lane < 2 ? scratch[lane] : 0.0f);
+    if (tid == 0) {
         device const float* scalars = reinterpret_cast<device const float*>(logits);
         losses[row] = mx + log(sum) - scalars[row * 4096 + labels[row].x];
     }
@@ -60,6 +67,6 @@
     rows = logits.shape[0]
     losses = torch.empty((rows,), device=logits.device, dtype=torch.float32)
     out = torch.empty((), device=logits.device, dtype=torch.float32)
-    _shader.row_loss(logits, y, losses, threads=rows * 32, group_size=256)
+    _shader.row_loss(logits, y, losses, threads=rows * 64, group_size=64)
     _shader.mean_loss(losses, out, rows, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=1200, check=True)`  
→ run 3 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 17.13 ms, kernel 2.79 ms, speedup 6.14x

- `cat kernel.py`
> Using 128 threads per row improved the time to 1.70 ms. The 32-thread version took 2.79 ms; its larger register workload may be limiting throughput. I’ll test 64 threads next, then try a different reduction strategy.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 0.00e+00 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -12,30 +12,31 @@
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
-    float4 v[16];
+    float4 v[8];
     float mx = -INFINITY;
     #pragma unroll
-    for (uint i = 0; i < 16; ++i) {
-        v[i] = logits[row * 1024 + tid + i * 64];
+    for (uint i = 0; i < 8; ++i) {
+        v[i] = logits[row * 1024 + tid + i * 128];
         mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
     }
-    threadgroup float scratch[2];
     mx = simd_max(mx);
-    if (lane == 0) scratch[sg] = mx;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    mx = simd_max(lane < 2 ? scratch[lane] : -INFINITY);
     float sum = 0.0f;
     #pragma unroll
-    for (uint i = 0; i < 16; ++i) {
+    for (uint i = 0; i < 8; ++i) {
         float4 e = exp(v[i] - mx);
         sum += (e.x + e.y) + (e.z + e.w);
     }
     sum = simd_sum(sum);
+    threadgroup float2 scratch[4];
+    if (lane == 0) scratch[sg] = float2(mx, sum);
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    if (lane == 0) scratch[sg] = sum;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    sum = simd_sum(lane < 2 ? scratch[lane] : 0.0f);
     if (tid == 0) {
+        float row_max = max(max(scratch[0].x, scratch[1].x), max(scratch[2].x, scratch[3].x));
+        float row_sum = 0.0f;
+        #pragma unroll
+        for (uint i = 0; i < 4; ++i) row_sum += scratch[i].y * exp(scratch[i].x - row_max);
+        mx = row_max;
+        sum = row_sum;
         device const float* scalars = reinterpret_cast<device const float*>(logits);
         losses[row] = mx + log(sum) - scalars[row * 4096 + labels[row].x];
     }
@@ -67,6 +68,6 @@
     rows = logits.shape[0]
     losses = torch.empty((rows,), device=logits.device, dtype=torch.float32)
     out = torch.empty((), device=logits.device, dtype=torch.float32)
-    _shader.row_loss(logits, y, losses, threads=rows * 64, group_size=64)
+    _shader.row_loss(logits, y, losses, threads=rows * 128, group_size=128)
     _shader.mean_loss(losses, out, rows, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=1200, check=True)`  
→ run 4 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 15.28 ms, kernel 2.14 ms, speedup 7.14x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 0.00e+00 (allowed 2e-03)
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -12,31 +12,51 @@
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
-    float4 v[8];
     float mx = -INFINITY;
-    #pragma unroll
-    for (uint i = 0; i < 8; ++i) {
-        v[i] = logits[row * 1024 + tid + i * 128];
-        mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
-    }
+    float4 v0 = logits[row * 1024 + tid + 0];
+    mx = max(mx, max(max(v0.x, v0.y), max(v0.z, v0.w)));
+    float4 v1 = logits[row * 1024 + tid + 128];
+    mx = max(mx, max(max(v1.x, v1.y), max(v1.z, v1.w)));
+    float4 v2 = logits[row * 1024 + tid + 256];
+    mx = max(mx, max(max(v2.x, v2.y), max(v2.z, v2.w)));
+    float4 v3 = logits[row * 1024 + tid + 384];
+    mx = max(mx, max(max(v3.x, v3.y), max(v3.z, v3.w)));
+    float4 v4 = logits[row * 1024 + tid + 512];
+    mx = max(mx, max(max(v4.x, v4.y), max(v4.z, v4.w)));
+    float4 v5 = logits[row * 1024 + tid + 640];
+    mx = max(mx, max(max(v5.x, v5.y), max(v5.z, v5.w)));
+    float4 v6 = logits[row * 1024 + tid + 768];
+    mx = max(mx, max(max(v6.x, v6.y), max(v6.z, v6.w)));
+    float4 v7 = logits[row * 1024 + tid + 896];
+    mx = max(mx, max(max(v7.x, v7.y), max(v7.z, v7.w)));
+    threadgroup float scratch[4];
     mx = simd_max(mx);
+    if (lane == 0) scratch[sg] = mx;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    mx = simd_max(lane < 4 ? scratch[lane] : -INFINITY);
     float sum = 0.0f;
-    #pragma unroll
-    for (uint i = 0; i < 8; ++i) {
-        float4 e = exp(v[i] - mx);
-        sum += (e.x + e.y) + (e.z + e.w);
-    }
+    float4 e0 = exp(v0 - mx);
+    sum += (e0.x + e0.y) + (e0.z + e0.w);
+    float4 e1 = exp(v1 - mx);
+    sum += (e1.x + e1.y) + (e1.z + e1.w);
+    float4 e2 = exp(v2 - mx);
+    sum += (e2.x + e2.y) + (e2.z + e2.w);
+    float4 e3 = exp(v3 - mx);
+    sum += (e3.x + e3.y) + (e3.z + e3.w);
+    float4 e4 = exp(v4 - mx);
+    sum += (e4.x + e4.y) + (e4.z + e4.w);
+    float4 e5 = exp(v5 - mx);
+    sum += (e5.x + e5.y) + (e5.z + e5.w);
+    float4 e6 = exp(v6 - mx);
+    sum += (e6.x + e6.y) + (e6.z + e6.w);
+    float4 e7 = exp(v7 - mx);
+    sum += (e7.x + e7.y) + (e7.z + e7.w);
     sum = simd_sum(sum);
-    threadgroup float2 scratch[4];
-    if (lane == 0) scratch[sg] = float2(mx, sum);
     threadgroup_barrier(mem_flags::mem_threadgroup);
+    if (lane == 0) scratch[sg] = sum;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    sum = simd_sum(lane < 4 ? scratch[lane] : 0.0f);
     if (tid == 0) {
-        float row_max = max(max(scratch[0].x, scratch[1].x), max(scratch[2].x, scratch[3].x));
-        float row_sum = 0.0f;
-        #pragma unroll
-        for (uint i = 0; i < 4; ++i) row_sum += scratch[i].y * exp(scratch[i].x - row_max);
-        mx = row_max;
-        sum = row_sum;
         device const float* scalars = reinterpret_cast<device const float*>(logits);
         losses[row] = mx + log(sum) - scalars[row * 4096 + labels[row].x];
     }

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=1200, check=True)`  
→ run 5 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 17.15 ms, kernel 2.99 ms, speedup 5.73x

> The 64-thread version and the alternative reduction were both slower, so I’ll return to the 128-thread design. Next I’ll expand the fixed loops explicitly and compare exponentiation options.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 0.00e+00 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -12,45 +12,24 @@
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
+    float4 v[8];
     float mx = -INFINITY;
-    float4 v0 = logits[row * 1024 + tid + 0];
-    mx = max(mx, max(max(v0.x, v0.y), max(v0.z, v0.w)));
-    float4 v1 = logits[row * 1024 + tid + 128];
-    mx = max(mx, max(max(v1.x, v1.y), max(v1.z, v1.w)));
-    float4 v2 = logits[row * 1024 + tid + 256];
-    mx = max(mx, max(max(v2.x, v2.y), max(v2.z, v2.w)));
-    float4 v3 = logits[row * 1024 + tid + 384];
-    mx = max(mx, max(max(v3.x, v3.y), max(v3.z, v3.w)));
-    float4 v4 = logits[row * 1024 + tid + 512];
-    mx = max(mx, max(max(v4.x, v4.y), max(v4.z, v4.w)));
-    float4 v5 = logits[row * 1024 + tid + 640];
-    mx = max(mx, max(max(v5.x, v5.y), max(v5.z, v5.w)));
-    float4 v6 = logits[row * 1024 + tid + 768];
-    mx = max(mx, max(max(v6.x, v6.y), max(v6.z, v6.w)));
-    float4 v7 = logits[row * 1024 + tid + 896];
-    mx = max(mx, max(max(v7.x, v7.y), max(v7.z, v7.w)));
+    #pragma unroll
+    for (uint i = 0; i < 8; ++i) {
+        v[i] = logits[row * 1024 + tid + i * 128];
+        mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
+    }
     threadgroup float scratch[4];
     mx = simd_max(mx);
     if (lane == 0) scratch[sg] = mx;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     mx = simd_max(lane < 4 ? scratch[lane] : -INFINITY);
     float sum = 0.0f;
-    float4 e0 = exp(v0 - mx);
-    sum += (e0.x + e0.y) + (e0.z + e0.w);
-    float4 e1 = exp(v1 - mx);
-    sum += (e1.x + e1.y) + (e1.z + e1.w);
-    float4 e2 = exp(v2 - mx);
-    sum += (e2.x + e2.y) + (e2.z + e2.w);
-    float4 e3 = exp(v3 - mx);
-    sum += (e3.x + e3.y) + (e3.z + e3.w);
-    float4 e4 = exp(v4 - mx);
-    sum += (e4.x + e4.y) + (e4.z + e4.w);
-    float4 e5 = exp(v5 - mx);
-    sum += (e5.x + e5.y) + (e5.z + e5.w);
-    float4 e6 = exp(v6 - mx);
-    sum += (e6.x + e6.y) + (e6.z + e6.w);
-    float4 e7 = exp(v7 - mx);
-    sum += (e7.x + e7.y) + (e7.z + e7.w);
+    #pragma unroll
+    for (uint i = 0; i < 8; ++i) {
+        float4 e = fast::exp2((v[i] - mx) * 1.4426950408889634f);
+        sum += (e.x + e.y) + (e.z + e.w);
+    }
     sum = simd_sum(sum);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     if (lane == 0) scratch[sg] = sum;
@@ -58,7 +37,7 @@
     sum = simd_sum(lane < 4 ? scratch[lane] : 0.0f);
     if (tid == 0) {
         device const float* scalars = reinterpret_cast<device const float*>(logits);
-        losses[row] = mx + log(sum) - scalars[row * 4096 + labels[row].x];
+        losses[row] = mx + fast::log2(sum) * 0.6931471805599453f - scalars[row * 4096 + labels[row].x];
     }
 }
 

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=1200, check=True)`  
→ run 6 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 14.77 ms, kernel 2.23 ms, speedup 6.63x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 7.47e-08 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -19,25 +19,37 @@
         v[i] = logits[row * 1024 + tid + i * 128];
         mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
     }
-    threadgroup float scratch[4];
-    mx = simd_max(mx);
-    if (lane == 0) scratch[sg] = mx;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    mx = simd_max(lane < 4 ? scratch[lane] : -INFINITY);
     float sum = 0.0f;
     #pragma unroll
     for (uint i = 0; i < 8; ++i) {
-        float4 e = fast::exp2((v[i] - mx) * 1.4426950408889634f);
+        float4 e = exp(v[i]);
         sum += (e.x + e.y) + (e.z + e.w);
     }
+    mx = simd_max(mx);
     sum = simd_sum(sum);
+    threadgroup float2 scratch[4];
+    if (lane == 0) scratch[sg] = float2(mx, sum);
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    if (lane == 0) scratch[sg] = sum;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    sum = simd_sum(lane < 4 ? scratch[lane] : 0.0f);
+    mx = simd_max(lane < 4 ? scratch[lane].x : -INFINITY);
+    sum = simd_sum(lane < 4 ? scratch[lane].y : 0.0f);
+    float shift = 0.0f;
+    if (mx < -60.0f || mx > 60.0f) {
+        shift = mx;
+        sum = 0.0f;
+        #pragma unroll
+        for (uint i = 0; i < 8; ++i) {
+            float4 e = exp(v[i] - mx);
+            sum += (e.x + e.y) + (e.z + e.w);
+        }
+        sum = simd_sum(sum);
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        if (lane == 0) scratch[sg].y = sum;
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        sum = simd_sum(lane < 4 ? scratch[lane].y : 0.0f);
+    }
     if (tid == 0) {
         device const float* scalars = reinterpret_cast<device const float*>(logits);
-        losses[row] = mx + fast::log2(sum) * 0.6931471805599453f - scalars[row * 4096 + labels[row].x];
+        losses[row] = shift + log(sum) - scalars[row * 4096 + labels[row].x];
     }
 }
 

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=1200, check=True)`  
→ run 7 of 10 submitted.  
→ cross_entropy: correct (relative error 7.5e-08). reference 13.46 ms, kernel 1.81 ms, speedup 7.44x

> Explicit loop expansion and fast base-two exponentials did not beat 1.70 ms. I’ll try one final approach: sum exponentials directly for moderate logits, with a stable shifted calculation for extreme values. That may reduce synchronization on the timed inputs.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 0.00e+00 (allowed 2e-03)
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -8,10 +8,13 @@
 kernel void row_loss(device const float4* logits,
                      device const uint2* labels,
                      device float* losses,
-                     uint row [[threadgroup_position_in_grid]],
-                     uint tid [[thread_index_in_threadgroup]],
+                     uint group [[threadgroup_position_in_grid]],
+                     uint thread_id [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
+    uint row = group * 2 + sg / 4;
+    uint tid = thread_id % 128;
+    uint row_slot = (sg / 4) * 4;
     float4 v[8];
     float mx = -INFINITY;
     #pragma unroll
@@ -19,37 +22,25 @@
         v[i] = logits[row * 1024 + tid + i * 128];
         mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
     }
+    threadgroup float scratch[8];
+    mx = simd_max(mx);
+    if (lane == 0) scratch[sg] = mx;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    mx = simd_max(lane < 4 ? scratch[row_slot + lane] : -INFINITY);
     float sum = 0.0f;
     #pragma unroll
     for (uint i = 0; i < 8; ++i) {
-        float4 e = exp(v[i]);
+        float4 e = exp(v[i] - mx);
         sum += (e.x + e.y) + (e.z + e.w);
     }
-    mx = simd_max(mx);
     sum = simd_sum(sum);
-    threadgroup float2 scratch[4];
-    if (lane == 0) scratch[sg] = float2(mx, sum);
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    mx = simd_max(lane < 4 ? scratch[lane].x : -INFINITY);
-    sum = simd_sum(lane < 4 ? scratch[lane].y : 0.0f);
-    float shift = 0.0f;
-    if (mx < -60.0f || mx > 60.0f) {
-        shift = mx;
-        sum = 0.0f;
-        #pragma unroll
-        for (uint i = 0; i < 8; ++i) {
-            float4 e = exp(v[i] - mx);
-            sum += (e.x + e.y) + (e.z + e.w);
-        }
-        sum = simd_sum(sum);
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        if (lane == 0) scratch[sg].y = sum;
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        sum = simd_sum(lane < 4 ? scratch[lane].y : 0.0f);
-    }
+    if (lane == 0) scratch[sg] = sum;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    sum = simd_sum(lane < 4 ? scratch[row_slot + lane] : 0.0f);
     if (tid == 0) {
         device const float* scalars = reinterpret_cast<device const float*>(logits);
-        losses[row] = shift + log(sum) - scalars[row * 4096 + labels[row].x];
+        losses[row] = mx + log(sum) - scalars[row * 4096 + labels[row].x];
     }
 }
 
@@ -74,11 +65,11 @@
 
 
 def kernel(logits, y):
-    if logits.shape[1] != 4096 or not logits.is_contiguous() or not y.is_contiguous():
+    if logits.shape[0] % 2 or logits.shape[1] != 4096 or not logits.is_contiguous() or not y.is_contiguous():
         return (torch.logsumexp(logits, -1) - logits.gather(-1, y[:, None]).squeeze(-1)).mean()
     rows = logits.shape[0]
     losses = torch.empty((rows,), device=logits.device, dtype=torch.float32)
     out = torch.empty((), device=logits.device, dtype=torch.float32)
-    _shader.row_loss(logits, y, losses, threads=rows * 128, group_size=128)
+    _shader.row_loss(logits, y, losses, threads=rows * 128, group_size=256)
     _shader.mean_loss(losses, out, rows, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=1200, check=True)`  
→ run 8 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 14.87 ms, kernel 1.97 ms, speedup 7.53x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 0.00e+00 (allowed 2e-03)
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -6,15 +6,12 @@
 using namespace metal;
 
 kernel void row_loss(device const float4* logits,
-                     device const uint2* labels,
+                     device const uint2* labels, // int64 labels; the low word is the class index
                      device float* losses,
-                     uint group [[threadgroup_position_in_grid]],
-                     uint thread_id [[thread_index_in_threadgroup]],
+                     uint row [[threadgroup_position_in_grid]],
+                     uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
-    uint row = group * 2 + sg / 4;
-    uint tid = thread_id % 128;
-    uint row_slot = (sg / 4) * 4;
     float4 v[8];
     float mx = -INFINITY;
     #pragma unroll
@@ -22,11 +19,11 @@
         v[i] = logits[row * 1024 + tid + i * 128];
         mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
     }
-    threadgroup float scratch[8];
+    threadgroup float scratch[4];
     mx = simd_max(mx);
     if (lane == 0) scratch[sg] = mx;
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    mx = simd_max(lane < 4 ? scratch[row_slot + lane] : -INFINITY);
+    mx = simd_max(lane < 4 ? scratch[lane] : -INFINITY);
     float sum = 0.0f;
     #pragma unroll
     for (uint i = 0; i < 8; ++i) {
@@ -37,7 +34,7 @@
     threadgroup_barrier(mem_flags::mem_threadgroup);
     if (lane == 0) scratch[sg] = sum;
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    sum = simd_sum(lane < 4 ? scratch[row_slot + lane] : 0.0f);
+    sum = simd_sum(lane < 4 ? scratch[lane] : 0.0f);
     if (tid == 0) {
         device const float* scalars = reinterpret_cast<device const float*>(logits);
         losses[row] = mx + log(sum) - scalars[row * 4096 + labels[row].x];
@@ -65,11 +62,13 @@
 
 
 def kernel(logits, y):
-    if logits.shape[0] % 2 or logits.shape[1] != 4096 or not logits.is_contiguous() or not y.is_contiguous():
+    if (logits.shape[1] != 4096 or logits.dtype != torch.float32
+            or y.dtype != torch.int64 or logits.device.type != "mps"
+            or not logits.is_contiguous() or not y.is_contiguous()):
         return (torch.logsumexp(logits, -1) - logits.gather(-1, y[:, None]).squeeze(-1)).mean()
     rows = logits.shape[0]
     losses = torch.empty((rows,), device=logits.device, dtype=torch.float32)
     out = torch.empty((), device=logits.device, dtype=torch.float32)
-    _shader.row_loss(logits, y, losses, threads=rows * 128, group_size=256)
+    _shader.row_loss(logits, y, losses, threads=rows * 128, group_size=128)
     _shader.mean_loss(losses, out, rows, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=1200, check=True)`  
→ run 9 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 17.39 ms, kernel 2.36 ms, speedup 7.37x

> The original stable 128-thread version still has the best measured time. I’ll restore it, use the last benchmark run to verify it, and finish with `check.py`.

- edited `solution.py`
### Bench run 11

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=1200, check=True)`  
→ run 10 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 15.20 ms, kernel 2.15 ms, speedup 7.07x

- `python check.py`  
→ OK: relative error 0.00e+00 (allowed 2e-03)
> Final benchmark: **2.15 ms, 7.07× speedup**. Final `check.py` passed with zero measured error.
> 
> EDIT: kernel.py computes stable cross-entropy with a fused Metal shader using 128 threads per row, followed by a GPU mean reduction.

