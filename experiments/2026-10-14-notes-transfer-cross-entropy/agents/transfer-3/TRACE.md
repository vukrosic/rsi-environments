# Agent `transfer-3`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who optimized a different kernel (causal attention) on this same Mac. Read it before your first bench run.

9 min, 63,411 tokens. Unchanged code: 18.5 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 1.814 ms | 7.71x |
| run2 | 1.667 ms | 7.82x |
| run3 | 2.466 ms | 5.93x |
| run4 | 2.2 ms | 7.04x |
| run5 | wrong: crashed |  |
| run6 | 1.823 ms | 7.31x |
| run7 | 2.531 ms | 6.78x |
| run8 | 2.746 ms | 5.83x |
| run9 | 1.691 ms | 7.36x |
| run10 | 1.993 ms | 6.88x |
| final | 1.755 ms | 7.67x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the task and the local optimization notes, then tune a Metal implementation with the benchmark runs available.

- `pwd && rg --files -g '"'!__pycache__'"' -g '"'!*.pyc`
### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -1,8 +1,67 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
 import torch
 
+_source = r'''
+#include <metal_stdlib>
+using namespace metal;
+kernel void rows(device const float4* x [[buffer(0)]],
+                 device const long* y [[buffer(1)]],
+                 device float* out [[buffer(2)]],
+                 uint row [[threadgroup_position_in_grid]],
+                 uint tid [[thread_index_in_threadgroup]],
+                 uint lane [[thread_index_in_simdgroup]],
+                 uint sg [[simdgroup_index_in_threadgroup]]) {
+    float4 v[4];
+    float m = -INFINITY;
+    #pragma unroll
+    for (uint i=0; i<4; ++i) {
+        v[i] = x[row*1024 + tid + i*256];
+        m = max(m, max(max(v[i].x,v[i].y),max(v[i].z,v[i].w)));
+    }
+    threadgroup float shared[8];
+    m = simd_max(m);
+    if (lane==0) shared[sg]=m;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    m=-INFINITY;
+    for(uint i=0;i<8;++i) m=max(m,shared[i]);
+    float s=0;
+    #pragma unroll
+    for (uint i=0;i<4;++i) {
+        float4 e=fast::exp(v[i]-m);
+        s+=e.x+e.y+e.z+e.w;
+    }
+    s=simd_sum(s);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if(lane==0) shared[sg]=s;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if(tid==0) {
+        s=0;
+        for(uint i=0;i<8;++i) s+=shared[i];
+        out[row]=log(s)+m-((device const float*)x)[row*4096+y[row]];
+    }
+}
+kernel void finish(device const float* rows [[buffer(0)]],
+                   device float* out [[buffer(1)]],
+                   uint tid [[thread_index_in_threadgroup]],
+                   uint lane [[thread_index_in_simdgroup]],
+                   uint sg [[simdgroup_index_in_threadgroup]]) {
+    float s=0;
+    for(uint i=tid;i<8192;i+=256) s+=rows[i];
+    s=simd_sum(s);
+    threadgroup float shared[8];
+    if(lane==0) shared[sg]=s;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if(tid==0) {
+        s=0;
+        for(uint i=0;i<8;++i) s+=shared[i];
+        out[0]=s/8192.0f;
+    }
+}
+'''
+_lib = torch.mps.compile_shader(_source)
 
 def kernel(logits, y):
-    m = logits.max(-1, keepdim=True).values
-    lse = (logits - m).exp().sum(-1).log() + m.squeeze(-1)
-    return (lse - logits.gather(-1, y[:, None]).squeeze(-1)).mean()
+    losses = torch.empty((8192,), device=logits.device)
+    out = torch.empty((), device=logits.device)
+    _lib.rows(logits, y, losses, threads=8192*256, group_size=256)
+    _lib.finish(losses, out, threads=256, group_size=256)
+    return out

```

</details>

`cat kernel.py reference.py bench.py check.py LESSONS.md`  
→ Explicitly expand fixed matrix loops before compilation. Compiler unroll hints alone measured 6.43 milliseconds; literal expansion measured 3.74. Other measurements: half sixteen-key partial sums with hints, 7.24; direct query loads, 3.87; expanded half partial sums, 3.97; 128-query tiles, 4.12; 32-key tiles, 3.90; separate diagonal-mask branch, 3.94; fast exponential intrinsic, 3.97.  
→ Earlier rounds measured built-in attention around seventeen milliseconds in float32 and 10.5 to 11.7 with half inputs. Fused register implementations previously reached 4.10 and 3.88 milliseconds. Shared score buffers, insufficient unrolling, and separate input conversion often regressed. Half dot-product accumulation and long half output accumulation can exceed tolerance.

- `python - <<'PY` (exited 1 in 0ms:)
- `python -c 'import torch; print(torch.__version__); print(torch.mps.compile_shader.__doc__)`
- `cat > kernel.py <<'PY` (exited 1 in 1127ms:)
- edited `solution.py`
- `python check.py`  
→ OK: relative error 7.47e-08 (allowed 2e-03)
### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -10,22 +10,21 @@
                  uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
-    float4 v[4];
+    float4 v[8];
     float m = -INFINITY;
     #pragma unroll
-    for (uint i=0; i<4; ++i) {
-        v[i] = x[row*1024 + tid + i*256];
+    for (uint i=0; i<8; ++i) {
+        v[i] = x[row*1024 + tid + i*128];
         m = max(m, max(max(v[i].x,v[i].y),max(v[i].z,v[i].w)));
     }
-    threadgroup float shared[8];
+    threadgroup float shared[4];
     m = simd_max(m);
     if (lane==0) shared[sg]=m;
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    m=-INFINITY;
-    for(uint i=0;i<8;++i) m=max(m,shared[i]);
+    m=simd_max(lane<4 ? shared[lane] : -INFINITY);
     float s=0;
     #pragma unroll
-    for (uint i=0;i<4;++i) {
+    for (uint i=0;i<8;++i) {
         float4 e=fast::exp(v[i]-m);
         s+=e.x+e.y+e.z+e.w;
     }
@@ -35,7 +34,7 @@
     threadgroup_barrier(mem_flags::mem_threadgroup);
     if(tid==0) {
         s=0;
-        for(uint i=0;i<8;++i) s+=shared[i];
+        for(uint i=0;i<4;++i) s+=shared[i];
         out[row]=log(s)+m-((device const float*)x)[row*4096+y[row]];
     }
 }
@@ -62,6 +61,6 @@
 def kernel(logits, y):
     losses = torch.empty((8192,), device=logits.device)
     out = torch.empty((), device=logits.device)
-    _lib.rows(logits, y, losses, threads=8192*256, group_size=256)
+    _lib.rows(logits, y, losses, threads=8192*128, group_size=128)
     _lib.finish(losses, out, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=600, check=True)`  
→ run 1 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 13.98 ms, kernel 1.81 ms, speedup 7.71x

> The first fused Metal version passes the check and runs in 1.81 ms, a 7.71× speedup in that benchmark. I’m now tuning the row reductions and memory access.

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text(); s=s.replace(\"float4 v[4]\", \"float4 v[8]\").replace(\"i<4\", \"i<8\").replace(...`
- `python check.py`  
→ OK: relative error 7.51e-08 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -6,37 +6,26 @@
 kernel void rows(device const float4* x [[buffer(0)]],
                  device const long* y [[buffer(1)]],
                  device float* out [[buffer(2)]],
-                 uint row [[threadgroup_position_in_grid]],
-                 uint tid [[thread_index_in_threadgroup]],
+                 uint group [[threadgroup_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
-    float4 v[8];
-    float m = -INFINITY;
+    uint row=group*8+sg;
+    float4 v[32];
+    float m=-INFINITY;
     #pragma unroll
-    for (uint i=0; i<8; ++i) {
-        v[i] = x[row*1024 + tid + i*128];
-        m = max(m, max(max(v[i].x,v[i].y),max(v[i].z,v[i].w)));
+    for(uint i=0;i<32;++i) {
+        v[i]=x[row*1024+lane+i*32];
+        m=max(m,max(max(v[i].x,v[i].y),max(v[i].z,v[i].w)));
     }
-    threadgroup float shared[4];
-    m = simd_max(m);
-    if (lane==0) shared[sg]=m;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    m=simd_max(lane<4 ? shared[lane] : -INFINITY);
+    m=simd_max(m);
     float s=0;
     #pragma unroll
-    for (uint i=0;i<8;++i) {
+    for(uint i=0;i<32;++i) {
         float4 e=fast::exp(v[i]-m);
         s+=e.x+e.y+e.z+e.w;
     }
     s=simd_sum(s);
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    if(lane==0) shared[sg]=s;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    if(tid==0) {
-        s=0;
-        for(uint i=0;i<4;++i) s+=shared[i];
-        out[row]=log(s)+m-((device const float*)x)[row*4096+y[row]];
-    }
+    if(lane==0) out[row]=log(s)+m-((device const float*)x)[row*4096+y[row]];
 }
 kernel void finish(device const float* rows [[buffer(0)]],
                    device float* out [[buffer(1)]],
@@ -61,6 +50,6 @@
 def kernel(logits, y):
     losses = torch.empty((8192,), device=logits.device)
     out = torch.empty((), device=logits.device)
-    _lib.rows(logits, y, losses, threads=8192*128, group_size=128)
+    _lib.rows(logits, y, losses, threads=8192*32, group_size=256)
     _lib.finish(losses, out, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=600, check=True)`  
→ run 2 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 13.04 ms, kernel 1.67 ms, speedup 7.82x

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text(); start=s.index(\"kernel void rows\"); end=s.index(\"kernel void finish\"); row=\"...`
- `python check.py`  
→ OK: relative error 0.00e+00 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -7,25 +7,37 @@
                  device const long* y [[buffer(1)]],
                  device float* out [[buffer(2)]],
                  uint group [[threadgroup_position_in_grid]],
+                 uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
-    uint row=group*8+sg;
-    float4 v[32];
+    uint row=group*4+tid/64;
+    uint rt=tid%64;
+    float4 v[16];
     float m=-INFINITY;
     #pragma unroll
-    for(uint i=0;i<32;++i) {
-        v[i]=x[row*1024+lane+i*32];
+    for(uint i=0;i<16;++i) {
+        v[i]=x[row*1024+rt+i*64];
         m=max(m,max(max(v[i].x,v[i].y),max(v[i].z,v[i].w)));
     }
+    threadgroup float shared[8];
     m=simd_max(m);
+    if(lane==0) shared[sg]=m;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    m=max(shared[sg&~1u],shared[(sg&~1u)+1]);
     float s=0;
     #pragma unroll
-    for(uint i=0;i<32;++i) {
+    for(uint i=0;i<16;++i) {
         float4 e=fast::exp(v[i]-m);
         s+=e.x+e.y+e.z+e.w;
     }
     s=simd_sum(s);
-    if(lane==0) out[row]=log(s)+m-((device const float*)x)[row*4096+y[row]];
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if(lane==0) shared[sg]=s;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if(rt==0) {
+        s=shared[sg]+shared[sg+1];
+        out[row]=log(s)+m-((device const float*)x)[row*4096+y[row]];
+    }
 }
 kernel void finish(device const float* rows [[buffer(0)]],
                    device float* out [[buffer(1)]],
@@ -50,6 +62,6 @@
 def kernel(logits, y):
     losses = torch.empty((8192,), device=logits.device)
     out = torch.empty((), device=logits.device)
-    _lib.rows(logits, y, losses, threads=8192*32, group_size=256)
+    _lib.rows(logits, y, losses, threads=8192*64, group_size=256)
     _lib.finish(losses, out, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=600, check=True)`  
→ run 3 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 14.61 ms, kernel 2.47 ms, speedup 5.93x

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text(); start=s.index(\"kernel void rows\"); end=s.index(\"kernel void finish\"); row=\"...`
- `python check.py`  
→ OK: relative error 7.53e-08 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -6,36 +6,35 @@
 kernel void rows(device const float4* x [[buffer(0)]],
                  device const long* y [[buffer(1)]],
                  device float* out [[buffer(2)]],
-                 uint group [[threadgroup_position_in_grid]],
+                 uint row [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
-    uint row=group*4+tid/64;
-    uint rt=tid%64;
-    float4 v[16];
-    float m=-INFINITY;
+    half4 v[8];
+    float m = -INFINITY;
     #pragma unroll
-    for(uint i=0;i<16;++i) {
-        v[i]=x[row*1024+rt+i*64];
-        m=max(m,max(max(v[i].x,v[i].y),max(v[i].z,v[i].w)));
+    for (uint i=0; i<8; ++i) {
+        v[i] = half4(x[row*1024 + tid + i*128]);
+        m = max(m, max(max(v[i].x,v[i].y),max(v[i].z,v[i].w)));
     }
-    threadgroup float shared[8];
-    m=simd_max(m);
-    if(lane==0) shared[sg]=m;
+    threadgroup float shared[4];
+    m = simd_max(m);
+    if (lane==0) shared[sg]=m;
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    m=max(shared[sg&~1u],shared[(sg&~1u)+1]);
+    m=simd_max(lane<4 ? shared[lane] : -INFINITY);
     float s=0;
     #pragma unroll
-    for(uint i=0;i<16;++i) {
-        float4 e=fast::exp(v[i]-m);
+    for (uint i=0;i<8;++i) {
+        float4 e=fast::exp(float4(v[i])-m);
         s+=e.x+e.y+e.z+e.w;
     }
     s=simd_sum(s);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     if(lane==0) shared[sg]=s;
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    if(rt==0) {
-        s=shared[sg]+shared[sg+1];
+    if(tid==0) {
+        s=0;
+        for(uint i=0;i<4;++i) s+=shared[i];
         out[row]=log(s)+m-((device const float*)x)[row*4096+y[row]];
     }
 }
@@ -62,6 +61,6 @@
 def kernel(logits, y):
     losses = torch.empty((8192,), device=logits.device)
     out = torch.empty((), device=logits.device)
-    _lib.rows(logits, y, losses, threads=8192*64, group_size=256)
+    _lib.rows(logits, y, losses, threads=8192*128, group_size=128)
     _lib.finish(losses, out, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=600, check=True)`  
→ run 4 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 15.49 ms, kernel 2.20 ms, speedup 7.04x

> Using 128 threads per row improved the time to 1.67 ms. A smaller, barrier-free reduction was slower, so I’m testing layouts that balance synchronization costs against register use.

- `python -c 'from pathlib import Path; s=Path(\"queue/run2.py\").read_text().replace(\"float4 v[8]\", \"half4 v[8]\").replace(\"v[i] = x[\", \"v[i] = half4(x[\...`
- `python check.py`  
→ METAL_FUNC float max(float x, float y)  
→                  ^
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -15,7 +15,7 @@
     #pragma unroll
     for (uint i=0; i<8; ++i) {
         v[i] = half4(x[row*1024 + tid + i*128]);
-        m = max(m, max(max(v[i].x,v[i].y),max(v[i].z,v[i].w)));
+        m = max(m, float(max(max(v[i].x,v[i].y),max(v[i].z,v[i].w))));
     }
     threadgroup float shared[4];
     m = simd_max(m);

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=600, check=True)`  
→ METAL_FUNC float max(float x, float y)  
→                  ^

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text().replace(\"m = max(m, max(max(v[i].x,v[i].y),max(v[i].z,v[i].w)));\", \"m = max(m,...`
- `python check.py`  
→ OK: relative error 1.42e-06 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -10,32 +10,19 @@
                  uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
-    half4 v[8];
-    float m = -INFINITY;
-    #pragma unroll
-    for (uint i=0; i<8; ++i) {
-        v[i] = half4(x[row*1024 + tid + i*128]);
-        m = max(m, float(max(max(v[i].x,v[i].y),max(v[i].z,v[i].w))));
-    }
-    threadgroup float shared[4];
-    m = simd_max(m);
-    if (lane==0) shared[sg]=m;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    m=simd_max(lane<4 ? shared[lane] : -INFINITY);
     float s=0;
     #pragma unroll
-    for (uint i=0;i<8;++i) {
-        float4 e=fast::exp(float4(v[i])-m);
+    for(uint i=0;i<8;++i) {
+        float4 e=fast::exp(x[row*1024+tid+i*128]);
         s+=e.x+e.y+e.z+e.w;
     }
     s=simd_sum(s);
-    threadgroup_barrier(mem_flags::mem_threadgroup);
+    threadgroup float shared[4];
     if(lane==0) shared[sg]=s;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     if(tid==0) {
-        s=0;
-        for(uint i=0;i<4;++i) s+=shared[i];
-        out[row]=log(s)+m-((device const float*)x)[row*4096+y[row]];
+        s=shared[0]+shared[1]+shared[2]+shared[3];
+        out[row]=log(s)-((device const float*)x)[row*4096+y[row]];
     }
 }
 kernel void finish(device const float* rows [[buffer(0)]],

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=600, check=True)`  
→ run 6 of 10 submitted.  
→ cross_entropy: correct (relative error 3.1e-06). reference 13.32 ms, kernel 1.82 ms, speedup 7.31x

- `python -c 'from pathlib import Path; s=Path(\"queue/run2.py\").read_text(); start=s.index(\"    float4 v[8];\"); end=s.index(\"kernel void finish\"); row=\"\...`
- `python check.py`  
→ OK: relative error 0.00e+00 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -6,24 +6,15 @@
 kernel void rows(device const float4* x [[buffer(0)]],
                  device const long* y [[buffer(1)]],
                  device float* out [[buffer(2)]],
-                 uint row [[threadgroup_position_in_grid]],
-                 uint tid [[thread_index_in_threadgroup]],
+                 uint group [[threadgroup_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
-    float s=0;
+    uint row=group*8+sg;
+    float4 acc=0;
     #pragma unroll
-    for(uint i=0;i<8;++i) {
-        float4 e=fast::exp(x[row*1024+tid+i*128]);
-        s+=e.x+e.y+e.z+e.w;
-    }
-    s=simd_sum(s);
-    threadgroup float shared[4];
-    if(lane==0) shared[sg]=s;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    if(tid==0) {
-        s=shared[0]+shared[1]+shared[2]+shared[3];
-        out[row]=log(s)-((device const float*)x)[row*4096+y[row]];
-    }
+    for(uint i=0;i<32;++i) acc+=fast::exp(x[row*1024+lane+i*32]);
+    float s=simd_sum(acc.x+acc.y+acc.z+acc.w);
+    if(lane==0) out[row]=log(s)-((device const float*)x)[row*4096+y[row]];
 }
 kernel void finish(device const float* rows [[buffer(0)]],
                    device float* out [[buffer(1)]],
@@ -48,6 +39,6 @@
 def kernel(logits, y):
     losses = torch.empty((8192,), device=logits.device)
     out = torch.empty((), device=logits.device)
-    _lib.rows(logits, y, losses, threads=8192*128, group_size=128)
+    _lib.rows(logits, y, losses, threads=8192*32, group_size=256)
     _lib.finish(losses, out, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=600, check=True)`  
→ run 7 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 17.15 ms, kernel 2.53 ms, speedup 6.78x

> The reduced-precision version did not improve performance. GPU contention is also varying between runs, so I’ll use the remaining trials to compare the best stable version with simpler reductions, then check the final choice on fresh inputs.

- `python -c 'from pathlib import Path; s=Path(\"queue/run3.py\").read_text(); start=s.index(\"    float4 v[32];\"); end=s.index(\"kernel void finish\"); row=\"...`
- `python check.py`  
→ OK: relative error 1.50e-07 (allowed 2e-03)
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -7,14 +7,60 @@
                  device const long* y [[buffer(1)]],
                  device float* out [[buffer(2)]],
                  uint group [[threadgroup_position_in_grid]],
+                 uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
-    uint row=group*8+sg;
-    float4 acc=0;
-    #pragma unroll
-    for(uint i=0;i<32;++i) acc+=fast::exp(x[row*1024+lane+i*32]);
-    float s=simd_sum(acc.x+acc.y+acc.z+acc.w);
-    if(lane==0) out[row]=log(s)-((device const float*)x)[row*4096+y[row]];
+    uint row=group*2+tid/128;
+    uint rt=tid%128;
+    float4 v[8];
+    float m = -INFINITY;
+    v[0]=x[row*1024+rt+0];
+    m=max(m,max(max(v[0].x,v[0].y),max(v[0].z,v[0].w)));
+    v[1]=x[row*1024+rt+128];
+    m=max(m,max(max(v[1].x,v[1].y),max(v[1].z,v[1].w)));
+    v[2]=x[row*1024+rt+256];
+    m=max(m,max(max(v[2].x,v[2].y),max(v[2].z,v[2].w)));
+    v[3]=x[row*1024+rt+384];
+    m=max(m,max(max(v[3].x,v[3].y),max(v[3].z,v[3].w)));
+    v[4]=x[row*1024+rt+512];
+    m=max(m,max(max(v[4].x,v[4].y),max(v[4].z,v[4].w)));
+    v[5]=x[row*1024+rt+640];
+    m=max(m,max(max(v[5].x,v[5].y),max(v[5].z,v[5].w)));
+    v[6]=x[row*1024+rt+768];
+    m=max(m,max(max(v[6].x,v[6].y),max(v[6].z,v[6].w)));
+    v[7]=x[row*1024+rt+896];
+    m=max(m,max(max(v[7].x,v[7].y),max(v[7].z,v[7].w)));
+    threadgroup float shared[8];
+    m = simd_max(m);
+    if (lane==0) shared[sg]=m;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    m=simd_max(lane<4 ? shared[(sg&~3u)+lane] : -INFINITY);
+    float s=0;
+    float4 e0=fast::exp(v[0]-m);
+    s+=e0.x+e0.y+e0.z+e0.w;
+    float4 e1=fast::exp(v[1]-m);
+    s+=e1.x+e1.y+e1.z+e1.w;
+    float4 e2=fast::exp(v[2]-m);
+    s+=e2.x+e2.y+e2.z+e2.w;
+    float4 e3=fast::exp(v[3]-m);
+    s+=e3.x+e3.y+e3.z+e3.w;
+    float4 e4=fast::exp(v[4]-m);
+    s+=e4.x+e4.y+e4.z+e4.w;
+    float4 e5=fast::exp(v[5]-m);
+    s+=e5.x+e5.y+e5.z+e5.w;
+    float4 e6=fast::exp(v[6]-m);
+    s+=e6.x+e6.y+e6.z+e6.w;
+    float4 e7=fast::exp(v[7]-m);
+    s+=e7.x+e7.y+e7.z+e7.w;
+    s=simd_sum(s);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if(lane==0) shared[sg]=s;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if(rt==0) {
+        s=0;
+        for(uint i=0;i<4;++i) s+=shared[sg+i];
+        out[row]=log(s)+m-((device const float*)x)[row*4096+y[row]];
+    }
 }
 kernel void finish(device const float* rows [[buffer(0)]],
                    device float* out [[buffer(1)]],
@@ -39,6 +85,6 @@
 def kernel(logits, y):
     losses = torch.empty((8192,), device=logits.device)
     out = torch.empty((), device=logits.device)
-    _lib.rows(logits, y, losses, threads=8192*32, group_size=256)
+    _lib.rows(logits, y, losses, threads=8192*128, group_size=256)
     _lib.finish(losses, out, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=600, check=True)`  
→ run 8 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 16.01 ms, kernel 2.75 ms, speedup 5.83x

- `python -c 'from pathlib import Path; s=Path(\"queue/run2.py\").read_text(); s=s.replace(\"uint row [[threadgroup_position_in_grid]]\", \"uint group [[threadg...`
- `python check.py`  
→ OK: relative error 1.50e-07 (allowed 2e-03)
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -1,3 +1,4 @@
+"""Fused float32 Metal cross entropy for 8192 by 4096 logits."""
 import torch
 
 _source = r'''
@@ -10,8 +11,8 @@
                  uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
-    uint row=group*2+tid/128;
-    uint rt=tid%128;
+    uint row=group;
+    uint rt=tid;
     float4 v[8];
     float m = -INFINITY;
     v[0]=x[row*1024+rt+0];
@@ -30,11 +31,11 @@
     m=max(m,max(max(v[6].x,v[6].y),max(v[6].z,v[6].w)));
     v[7]=x[row*1024+rt+896];
     m=max(m,max(max(v[7].x,v[7].y),max(v[7].z,v[7].w)));
-    threadgroup float shared[8];
+    threadgroup float shared[4];
     m = simd_max(m);
     if (lane==0) shared[sg]=m;
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    m=simd_max(lane<4 ? shared[(sg&~3u)+lane] : -INFINITY);
+    m=simd_max(lane<4 ? shared[lane] : -INFINITY);
     float s=0;
     float4 e0=fast::exp(v[0]-m);
     s+=e0.x+e0.y+e0.z+e0.w;
@@ -58,7 +59,7 @@
     threadgroup_barrier(mem_flags::mem_threadgroup);
     if(rt==0) {
         s=0;
-        for(uint i=0;i<4;++i) s+=shared[sg+i];
+        for(uint i=0;i<4;++i) s+=shared[i];
         out[row]=log(s)+m-((device const float*)x)[row*4096+y[row]];
     }
 }
@@ -83,8 +84,8 @@
 _lib = torch.mps.compile_shader(_source)
 
 def kernel(logits, y):
-    losses = torch.empty((8192,), device=logits.device)
-    out = torch.empty((), device=logits.device)
-    _lib.rows(logits, y, losses, threads=8192*128, group_size=256)
+    losses = torch.empty((8192,), device=logits.device, dtype=torch.float32)
+    out = torch.empty((), device=logits.device, dtype=torch.float32)
+    _lib.rows(logits, y, losses, threads=8192*128, group_size=128)
     _lib.finish(losses, out, threads=256, group_size=256)
     return out

```

</details>

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=600, check=True)`  
→     _lib.finish(losses, out, threads=256, group_size=256)  
→     return out

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text(); s=s.replace(\"import torch\", \"\\\"\\\"\\\"Fused float32 Metal cross entropy fo...`
- `python check.py`  
→ OK: relative error 1.50e-07 (allowed 2e-03)
### Bench run 11

`python -c 'import subprocess; subprocess.run([\"python\", \"bench.py\"], timeout=600, check=True)`  
→ run 10 of 10 submitted.  
→ cross_entropy: correct (relative error 1.5e-07). reference 13.72 ms, kernel 1.99 ms, speedup 6.88x

- `python -c 'from pathlib import Path; s=Path(\"queue/run2.py\").read_text(); s=\"\\\"\\\"\\\"Fused float32 Metal cross entropy for 8192 by 4096 logits.\\\"\\\...`
- `python check.py`  
→ OK: relative error 7.52e-08 (allowed 2e-03)
> Best benchmark: **1.67 ms, 7.82× speedup**. Final `check.py` passed with relative error **7.52×10⁻⁸**.
> 
> EDIT: Final kernel.py uses fused Metal shaders for stable float32 cross-entropy and GPU mean reduction.

