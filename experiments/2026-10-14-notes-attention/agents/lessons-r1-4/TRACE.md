# Agent `lessons-r1-4`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what is still true from the notes you were given. At most 300 words, in words, no code.

10 min, 71,269 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 16.637 ms | 4.09x |
| run2 | 11.665 ms | 6.26x |
| run3 | 9.256 ms | 6.89x |
| run4 | 10.448 ms | 6.02x |
| run5 | 10.397 ms | 6.19x |
| run6 | 11.247 ms | 6.06x |
| run7 | wrong: crashed |  |
| run8 | 10.305 ms | 6.23x |
| run9 | 10.518 ms | 6.74x |
| run10 | 8.3 ms | 7.67x |
| final | 8.9 ms | 7.66x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the task files and prior notes, then use the benchmark budget to test optimizations and check the final kernel for correctness.

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

`pwd; rg --files -g 'AGENTS.md' -g 'kernel.py' -g 'reference.py' -g 'bench.py' -g 'check.py' -g 'LESSONS.md`  
→ check.py  
→ kernel.py

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

`cat kernel.py reference.py bench.py check.py LESSONS.md`  
→ print(f"{'OK' if got.shape == want.shape and err <= TOL else 'WRONG'}: relative error {err:.2e} (allowed {TOL:.0e})")  
→ No notes yet: you are the first agents on this task.

- `python -c 'import torch; print(torch.__version__); print(torch.mps.compile_shader.__doc__); print(torch.nn.functional.scaled_dot_product_attention.__doc__[:4...`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 1.85e-07 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -1,6 +1,51 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
+"""Causal attention with half precision matrix products and fused softmax."""
 import torch
 
+_lib = torch.mps.compile_shader(r"""
+#include <metal_stdlib>
+using namespace metal;
+kernel void causal_softmax(device const half* scores [[buffer(0)]],
+                           device half* probs [[buffer(1)]],
+                           uint row [[threadgroup_position_in_grid]],
+                           uint tid [[thread_index_in_threadgroup]],
+                           uint lane [[thread_index_in_simdgroup]],
+                           uint sg [[simdgroup_index_in_threadgroup]]) {
+    threadgroup float scratch[8];
+    uint pos = row % 1024;
+    uint base = row * 1024;
+    float x[4];
+    float mx = -INFINITY;
+    for (uint i=0; i<4; ++i) {
+        uint col = tid + i*256;
+        x[i] = col <= pos ? float(scores[base+col])*0.125f : -INFINITY;
+        mx = max(mx,x[i]);
+    }
+    mx = simd_max(mx);
+    if (lane==0) scratch[sg]=mx;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    mx = tid<8 ? scratch[tid] : -INFINITY;
+    mx = simd_max(mx);
+    if (tid==0) scratch[0]=mx;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    mx=scratch[0];
+    float sum=0;
+    for (uint i=0; i<4; ++i) { x[i]=fast::exp(x[i]-mx); sum+=x[i]; }
+    sum=simd_sum(sum);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if(lane==0) scratch[sg]=sum;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    sum=tid<8?scratch[tid]:0;
+    sum=simd_sum(sum);
+    if(tid==0) scratch[0]=sum;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    float inv=1.0f/scratch[0];
+    for(uint i=0;i<4;++i) probs[base+tid+i*256]=half(x[i]*inv);
+}
+""")
 
 def kernel(q, k, v):
-    return torch.nn.functional.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=True).float()
+    qh, kh, vh = q.half(), k.half(), v.half()
+    scores = qh @ kh.transpose(-1,-2)
+    probs = torch.empty_like(scores)
+    _lib.causal_softmax(scores, probs, threads=(q.numel()//64*256), group_size=256)
+    return (probs @ vh).float()

```

</details>

`python bench.py`  
→ run 1 of 10 submitted.  
→ attention: correct (relative error 2.0e-07). reference 67.98 ms, kernel 16.64 ms, speedup 4.09x

- `python -c 'import torch; lib=torch.mps.compile_shader(\"kernel void copy(device const float* x, device float* y, uint i [[thread_position_in_grid]]) { y[i]=x...`
> PyTorch’s built-in causal attention passed the check and reduced runtime from about 68 ms to 16.64 ms. I’ll test whether float16 inputs improve it further while staying within the allowed error.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 5.20e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -3,49 +3,101 @@
 
 _lib = torch.mps.compile_shader(r"""
 #include <metal_stdlib>
+#include <metal_simdgroup_matrix>
 using namespace metal;
-kernel void causal_softmax(device const half* scores [[buffer(0)]],
-                           device half* probs [[buffer(1)]],
-                           uint row [[threadgroup_position_in_grid]],
-                           uint tid [[thread_index_in_threadgroup]],
-                           uint lane [[thread_index_in_simdgroup]],
-                           uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup float scratch[8];
-    uint pos = row % 1024;
-    uint base = row * 1024;
-    float x[4];
-    float mx = -INFINITY;
-    for (uint i=0; i<4; ++i) {
-        uint col = tid + i*256;
-        x[i] = col <= pos ? float(scores[base+col])*0.125f : -INFINITY;
-        mx = max(mx,x[i]);
+kernel void attention(device const half* q [[buffer(0)]],
+                      device const half* k [[buffer(1)]],
+                      device const half* v [[buffer(2)]],
+                      device float* out [[buffer(3)]],
+                      uint2 group [[threadgroup_position_in_grid]],
+                      uint tid [[thread_index_in_threadgroup]],
+                      uint lane [[thread_index_in_simdgroup]],
+                      uint sg [[simdgroup_index_in_threadgroup]]) {
+    threadgroup half qs[32*64];
+    threadgroup half ks[32*64];
+    threadgroup half vs[32*64];
+    threadgroup half ps[32*32];
+    uint head = group.y * 1024 * 64;
+    uint qr = group.x * 32;
+    uint rr = (lane%8)/2 + (lane/16)*4;
+    uint cc = (lane%2)*2 + ((lane/8)%2)*4;
+    uint row = qr + sg*8 + rr;
+    for(uint i=tid; i<32*64; i+=128) qs[i]=q[head+qr*64+i];
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    simdgroup_half8x8 qm[8];
+    for(uint d=0;d<8;++d) simdgroup_load(qm[d],qs+sg*8*64+d*8,64);
+    simdgroup_float8x8 acc[8];
+    for(uint d=0;d<8;++d) acc[d]=simdgroup_float8x8(0);
+    float m=-INFINITY;
+    float l=0;
+    for(uint kb=0;kb<=qr;kb+=32) {
+        for(uint i=tid;i<32*64;i+=128) {
+            ks[i]=k[head+kb*64+i];
+            vs[i]=v[head+kb*64+i];
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        simdgroup_float8x8 sm[4];
+        for(uint j=0;j<4;++j) sm[j]=simdgroup_float8x8(0);
+        for(uint d=0;d<8;++d) {
+            for(uint j=0;j<4;++j) {
+                simdgroup_half8x8 km;
+                simdgroup_load(km,ks+j*8*64+d*8,64,ulong2(0),true);
+                simdgroup_multiply_accumulate(sm[j],qm[d],km,sm[j]);
+            }
+        }
+        float mx=-INFINITY;
+        for(uint j=0;j<4;++j) {
+            for(uint e=0;e<2;++e) {
+                float x=sm[j].thread_elements()[e]*0.125f;
+                if(kb+j*8+cc+e>row) x=-INFINITY;
+                sm[j].thread_elements()[e]=x;
+                mx=max(mx,x);
+            }
+        }
+        mx=max(mx,simd_shuffle_xor(mx,1));
+        mx=max(mx,simd_shuffle_xor(mx,8));
+        float nm=max(m,mx);
+        float alpha=fast::exp(m-nm);
+        float sum=0;
+        for(uint j=0;j<4;++j) {
+            simdgroup_half8x8 pm;
+            for(uint e=0;e<2;++e) {
+                float p=fast::exp(sm[j].thread_elements()[e]-nm);
+                pm.thread_elements()[e]=half(p);
+                sum+=p;
+            }
+            simdgroup_store(pm,ps+sg*8*32+j*8,32);
+        }
+        sum+=simd_shuffle_xor(sum,1);
+        sum+=simd_shuffle_xor(sum,8);
+        l=l*alpha+sum;
+        m=nm;
+        for(uint d=0;d<8;++d) {
+            acc[d].thread_elements()[0]*=alpha;
+            acc[d].thread_elements()[1]*=alpha;
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for(uint j=0;j<4;++j) {
+            simdgroup_half8x8 pm;
+            simdgroup_load(pm,ps+sg*8*32+j*8,32);
+            for(uint d=0;d<8;++d) {
+                simdgroup_half8x8 vm;
+                simdgroup_load(vm,vs+j*8*64+d*8,64);
+                simdgroup_multiply_accumulate(acc[d],pm,vm,acc[d]);
+            }
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
     }
-    mx = simd_max(mx);
-    if (lane==0) scratch[sg]=mx;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    mx = tid<8 ? scratch[tid] : -INFINITY;
-    mx = simd_max(mx);
-    if (tid==0) scratch[0]=mx;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    mx=scratch[0];
-    float sum=0;
-    for (uint i=0; i<4; ++i) { x[i]=fast::exp(x[i]-mx); sum+=x[i]; }
-    sum=simd_sum(sum);
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    if(lane==0) scratch[sg]=sum;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    sum=tid<8?scratch[tid]:0;
-    sum=simd_sum(sum);
-    if(tid==0) scratch[0]=sum;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    float inv=1.0f/scratch[0];
-    for(uint i=0;i<4;++i) probs[base+tid+i*256]=half(x[i]*inv);
+    for(uint d=0;d<8;++d) {
+        acc[d].thread_elements()[0]/=l;
+        acc[d].thread_elements()[1]/=l;
+        simdgroup_store(acc[d],out+head+(qr+sg*8)*64+d*8,64);
+    }
 }
 """)
 
 def kernel(q, k, v):
     qh, kh, vh = q.half(), k.half(), v.half()
-    scores = qh @ kh.transpose(-1,-2)
-    probs = torch.empty_like(scores)
-    _lib.causal_softmax(scores, probs, threads=(q.numel()//64*256), group_size=256)
-    return (probs @ vh).float()
+    out = torch.empty_like(q)
+    _lib.attention(qh, kh, vh, out, threads=(32*128,64), group_size=(128,1))
+    return out

```

</details>

`python bench.py`  
→ run 2 of 10 submitted.  
→ attention: correct (relative error 8.6e-04). reference 73.00 ms, kernel 11.67 ms, speedup 6.26x

- `python -c 'import torch; print(torch.backends.mps.is_available()); print(torch.ops.aten._scaled_dot_product_attention_math_for_mps.default._schema)`
- `cat > kernel.py <<'PY` (exited 1 in 477ms:)
- edited `solution.py`
- `python check.py`  
→ OK: relative error 6.43e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -5,6 +5,9 @@
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
+#define BQ 64
+#define BK 32
+#define NT (BQ*4)
 kernel void attention(device const half* q [[buffer(0)]],
                       device const half* k [[buffer(1)]],
                       device const half* v [[buffer(2)]],
@@ -13,16 +16,16 @@
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup half qs[32*64];
-    threadgroup half ks[32*64];
-    threadgroup half vs[32*64];
-    threadgroup half ps[32*32];
+    threadgroup half qs[BQ*64];
+    threadgroup half ks[BK*64];
+    threadgroup half vs[BK*64];
+    threadgroup half ps[BQ*BK];
     uint head = group.y * 1024 * 64;
-    uint qr = group.x * 32;
+    uint qr = group.x * BQ;
     uint rr = (lane%8)/2 + (lane/16)*4;
     uint cc = (lane%2)*2 + ((lane/8)%2)*4;
     uint row = qr + sg*8 + rr;
-    for(uint i=tid; i<32*64; i+=128) qs[i]=q[head+qr*64+i];
+    for(uint i=tid; i<BQ*64; i+=NT) qs[i]=q[head+qr*64+i];
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
     for(uint d=0;d<8;++d) simdgroup_load(qm[d],qs+sg*8*64+d*8,64);
@@ -30,23 +33,23 @@
     for(uint d=0;d<8;++d) acc[d]=simdgroup_float8x8(0);
     float m=-INFINITY;
     float l=0;
-    for(uint kb=0;kb<=qr;kb+=32) {
-        for(uint i=tid;i<32*64;i+=128) {
+    for(uint kb=0;kb<qr+BQ;kb+=BK) {
+        for(uint i=tid;i<BK*64;i+=NT) {
             ks[i]=k[head+kb*64+i];
             vs[i]=v[head+kb*64+i];
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 sm[4];
-        for(uint j=0;j<4;++j) sm[j]=simdgroup_float8x8(0);
+        simdgroup_float8x8 sm[BK/8];
+        for(uint j=0;j<BK/8;++j) sm[j]=simdgroup_float8x8(0);
         for(uint d=0;d<8;++d) {
-            for(uint j=0;j<4;++j) {
+            for(uint j=0;j<BK/8;++j) {
                 simdgroup_half8x8 km;
                 simdgroup_load(km,ks+j*8*64+d*8,64,ulong2(0),true);
                 simdgroup_multiply_accumulate(sm[j],qm[d],km,sm[j]);
             }
         }
         float mx=-INFINITY;
-        for(uint j=0;j<4;++j) {
+        for(uint j=0;j<BK/8;++j) {
             for(uint e=0;e<2;++e) {
                 float x=sm[j].thread_elements()[e]*0.125f;
                 if(kb+j*8+cc+e>row) x=-INFINITY;
@@ -59,14 +62,14 @@
         float nm=max(m,mx);
         float alpha=fast::exp(m-nm);
         float sum=0;
-        for(uint j=0;j<4;++j) {
+        for(uint j=0;j<BK/8;++j) {
             simdgroup_half8x8 pm;
             for(uint e=0;e<2;++e) {
                 float p=fast::exp(sm[j].thread_elements()[e]-nm);
                 pm.thread_elements()[e]=half(p);
                 sum+=p;
             }
-            simdgroup_store(pm,ps+sg*8*32+j*8,32);
+            simdgroup_store(pm,ps+sg*8*BK+j*8,BK);
         }
         sum+=simd_shuffle_xor(sum,1);
         sum+=simd_shuffle_xor(sum,8);
@@ -77,9 +80,9 @@
             acc[d].thread_elements()[1]*=alpha;
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        for(uint j=0;j<4;++j) {
+        for(uint j=0;j<BK/8;++j) {
             simdgroup_half8x8 pm;
-            simdgroup_load(pm,ps+sg*8*32+j*8,32);
+            simdgroup_load(pm,ps+sg*8*BK+j*8,BK);
             for(uint d=0;d<8;++d) {
                 simdgroup_half8x8 vm;
                 simdgroup_load(vm,vs+j*8*64+d*8,64);
@@ -99,5 +102,5 @@
 def kernel(q, k, v):
     qh, kh, vh = q.half(), k.half(), v.half()
     out = torch.empty_like(q)
-    _lib.attention(qh, kh, vh, out, threads=(32*128,64), group_size=(128,1))
+    _lib.attention(qh, kh, vh, out, threads=(16*256,64), group_size=(256,1))
     return out

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 8.6e-04). reference 63.75 ms, kernel 9.26 ms, speedup 6.89x

> Float16 attention took 11.67 ms. Using half-precision matrix products with a custom Metal softmax reduced that to 9.26 ms, with error still below tolerance. I’ll now try a tiled shader that computes attention without storing the full score matrix.

- `python -c 'import torch; lib=torch.mps.compile_shader(\"#include <metal_stdlib>\\n#include <metal_simdgroup_matrix>\\nusing namespace metal;\\nkernel void la...`
- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.59e-04 (allowed 2e-03)
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -6,7 +6,7 @@
 #include <metal_simdgroup_matrix>
 using namespace metal;
 #define BQ 64
-#define BK 32
+#define BK 64
 #define NT (BQ*4)
 kernel void attention(device const half* q [[buffer(0)]],
                       device const half* k [[buffer(1)]],

```

</details>

`python bench.py`  
→ run 4 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 62.86 ms, kernel 10.45 ms, speedup 6.02x

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.52e-04 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -16,19 +16,19 @@
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup half qs[BQ*64];
-    threadgroup half ks[BK*64];
-    threadgroup half vs[BK*64];
-    threadgroup half ps[BQ*BK];
+    threadgroup half tiles[max(BQ*64,2*BK*64)];
+    threadgroup half* ks=tiles;
+    threadgroup half* vs=tiles+BK*64;
     uint head = group.y * 1024 * 64;
     uint qr = group.x * BQ;
     uint rr = (lane%8)/2 + (lane/16)*4;
     uint cc = (lane%2)*2 + ((lane/8)%2)*4;
     uint row = qr + sg*8 + rr;
-    for(uint i=tid; i<BQ*64; i+=NT) qs[i]=q[head+qr*64+i];
+    for(uint i=tid; i<BQ*64; i+=NT) tiles[i]=q[head+qr*64+i];
     threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_half8x8 qm[8];
-    for(uint d=0;d<8;++d) simdgroup_load(qm[d],qs+sg*8*64+d*8,64);
+    for(uint d=0;d<8;++d) simdgroup_load(qm[d],tiles+sg*8*64+d*8,64);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     simdgroup_float8x8 acc[8];
     for(uint d=0;d<8;++d) acc[d]=simdgroup_float8x8(0);
     float m=-INFINITY;
@@ -61,6 +61,10 @@
         mx=max(mx,simd_shuffle_xor(mx,8));
         float nm=max(m,mx);
         float alpha=fast::exp(m-nm);
+        for(uint d=0;d<8;++d) {
+            acc[d].thread_elements()[0]*=alpha;
+            acc[d].thread_elements()[1]*=alpha;
+        }
         float sum=0;
         for(uint j=0;j<BK/8;++j) {
             simdgroup_half8x8 pm;
@@ -69,26 +73,16 @@
                 pm.thread_elements()[e]=half(p);
                 sum+=p;
             }
-            simdgroup_store(pm,ps+sg*8*BK+j*8,BK);
-        }
-        sum+=simd_shuffle_xor(sum,1);
-        sum+=simd_shuffle_xor(sum,8);
-        l=l*alpha+sum;
-        m=nm;
-        for(uint d=0;d<8;++d) {
-            acc[d].thread_elements()[0]*=alpha;
-            acc[d].thread_elements()[1]*=alpha;
-        }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        for(uint j=0;j<BK/8;++j) {
-            simdgroup_half8x8 pm;
-            simdgroup_load(pm,ps+sg*8*BK+j*8,BK);
             for(uint d=0;d<8;++d) {
                 simdgroup_half8x8 vm;
                 simdgroup_load(vm,vs+j*8*64+d*8,64);
                 simdgroup_multiply_accumulate(acc[d],pm,vm,acc[d]);
             }
         }
+        sum+=simd_shuffle_xor(sum,1);
+        sum+=simd_shuffle_xor(sum,8);
+        l=l*alpha+sum;
+        m=nm;
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
     for(uint d=0;d<8;++d) {

```

</details>

`python bench.py`  
→ run 5 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 64.40 ms, kernel 10.40 ms, speedup 6.19x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.26e-04 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -5,7 +5,7 @@
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-#define BQ 64
+#define BQ 128
 #define BK 64
 #define NT (BQ*4)
 kernel void attention(device const half* q [[buffer(0)]],
@@ -16,7 +16,7 @@
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup half tiles[max(BQ*64,2*BK*64)];
+    threadgroup half tiles[(BQ > 2*BK ? BQ : 2*BK)*64];
     threadgroup half* ks=tiles;
     threadgroup half* vs=tiles+BK*64;
     uint head = group.y * 1024 * 64;
@@ -96,5 +96,5 @@
 def kernel(q, k, v):
     qh, kh, vh = q.half(), k.half(), v.half()
     out = torch.empty_like(q)
-    _lib.attention(qh, kh, vh, out, threads=(16*256,64), group_size=(256,1))
+    _lib.attention(qh, kh, vh, out, threads=(8*512,64), group_size=(512,1))
     return out

```

</details>

`python bench.py`  
→ run 6 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 68.15 ms, kernel 11.25 ms, speedup 6.06x

> The tiled shader is correct, but its first two versions took about 10.4 ms—slower than the 9.26 ms matrix-product approach. I found some unnecessary shared-memory traffic in the tiled version and will test removing it before choosing the final kernel.

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ METAL_FUNC int max(int x, int y)  
→                ^
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -3,98 +3,37 @@
 
 _lib = torch.mps.compile_shader(r"""
 #include <metal_stdlib>
-#include <metal_simdgroup_matrix>
 using namespace metal;
-#define BQ 128
-#define BK 64
-#define NT (BQ*4)
-kernel void attention(device const half* q [[buffer(0)]],
-                      device const half* k [[buffer(1)]],
-                      device const half* v [[buffer(2)]],
-                      device float* out [[buffer(3)]],
-                      uint2 group [[threadgroup_position_in_grid]],
-                      uint tid [[thread_index_in_threadgroup]],
-                      uint lane [[thread_index_in_simdgroup]],
-                      uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup half tiles[(BQ > 2*BK ? BQ : 2*BK)*64];
-    threadgroup half* ks=tiles;
-    threadgroup half* vs=tiles+BK*64;
-    uint head = group.y * 1024 * 64;
-    uint qr = group.x * BQ;
-    uint rr = (lane%8)/2 + (lane/16)*4;
-    uint cc = (lane%2)*2 + ((lane/8)%2)*4;
-    uint row = qr + sg*8 + rr;
-    for(uint i=tid; i<BQ*64; i+=NT) tiles[i]=q[head+qr*64+i];
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    simdgroup_half8x8 qm[8];
-    for(uint d=0;d<8;++d) simdgroup_load(qm[d],tiles+sg*8*64+d*8,64);
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    simdgroup_float8x8 acc[8];
-    for(uint d=0;d<8;++d) acc[d]=simdgroup_float8x8(0);
-    float m=-INFINITY;
-    float l=0;
-    for(uint kb=0;kb<qr+BQ;kb+=BK) {
-        for(uint i=tid;i<BK*64;i+=NT) {
-            ks[i]=k[head+kb*64+i];
-            vs[i]=v[head+kb*64+i];
-        }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 sm[BK/8];
-        for(uint j=0;j<BK/8;++j) sm[j]=simdgroup_float8x8(0);
-        for(uint d=0;d<8;++d) {
-            for(uint j=0;j<BK/8;++j) {
-                simdgroup_half8x8 km;
-                simdgroup_load(km,ks+j*8*64+d*8,64,ulong2(0),true);
-                simdgroup_multiply_accumulate(sm[j],qm[d],km,sm[j]);
-            }
-        }
-        float mx=-INFINITY;
-        for(uint j=0;j<BK/8;++j) {
-            for(uint e=0;e<2;++e) {
-                float x=sm[j].thread_elements()[e]*0.125f;
-                if(kb+j*8+cc+e>row) x=-INFINITY;
-                sm[j].thread_elements()[e]=x;
-                mx=max(mx,x);
-            }
-        }
-        mx=max(mx,simd_shuffle_xor(mx,1));
-        mx=max(mx,simd_shuffle_xor(mx,8));
-        float nm=max(m,mx);
-        float alpha=fast::exp(m-nm);
-        for(uint d=0;d<8;++d) {
-            acc[d].thread_elements()[0]*=alpha;
-            acc[d].thread_elements()[1]*=alpha;
-        }
-        float sum=0;
-        for(uint j=0;j<BK/8;++j) {
-            simdgroup_half8x8 pm;
-            for(uint e=0;e<2;++e) {
-                float p=fast::exp(sm[j].thread_elements()[e]-nm);
-                pm.thread_elements()[e]=half(p);
-                sum+=p;
-            }
-            for(uint d=0;d<8;++d) {
-                simdgroup_half8x8 vm;
-                simdgroup_load(vm,vs+j*8*64+d*8,64);
-                simdgroup_multiply_accumulate(acc[d],pm,vm,acc[d]);
-            }
-        }
-        sum+=simd_shuffle_xor(sum,1);
-        sum+=simd_shuffle_xor(sum,8);
-        l=l*alpha+sum;
-        m=nm;
-        threadgroup_barrier(mem_flags::mem_threadgroup);
+kernel void causal_softmax(device const half* scores [[buffer(0)]],
+                           device half* probs [[buffer(1)]],
+                           uint row [[threadgroup_position_in_grid]],
+                           uint tid [[thread_index_in_threadgroup]],
+                           uint lane [[thread_index_in_simdgroup]],
+                           uint sg [[simdgroup_index_in_threadgroup]]) {
+    uint base = row * 4 * 1024 + sg * 1024;
+    uint pos = (row * 4 + sg) % 1024;
+    float x[32];
+    float mx = -INFINITY;
+    for (uint i=0; i<32; ++i) {
+        uint col = lane + i*32;
+        x[i] = col <= pos ? float(scores[base+col])*0.125f : -INFINITY;
+        mx = max(mx,x[i]);
     }
-    for(uint d=0;d<8;++d) {
-        acc[d].thread_elements()[0]/=l;
-        acc[d].thread_elements()[1]/=l;
-        simdgroup_store(acc[d],out+head+(qr+sg*8)*64+d*8,64);
+    mx = simd_max(mx);
+    float sum = 0;
+    for (uint i=0; i<32; ++i) {
+        x[i] = fast::exp(x[i]-mx);
+        sum += x[i];
     }
+    sum = simd_sum(sum);
+    float inv = 1.0f/sum;
+    for (uint i=0; i<32; ++i) probs[base+lane+i*32]=half(x[i]*inv);
 }
 """)
 
 def kernel(q, k, v):
     qh, kh, vh = q.half(), k.half(), v.half()
-    out = torch.empty_like(q)
-    _lib.attention(qh, kh, vh, out, threads=(8*512,64), group_size=(512,1))
-    return out
+    scores = qh @ kh.transpose(-1,-2)
+    probs = torch.empty_like(scores)
+    _lib.causal_softmax(scores, probs, threads=(q.numel()//64*32), group_size=128)
+    return (probs @ vh).float()

```

</details>

`python bench.py`  
→ METAL_FUNC int max(int x, int y)  
→                ^

- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.73e-04 (allowed 2e-03)
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -3,37 +3,87 @@
 
 _lib = torch.mps.compile_shader(r"""
 #include <metal_stdlib>
+#include <metal_simdgroup_matrix>
 using namespace metal;
+kernel void causal_scores(device const half* q [[buffer(0)]],
+                          device const half* k [[buffer(1)]],
+                          device half* scores [[buffer(2)]],
+                          uint3 group [[threadgroup_position_in_grid]],
+                          uint tid [[thread_index_in_threadgroup]],
+                          uint sg [[simdgroup_index_in_threadgroup]]) {
+    if(group.y > group.x) return;
+    threadgroup half qs[64*64];
+    threadgroup half ks[64*64];
+    uint head=group.z*1024*64;
+    uint qr=group.x*64;
+    uint kr=group.y*64;
+    for(uint i=tid;i<64*64;i+=256) {
+        qs[i]=q[head+qr*64+i];
+        ks[i]=k[head+kr*64+i];
+    }
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    simdgroup_float8x8 acc[8];
+    for(uint j=0;j<8;++j) acc[j]=simdgroup_float8x8(0);
+    for(uint d=0;d<8;++d) {
+        simdgroup_half8x8 qm;
+        simdgroup_load(qm,qs+sg*8*64+d*8,64);
+        for(uint j=0;j<8;++j) {
+            simdgroup_half8x8 km;
+            simdgroup_load(km,ks+j*8*64+d*8,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(acc[j],qm,km,acc[j]);
+        }
+    }
+    uint base=group.z*1024*1024+(qr+sg*8)*1024+kr;
+    for(uint j=0;j<8;++j) {
+        simdgroup_half8x8 result;
+        result.thread_elements()[0]=half(acc[j].thread_elements()[0]);
+        result.thread_elements()[1]=half(acc[j].thread_elements()[1]);
+        simdgroup_store(result,scores+base+j*8,1024);
+    }
+}
 kernel void causal_softmax(device const half* scores [[buffer(0)]],
                            device half* probs [[buffer(1)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
-    uint base = row * 4 * 1024 + sg * 1024;
-    uint pos = (row * 4 + sg) % 1024;
-    float x[32];
+    threadgroup float scratch[8];
+    uint pos = row % 1024;
+    uint base = row * 1024;
+    float x[4];
     float mx = -INFINITY;
-    for (uint i=0; i<32; ++i) {
-        uint col = lane + i*32;
+    for (uint i=0; i<4; ++i) {
+        uint col = tid + i*256;
         x[i] = col <= pos ? float(scores[base+col])*0.125f : -INFINITY;
         mx = max(mx,x[i]);
     }
     mx = simd_max(mx);
-    float sum = 0;
-    for (uint i=0; i<32; ++i) {
-        x[i] = fast::exp(x[i]-mx);
-        sum += x[i];
-    }
-    sum = simd_sum(sum);
-    float inv = 1.0f/sum;
-    for (uint i=0; i<32; ++i) probs[base+lane+i*32]=half(x[i]*inv);
+    if (lane==0) scratch[sg]=mx;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    mx = tid<8 ? scratch[tid] : -INFINITY;
+    mx = simd_max(mx);
+    if (tid==0) scratch[0]=mx;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    mx=scratch[0];
+    float sum=0;
+    for (uint i=0; i<4; ++i) { x[i]=fast::exp(x[i]-mx); sum+=x[i]; }
+    sum=simd_sum(sum);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if(lane==0) scratch[sg]=sum;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    sum=tid<8?scratch[tid]:0;
+    sum=simd_sum(sum);
+    if(tid==0) scratch[0]=sum;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    float inv=1.0f/scratch[0];
+    for(uint i=0;i<4;++i) probs[base+tid+i*256]=half(x[i]*inv);
 }
 """)
 
 def kernel(q, k, v):
     qh, kh, vh = q.half(), k.half(), v.half()
-    scores = qh @ kh.transpose(-1,-2)
+    scores = torch.empty((*q.shape[:-1],1024), device=q.device, dtype=torch.float16)
+    _lib.causal_scores(qh, kh, scores, threads=(16*256,16,64), group_size=(256,1,1))
     probs = torch.empty_like(scores)
-    _lib.causal_softmax(scores, probs, threads=(q.numel()//64*32), group_size=128)
+    _lib.causal_softmax(scores, probs, threads=(q.numel()//64*256), group_size=256)
     return (probs @ vh).float()

```

</details>

`python bench.py`  
→ run 8 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 64.17 ms, kernel 10.31 ms, speedup 6.23x

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 8.27e-04 (allowed 2e-03)
### Bench run 11

`python bench.py`  
→ run 9 of 10 submitted.  
→ attention: correct (relative error 8.6e-04). reference 70.88 ms, kernel 10.52 ms, speedup 6.74x

- `cat kernel.py`
> The 256-thread Metal softmax with half-precision matrix products remains fastest at 9.26 ms. I have one benchmark run left and will use it to test skipping the unused upper triangle in the score calculation, then retain the fastest passing version.

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 6.38e-04 (allowed 2e-03)
### Bench run 12

`python bench.py`  
→ run 10 of 10 submitted.  
→ attention: correct (relative error 8.6e-04). reference 63.65 ms, kernel 8.30 ms, speedup 7.67x

> Skipping the upper triangle brought runtime down to 8.30 ms, a 7.67× speedup in the final benchmark. I’ll keep that version, record the results in LESSONS.md, and run the final correctness check.

- `cat kernel.py`
- edited `solution.py`
- `python check.py`  
→ OK: relative error 5.88e-04 (allowed 2e-03)
- `wc -w LESSONS.md`
> Final benchmark: **8.30 ms, 7.67× speedup**. Final check passed with relative error **5.88e-4**, below 2e-3. Updated LESSONS.md.
> 
> EDIT: kernel.py uses triangular Metal score tiles, a fused causal softmax, and half-precision matrix products with float32 accumulation and output.

