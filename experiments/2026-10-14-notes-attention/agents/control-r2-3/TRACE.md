# Agent `control-r2-3`

**Prompt:** Make kernel() as fast as you can.

13 min, 103,575 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 16.03 ms | 4.14x |
| run2 | 9.271 ms | 6.9x |
| run3 | 5.507 ms | 12.31x |
| run4 | 5.854 ms | 11.62x |
| run5 | 5.457 ms | 12.45x |
| run6 | 9.076 ms | 8.05x |
| run7 | 9.102 ms | 7.61x |
| run8 | 4.575 ms | 15.42x |
| run9 | 10.972 ms | 6.36x |
| run10 | 4.509 ms | 15.16x |
| final | 4.139 ms | 14.28x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll inspect the kernel, reference, and benchmark harness, then optimize within the ten-run timing budget and finish with a correctness check.

- `pwd && rg --files -g '*' -g '"'!.*`
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

`cat kernel.py reference.py bench.py check.py`  
→     err = (got.float() - want).abs().max().item() / want.abs().max().item()  
→ print(f"{'OK' if got.shape == want.shape and err <= TOL else 'WRONG'}: relative error {err:.2e} (allowed {TOL:.0e})")

- `python -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_available()); print(hasattr(torch.mps, \"compile_shader\")); print(torch.mps.c...`
- `cat > kernel.py <<'PY` (exited 1 in 587ms:)
- edited `solution.py`
- `python check.py`  
→ OK: relative error 9.48e-08 (allowed 2e-03)
### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -1,6 +1,41 @@
 import torch
-import torch.nn.functional as F
+
+_lib = torch.mps.compile_shader(r'''
+#include <metal_stdlib>
+using namespace metal;
+kernel void causal_softmax(device const half* s, device half* p,
+                           uint row [[threadgroup_position_in_grid]],
+                           uint tid [[thread_index_in_threadgroup]],
+                           uint lane [[thread_index_in_simdgroup]],
+                           uint sg [[simdgroup_index_in_threadgroup]]) {
+    threadgroup float scratch[8];
+    uint last = row % 1024;
+    float x[4];
+    float mx = -INFINITY;
+    for (uint i=0;i<4;i++) {
+        uint col=tid+i*256;
+        x[i] = col<=last ? float(s[row*1024+col])*0.125f : -INFINITY;
+        mx=max(mx,x[i]);
+    }
+    mx=simd_max(mx);
+    if(lane==0) scratch[sg]=mx;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    mx=simd_max(lane<8?scratch[lane]:-INFINITY);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    float sum=0;
+    for(uint i=0;i<4;i++){x[i]=fast::exp(x[i]-mx);sum+=x[i];}
+    sum=simd_sum(sum);
+    if(lane==0) scratch[sg]=sum;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    sum=simd_sum(lane<8?scratch[lane]:0.f);
+    for(uint i=0;i<4;i++) p[row*1024+tid+i*256]=half(x[i]/sum);
+}
+''')
 
 
 def kernel(q, k, v):
-    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
+    qh, kh, vh = q.half(), k.half(), v.half()
+    scores = qh @ kh.transpose(-1, -2)
+    p = torch.empty_like(scores)
+    _lib.causal_softmax(scores, p, threads=(q.numel() // 64 * 256), group_size=256)
+    return (p @ vh).float()

```

</details>

`python bench.py`  
→ None  
→ ['__call__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_pybind11_conduit_v1_', 'max_threads_per_threadgroup', 'static_thread_group_memory_length', 'thread_execution_width']

> PyTorch’s built-in attention passes the check and runs in 16.03 ms, a 4.14× speedup. I’ll now test half-precision matrix products with a custom Metal shader that combines causal masking and softmax.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 6.17e-04 (allowed 2e-03)
### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -30,12 +30,93 @@
     sum=simd_sum(lane<8?scratch[lane]:0.f);
     for(uint i=0;i<4;i++) p[row*1024+tid+i*256]=half(x[i]/sum);
 }
+kernel void flash(device const half* q, device const half* k, device const half* v,
+                  device float* out,
+                  uint tg [[threadgroup_position_in_grid]],
+                  uint tid [[thread_index_in_threadgroup]],
+                  uint sg [[simdgroup_index_in_threadgroup]]) {
+    threadgroup float score[16*64];
+    threadgroup half prob[16*64];
+    threadgroup float acc[16*64];
+    threadgroup float maxes[16], sums[16];
+    uint head=tg/64, qr=(tg%64)*16;
+    uint base=head*1024*64;
+    uint row=tid/8, rl=tid%8;
+    for(uint c=rl;c<64;c+=8) acc[row*64+c]=0.f;
+    if(rl==0){maxes[row]=-INFINITY;sums[row]=0.f;}
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    for(uint kb=0;kb<qr+16;kb+=64){
+        uint mr=(sg/2)*8, nc=(sg%2)*32;
+        simdgroup_float8x8 s0(0.f),s1(0.f),s2(0.f),s3(0.f);
+        for(uint d=0;d<64;d+=8){
+            simdgroup_half8x8 a,b;
+            simdgroup_load(a,q+base+(qr+mr)*64+d,64);
+            simdgroup_load(b,k+base+(kb+nc)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0,a,b,s0);
+            simdgroup_load(b,k+base+(kb+nc+8)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s1,a,b,s1);
+            simdgroup_load(b,k+base+(kb+nc+16)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s2,a,b,s2);
+            simdgroup_load(b,k+base+(kb+nc+24)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s3,a,b,s3);
+        }
+        simdgroup_store(s0,score+mr*64+nc,64);
+        simdgroup_store(s1,score+mr*64+nc+8,64);
+        simdgroup_store(s2,score+mr*64+nc+16,64);
+        simdgroup_store(s3,score+mr*64+nc+24,64);
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        float xs[8]; float mx=-INFINITY;
+        for(uint i=0;i<8;i++){
+            uint c=rl+i*8;
+            xs[i]=kb+c<=qr+row?score[row*64+c]*0.125f:-INFINITY;
+            mx=max(mx,xs[i]);
+        }
+        mx=max(mx,simd_shuffle_xor(mx,1));
+        mx=max(mx,simd_shuffle_xor(mx,2));
+        mx=max(mx,simd_shuffle_xor(mx,4));
+        mx=max(mx,maxes[row]);
+        float alpha=fast::exp(maxes[row]-mx), sum=0.f;
+        for(uint i=0;i<8;i++){
+            xs[i]=fast::exp(xs[i]-mx);
+            sum+=xs[i];
+            prob[row*64+rl+i*8]=half(xs[i]);
+            acc[row*64+rl+i*8]*=alpha;
+        }
+        sum+=simd_shuffle_xor(sum,1);
+        sum+=simd_shuffle_xor(sum,2);
+        sum+=simd_shuffle_xor(sum,4);
+        if(rl==0){sums[row]=sums[row]*alpha+sum;maxes[row]=mx;}
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        simdgroup_float8x8 o0,o1,o2,o3;
+        simdgroup_load(o0,acc+mr*64+nc,64);
+        simdgroup_load(o1,acc+mr*64+nc+8,64);
+        simdgroup_load(o2,acc+mr*64+nc+16,64);
+        simdgroup_load(o3,acc+mr*64+nc+24,64);
+        for(uint d=0;d<64;d+=8){
+            simdgroup_half8x8 a,b;
+            simdgroup_load(a,prob+mr*64+d,64);
+            simdgroup_load(b,v+base+(kb+d)*64+nc,64);
+            simdgroup_multiply_accumulate(o0,a,b,o0);
+            simdgroup_load(b,v+base+(kb+d)*64+nc+8,64);
+            simdgroup_multiply_accumulate(o1,a,b,o1);
+            simdgroup_load(b,v+base+(kb+d)*64+nc+16,64);
+            simdgroup_multiply_accumulate(o2,a,b,o2);
+            simdgroup_load(b,v+base+(kb+d)*64+nc+24,64);
+            simdgroup_multiply_accumulate(o3,a,b,o3);
+        }
+        simdgroup_store(o0,acc+mr*64+nc,64);
+        simdgroup_store(o1,acc+mr*64+nc+8,64);
+        simdgroup_store(o2,acc+mr*64+nc+16,64);
+        simdgroup_store(o3,acc+mr*64+nc+24,64);
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+    }
+    for(uint c=rl;c<64;c+=8) out[base+(qr+row)*64+c]=acc[row*64+c]/sums[row];
+}
 ''')
 
 
 def kernel(q, k, v):
     qh, kh, vh = q.half(), k.half(), v.half()
-    scores = qh @ kh.transpose(-1, -2)
-    p = torch.empty_like(scores)
-    _lib.causal_softmax(scores, p, threads=(q.numel() // 64 * 256), group_size=256)
-    return (p @ vh).float()
+    out = torch.empty_like(q)
+    _lib.flash(qh, kh, vh, out, threads=q.numel()//1024*128, group_size=128)
+    return out

```

</details>

`python bench.py`  
→ run 2 of 10 submitted.  
→ attention: correct (relative error 8.6e-04). reference 63.95 ms, kernel 9.27 ms, speedup 6.90x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.63e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -35,17 +35,17 @@
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup float score[16*64];
-    threadgroup half prob[16*64];
-    threadgroup float acc[16*64];
-    threadgroup float maxes[16], sums[16];
-    uint head=tg/64, qr=(tg%64)*16;
+    threadgroup float score[32*64];
+    threadgroup half prob[32*64];
+    threadgroup float acc[32*64];
+    threadgroup float maxes[32], sums[32];
+    uint head=tg/32, qr=(tg%32)*32;
     uint base=head*1024*64;
     uint row=tid/8, rl=tid%8;
     for(uint c=rl;c<64;c+=8) acc[row*64+c]=0.f;
     if(rl==0){maxes[row]=-INFINITY;sums[row]=0.f;}
     threadgroup_barrier(mem_flags::mem_threadgroup);
-    for(uint kb=0;kb<qr+16;kb+=64){
+    for(uint kb=0;kb<qr+32;kb+=64){
         uint mr=(sg/2)*8, nc=(sg%2)*32;
         simdgroup_float8x8 s0(0.f),s1(0.f),s2(0.f),s3(0.f);
         for(uint d=0;d<64;d+=8){
@@ -118,5 +118,5 @@
 def kernel(q, k, v):
     qh, kh, vh = q.half(), k.half(), v.half()
     out = torch.empty_like(q)
-    _lib.flash(qh, kh, vh, out, threads=q.numel()//1024*128, group_size=128)
+    _lib.flash(qh, kh, vh, out, threads=q.numel()//2048*256, group_size=256)
     return out

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 67.81 ms, kernel 5.51 ms, speedup 12.31x

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text(); s=s.replace(\"score[16*64]\", \"score[32*64]\").replace(\"prob[16*64]\", \"prob[...`
- `python check.py`  
→ OK: relative error 3.12e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -3,6 +3,12 @@
 _lib = torch.mps.compile_shader(r'''
 #include <metal_stdlib>
 using namespace metal;
+kernel void layout(device const float* a, device float* b, uint tid [[thread_index_in_threadgroup]]) {
+    simdgroup_float8x8 m;
+    simdgroup_load(m,a,8);
+    b[tid*2]=m.thread_elements()[0];
+    b[tid*2+1]=m.thread_elements()[1];
+}
 kernel void causal_softmax(device const half* s, device half* p,
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
@@ -37,16 +43,17 @@
                   uint sg [[simdgroup_index_in_threadgroup]]) {
     threadgroup float score[32*64];
     threadgroup half prob[32*64];
-    threadgroup float acc[32*64];
+    threadgroup float scales[32];
     threadgroup float maxes[32], sums[32];
     uint head=tg/32, qr=(tg%32)*32;
     uint base=head*1024*64;
     uint row=tid/8, rl=tid%8;
-    for(uint c=rl;c<64;c+=8) acc[row*64+c]=0.f;
+    uint mr=(sg/2)*8, nc=(sg%2)*32;
+    uint lane=tid%32, orow=mr+(lane%8)/2+(lane/16)*4;
+    simdgroup_float8x8 o0(0.f),o1(0.f),o2(0.f),o3(0.f);
     if(rl==0){maxes[row]=-INFINITY;sums[row]=0.f;}
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for(uint kb=0;kb<qr+32;kb+=64){
-        uint mr=(sg/2)*8, nc=(sg%2)*32;
         simdgroup_float8x8 s0(0.f),s1(0.f),s2(0.f),s3(0.f);
         for(uint d=0;d<64;d+=8){
             simdgroup_half8x8 a,b;
@@ -80,18 +87,20 @@
             xs[i]=fast::exp(xs[i]-mx);
             sum+=xs[i];
             prob[row*64+rl+i*8]=half(xs[i]);
-            acc[row*64+rl+i*8]*=alpha;
+
         }
         sum+=simd_shuffle_xor(sum,1);
         sum+=simd_shuffle_xor(sum,2);
         sum+=simd_shuffle_xor(sum,4);
-        if(rl==0){sums[row]=sums[row]*alpha+sum;maxes[row]=mx;}
+        if(rl==0){sums[row]=sums[row]*alpha+sum;maxes[row]=mx;scales[row]=alpha;}
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 o0,o1,o2,o3;
-        simdgroup_load(o0,acc+mr*64+nc,64);
-        simdgroup_load(o1,acc+mr*64+nc+8,64);
-        simdgroup_load(o2,acc+mr*64+nc+16,64);
-        simdgroup_load(o3,acc+mr*64+nc+24,64);
+        float factor=scales[orow];
+        for(uint e=0;e<2;e++){
+            o0.thread_elements()[e]*=factor;
+            o1.thread_elements()[e]*=factor;
+            o2.thread_elements()[e]*=factor;
+            o3.thread_elements()[e]*=factor;
+        }
         for(uint d=0;d<64;d+=8){
             simdgroup_half8x8 a,b;
             simdgroup_load(a,prob+mr*64+d,64);
@@ -104,13 +113,18 @@
             simdgroup_load(b,v+base+(kb+d)*64+nc+24,64);
             simdgroup_multiply_accumulate(o3,a,b,o3);
         }
-        simdgroup_store(o0,acc+mr*64+nc,64);
-        simdgroup_store(o1,acc+mr*64+nc+8,64);
-        simdgroup_store(o2,acc+mr*64+nc+16,64);
-        simdgroup_store(o3,acc+mr*64+nc+24,64);
-        threadgroup_barrier(mem_flags::mem_threadgroup);
     }
-    for(uint c=rl;c<64;c+=8) out[base+(qr+row)*64+c]=acc[row*64+c]/sums[row];
+    float inv=1.f/sums[orow];
+    for(uint e=0;e<2;e++){
+        o0.thread_elements()[e]*=inv;
+        o1.thread_elements()[e]*=inv;
+        o2.thread_elements()[e]*=inv;
+        o3.thread_elements()[e]*=inv;
+    }
+    simdgroup_store(o0,out+base+(qr+mr)*64+nc,64);
+    simdgroup_store(o1,out+base+(qr+mr)*64+nc+8,64);
+    simdgroup_store(o2,out+base+(qr+mr)*64+nc+16,64);
+    simdgroup_store(o3,out+base+(qr+mr)*64+nc+24,64);
 }
 ''')
 

```

</details>

`python bench.py`  
→         [60., 61.],  
→         [62., 63.]])

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\"); s=p.read_text(); s=s.replace(\"threadgroup float acc[32*64];\", \"threadgroup float scales[32];\"...`
- `python check.py`  
→ OK: relative error 5.73e-04 (allowed 2e-03)
> The fused shader runs in 5.51 ms, a 12.31× speedup. Doubling the query tile was slightly slower. I’m testing whether keeping the output accumulators in registers improves it by reducing shared-memory traffic and synchronization.

### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -3,128 +3,83 @@
 _lib = torch.mps.compile_shader(r'''
 #include <metal_stdlib>
 using namespace metal;
-kernel void layout(device const float* a, device float* b, uint tid [[thread_index_in_threadgroup]]) {
-    simdgroup_float8x8 m;
-    simdgroup_load(m,a,8);
-    b[tid*2]=m.thread_elements()[0];
-    b[tid*2+1]=m.thread_elements()[1];
-}
-kernel void causal_softmax(device const half* s, device half* p,
-                           uint row [[threadgroup_position_in_grid]],
-                           uint tid [[thread_index_in_threadgroup]],
-                           uint lane [[thread_index_in_simdgroup]],
-                           uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup float scratch[8];
-    uint last = row % 1024;
-    float x[4];
-    float mx = -INFINITY;
-    for (uint i=0;i<4;i++) {
-        uint col=tid+i*256;
-        x[i] = col<=last ? float(s[row*1024+col])*0.125f : -INFINITY;
-        mx=max(mx,x[i]);
-    }
-    mx=simd_max(mx);
-    if(lane==0) scratch[sg]=mx;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    mx=simd_max(lane<8?scratch[lane]:-INFINITY);
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    float sum=0;
-    for(uint i=0;i<4;i++){x[i]=fast::exp(x[i]-mx);sum+=x[i];}
-    sum=simd_sum(sum);
-    if(lane==0) scratch[sg]=sum;
-    threadgroup_barrier(mem_flags::mem_threadgroup);
-    sum=simd_sum(lane<8?scratch[lane]:0.f);
-    for(uint i=0;i<4;i++) p[row*1024+tid+i*256]=half(x[i]/sum);
-}
 kernel void flash(device const half* q, device const half* k, device const half* v,
                   device float* out,
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup float score[32*64];
     threadgroup half prob[32*64];
-    threadgroup float scales[32];
-    threadgroup float maxes[32], sums[32];
-    uint head=tg/32, qr=(tg%32)*32;
-    uint base=head*1024*64;
-    uint row=tid/8, rl=tid%8;
-    uint mr=(sg/2)*8, nc=(sg%2)*32;
-    uint lane=tid%32, orow=mr+(lane%8)/2+(lane/16)*4;
-    simdgroup_float8x8 o0(0.f),o1(0.f),o2(0.f),o3(0.f);
-    if(rl==0){maxes[row]=-INFINITY;sums[row]=0.f;}
-    threadgroup_barrier(mem_flags::mem_threadgroup);
+    threadgroup float group_max[32*2], group_sum[32*2];
+    uint head=tg/32, qr=(tg%32)*32, base=head*1024*64;
+    uint mr=(sg/2)*8, nc=(sg%2)*32, lane=tid%32;
+    uint row=mr+(lane%8)/2+(lane/16)*4;
+    uint col=(lane%2)*2+((lane/8)%2)*4;
+    bool owner=col==0;
+    simdgroup_float8x8 oo[4];
+    for(uint i=0;i<4;i++) oo[i]=simdgroup_float8x8(0.f);
+    float m=-INFINITY, l=0.f;
     for(uint kb=0;kb<qr+32;kb+=64){
-        simdgroup_float8x8 s0(0.f),s1(0.f),s2(0.f),s3(0.f);
+        simdgroup_float8x8 ss[4];
+        for(uint i=0;i<4;i++) ss[i]=simdgroup_float8x8(0.f);
         for(uint d=0;d<64;d+=8){
             simdgroup_half8x8 a,b;
             simdgroup_load(a,q+base+(qr+mr)*64+d,64);
-            simdgroup_load(b,k+base+(kb+nc)*64+d,64,ulong2(0),true);
-            simdgroup_multiply_accumulate(s0,a,b,s0);
-            simdgroup_load(b,k+base+(kb+nc+8)*64+d,64,ulong2(0),true);
-            simdgroup_multiply_accumulate(s1,a,b,s1);
-            simdgroup_load(b,k+base+(kb+nc+16)*64+d,64,ulong2(0),true);
-            simdgroup_multiply_accumulate(s2,a,b,s2);
-            simdgroup_load(b,k+base+(kb+nc+24)*64+d,64,ulong2(0),true);
-            simdgroup_multiply_accumulate(s3,a,b,s3);
+            #pragma unroll
+            for(uint i=0;i<4;i++){
+                simdgroup_load(b,k+base+(kb+nc+i*8)*64+d,64,ulong2(0),true);
+                simdgroup_multiply_accumulate(ss[i],a,b,ss[i]);
+            }
         }
-        simdgroup_store(s0,score+mr*64+nc,64);
-        simdgroup_store(s1,score+mr*64+nc+8,64);
-        simdgroup_store(s2,score+mr*64+nc+16,64);
-        simdgroup_store(s3,score+mr*64+nc+24,64);
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        float xs[8]; float mx=-INFINITY;
-        for(uint i=0;i<8;i++){
-            uint c=rl+i*8;
-            xs[i]=kb+c<=qr+row?score[row*64+c]*0.125f:-INFINITY;
-            mx=max(mx,xs[i]);
+        float mx=-INFINITY;
+        #pragma unroll
+        for(uint i=0;i<4;i++){
+            #pragma unroll
+            for(uint e=0;e<2;e++){
+                float x=kb+nc+i*8+col+e<=qr+row?ss[i].thread_elements()[e]*0.125f:-INFINITY;
+                ss[i].thread_elements()[e]=x;
+                mx=max(mx,x);
+            }
         }
         mx=max(mx,simd_shuffle_xor(mx,1));
-        mx=max(mx,simd_shuffle_xor(mx,2));
-        mx=max(mx,simd_shuffle_xor(mx,4));
-        mx=max(mx,maxes[row]);
-        float alpha=fast::exp(maxes[row]-mx), sum=0.f;
-        for(uint i=0;i<8;i++){
-            xs[i]=fast::exp(xs[i]-mx);
-            sum+=xs[i];
-            prob[row*64+rl+i*8]=half(xs[i]);
-
+        mx=max(mx,simd_shuffle_xor(mx,8));
+        if(owner) group_max[row*2+sg%2]=mx;
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        mx=max(m,max(group_max[row*2],group_max[row*2+1]));
+        float alpha=fast::exp(m-mx), sum=0.f;
+        #pragma unroll
+        for(uint i=0;i<4;i++){
+            simdgroup_half8x8 pp;
+            #pragma unroll
+            for(uint e=0;e<2;e++){
+                float x=fast::exp(ss[i].thread_elements()[e]-mx);
+                pp.thread_elements()[e]=half(x);
+                sum+=x;
+                oo[i].thread_elements()[e]*=alpha;
+            }
+            simdgroup_store(pp,prob+mr*64+nc+i*8,64);
         }
         sum+=simd_shuffle_xor(sum,1);
-        sum+=simd_shuffle_xor(sum,2);
-        sum+=simd_shuffle_xor(sum,4);
-        if(rl==0){sums[row]=sums[row]*alpha+sum;maxes[row]=mx;scales[row]=alpha;}
+        sum+=simd_shuffle_xor(sum,8);
+        if(owner) group_sum[row*2+sg%2]=sum;
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        float factor=scales[orow];
-        for(uint e=0;e<2;e++){
-            o0.thread_elements()[e]*=factor;
-            o1.thread_elements()[e]*=factor;
-            o2.thread_elements()[e]*=factor;
-            o3.thread_elements()[e]*=factor;
-        }
+        l=l*alpha+group_sum[row*2]+group_sum[row*2+1];
+        m=mx;
         for(uint d=0;d<64;d+=8){
             simdgroup_half8x8 a,b;
             simdgroup_load(a,prob+mr*64+d,64);
-            simdgroup_load(b,v+base+(kb+d)*64+nc,64);
-            simdgroup_multiply_accumulate(o0,a,b,o0);
-            simdgroup_load(b,v+base+(kb+d)*64+nc+8,64);
-            simdgroup_multiply_accumulate(o1,a,b,o1);
-            simdgroup_load(b,v+base+(kb+d)*64+nc+16,64);
-            simdgroup_multiply_accumulate(o2,a,b,o2);
-            simdgroup_load(b,v+base+(kb+d)*64+nc+24,64);
-            simdgroup_multiply_accumulate(o3,a,b,o3);
+            #pragma unroll
+            for(uint i=0;i<4;i++){
+                simdgroup_load(b,v+base+(kb+d)*64+nc+i*8,64);
+                simdgroup_multiply_accumulate(oo[i],a,b,oo[i]);
+            }
         }
     }
-    float inv=1.f/sums[orow];
-    for(uint e=0;e<2;e++){
-        o0.thread_elements()[e]*=inv;
-        o1.thread_elements()[e]*=inv;
-        o2.thread_elements()[e]*=inv;
-        o3.thread_elements()[e]*=inv;
+    #pragma unroll
+    for(uint i=0;i<4;i++){
+        oo[i].thread_elements()[0]/=l;
+        oo[i].thread_elements()[1]/=l;
+        simdgroup_store(oo[i],out+base+(qr+mr)*64+nc+i*8,64);
     }
-    simdgroup_store(o0,out+base+(qr+mr)*64+nc,64);
-    simdgroup_store(o1,out+base+(qr+mr)*64+nc+8,64);
-    simdgroup_store(o2,out+base+(qr+mr)*64+nc+16,64);
-    simdgroup_store(o3,out+base+(qr+mr)*64+nc+24,64);
 }
 ''')
 

```

</details>

`python bench.py`  
→ run 5 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 67.91 ms, kernel 5.46 ms, speedup 12.45x

- `cat kernel.py`
- `python -c 'from pathlib import Path; Path('\\''kernel.py'\\'').write_text(\"import torch\\n\\n_lib = torch.mps.compile_shader(r'\\'''\\'''\\''\\n#include <me...`
- `python check.py`  
→ OK: relative error 4.44e-04 (allowed 2e-03)
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -8,25 +8,23 @@
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup half prob[32*64];
-    threadgroup float group_max[32*2], group_sum[32*2];
-    uint head=tg/32, qr=(tg%32)*32, base=head*1024*64;
-    uint mr=(sg/2)*8, nc=(sg%2)*32, lane=tid%32;
-    uint row=mr+(lane%8)/2+(lane/16)*4;
+    threadgroup half prob[32*32];
+    uint head=tg/32, qr=(tg%32)*32+sg*8, base=head*1024*64;
+    uint lane=tid%32;
+    uint row=(lane%8)/2+(lane/16)*4;
     uint col=(lane%2)*2+((lane/8)%2)*4;
-    bool owner=col==0;
-    simdgroup_float8x8 oo[4];
-    for(uint i=0;i<4;i++) oo[i]=simdgroup_float8x8(0.f);
+    simdgroup_float8x8 oo[8];
+    for(uint i=0;i<8;i++) oo[i]=simdgroup_float8x8(0.f);
     float m=-INFINITY, l=0.f;
-    for(uint kb=0;kb<qr+32;kb+=64){
+    for(uint kb=0;kb<qr+8;kb+=32){
         simdgroup_float8x8 ss[4];
         for(uint i=0;i<4;i++) ss[i]=simdgroup_float8x8(0.f);
         for(uint d=0;d<64;d+=8){
             simdgroup_half8x8 a,b;
-            simdgroup_load(a,q+base+(qr+mr)*64+d,64);
+            simdgroup_load(a,q+base+qr*64+d,64);
             #pragma unroll
             for(uint i=0;i<4;i++){
-                simdgroup_load(b,k+base+(kb+nc+i*8)*64+d,64,ulong2(0),true);
+                simdgroup_load(b,k+base+(kb+i*8)*64+d,64,ulong2(0),true);
                 simdgroup_multiply_accumulate(ss[i],a,b,ss[i]);
             }
         }
@@ -35,16 +33,14 @@
         for(uint i=0;i<4;i++){
             #pragma unroll
             for(uint e=0;e<2;e++){
-                float x=kb+nc+i*8+col+e<=qr+row?ss[i].thread_elements()[e]*0.125f:-INFINITY;
+                float x=kb+i*8+col+e<=qr+row?ss[i].thread_elements()[e]*0.125f:-INFINITY;
                 ss[i].thread_elements()[e]=x;
                 mx=max(mx,x);
             }
         }
         mx=max(mx,simd_shuffle_xor(mx,1));
         mx=max(mx,simd_shuffle_xor(mx,8));
-        if(owner) group_max[row*2+sg%2]=mx;
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        mx=max(m,max(group_max[row*2],group_max[row*2+1]));
+        mx=max(m,mx);
         float alpha=fast::exp(m-mx), sum=0.f;
         #pragma unroll
         for(uint i=0;i<4;i++){
@@ -54,31 +50,34 @@
                 float x=fast::exp(ss[i].thread_elements()[e]-mx);
                 pp.thread_elements()[e]=half(x);
                 sum+=x;
-                oo[i].thread_elements()[e]*=alpha;
             }
-            simdgroup_store(pp,prob+mr*64+nc+i*8,64);
+            simdgroup_store(pp,prob+sg*256+i*8,32);
         }
         sum+=simd_shuffle_xor(sum,1);
         sum+=simd_shuffle_xor(sum,8);
-        if(owner) group_sum[row*2+sg%2]=sum;
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        l=l*alpha+group_sum[row*2]+group_sum[row*2+1];
+        l=l*alpha+sum;
         m=mx;
-        for(uint d=0;d<64;d+=8){
+        #pragma unroll
+        for(uint i=0;i<8;i++){
+            oo[i].thread_elements()[0]*=alpha;
+            oo[i].thread_elements()[1]*=alpha;
+        }
+        simdgroup_barrier(mem_flags::mem_threadgroup);
+        for(uint d=0;d<32;d+=8){
             simdgroup_half8x8 a,b;
-            simdgroup_load(a,prob+mr*64+d,64);
+            simdgroup_load(a,prob+sg*256+d,32);
             #pragma unroll
-            for(uint i=0;i<4;i++){
-                simdgroup_load(b,v+base+(kb+d)*64+nc+i*8,64);
+            for(uint i=0;i<8;i++){
+                simdgroup_load(b,v+base+(kb+d)*64+i*8,64);
                 simdgroup_multiply_accumulate(oo[i],a,b,oo[i]);
             }
         }
     }
     #pragma unroll
-    for(uint i=0;i<4;i++){
+    for(uint i=0;i<8;i++){
         oo[i].thread_elements()[0]/=l;
         oo[i].thread_elements()[1]/=l;
-        simdgroup_store(oo[i],out+base+(qr+mr)*64+nc+i*8,64);
+        simdgroup_store(oo[i],out+base+qr*64+i*8,64);
     }
 }
 ''')
@@ -87,5 +86,5 @@
 def kernel(q, k, v):
     qh, kh, vh = q.half(), k.half(), v.half()
     out = torch.empty_like(q)
-    _lib.flash(qh, kh, vh, out, threads=q.numel()//2048*256, group_size=256)
+    _lib.flash(qh, kh, vh, out, threads=q.numel()//2048*128, group_size=128)
     return out

```

</details>

`python bench.py`  


- `python check.py`  
→ OK: relative error 3.16e-04 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -8,77 +8,248 @@
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup half prob[32*32];
-    uint head=tg/32, qr=(tg%32)*32+sg*8, base=head*1024*64;
-    uint lane=tid%32;
-    uint row=(lane%8)/2+(lane/16)*4;
-    uint col=(lane%2)*2+((lane/8)%2)*4;
-    simdgroup_float8x8 oo[8];
-    for(uint i=0;i<8;i++) oo[i]=simdgroup_float8x8(0.f);
-    float m=-INFINITY, l=0.f;
-    for(uint kb=0;kb<qr+8;kb+=32){
-        simdgroup_float8x8 ss[4];
-        for(uint i=0;i<4;i++) ss[i]=simdgroup_float8x8(0.f);
+    threadgroup half prob[2048];
+    uint head=tg/16, qr=(tg%16)*64+sg*16, base=head*1024*64;
+    uint lane=tid%32, row=(lane%8)/2+(lane/16)*4, col=(lane%2)*2+((lane/8)%2)*4;
+    float m0=-INFINITY, l0=0.f;
+    simdgroup_float8x8 o0_0(0.f);
+    simdgroup_float8x8 o0_1(0.f);
+    simdgroup_float8x8 o0_2(0.f);
+    simdgroup_float8x8 o0_3(0.f);
+    simdgroup_float8x8 o0_4(0.f);
+    simdgroup_float8x8 o0_5(0.f);
+    simdgroup_float8x8 o0_6(0.f);
+    simdgroup_float8x8 o0_7(0.f);
+    float m1=-INFINITY, l1=0.f;
+    simdgroup_float8x8 o1_0(0.f);
+    simdgroup_float8x8 o1_1(0.f);
+    simdgroup_float8x8 o1_2(0.f);
+    simdgroup_float8x8 o1_3(0.f);
+    simdgroup_float8x8 o1_4(0.f);
+    simdgroup_float8x8 o1_5(0.f);
+    simdgroup_float8x8 o1_6(0.f);
+    simdgroup_float8x8 o1_7(0.f);
+    for(uint kb=0;kb<qr+16;kb+=32){
+        simdgroup_float8x8 s0_0(0.f);
+        simdgroup_float8x8 s0_1(0.f);
+        simdgroup_float8x8 s0_2(0.f);
+        simdgroup_float8x8 s0_3(0.f);
+        simdgroup_float8x8 s1_0(0.f);
+        simdgroup_float8x8 s1_1(0.f);
+        simdgroup_float8x8 s1_2(0.f);
+        simdgroup_float8x8 s1_3(0.f);
         for(uint d=0;d<64;d+=8){
-            simdgroup_half8x8 a,b;
-            simdgroup_load(a,q+base+qr*64+d,64);
-            #pragma unroll
-            for(uint i=0;i<4;i++){
-                simdgroup_load(b,k+base+(kb+i*8)*64+d,64,ulong2(0),true);
-                simdgroup_multiply_accumulate(ss[i],a,b,ss[i]);
-            }
+            simdgroup_half8x8 b;
+            simdgroup_half8x8 a0;
+            simdgroup_load(a0,q+base+(qr+0)*64+d,64);
+            simdgroup_half8x8 a1;
+            simdgroup_load(a1,q+base+(qr+8)*64+d,64);
+            simdgroup_load(b,k+base+(kb+0)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0_0,a0,b,s0_0);
+            simdgroup_multiply_accumulate(s1_0,a1,b,s1_0);
+            simdgroup_load(b,k+base+(kb+8)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0_1,a0,b,s0_1);
+            simdgroup_multiply_accumulate(s1_1,a1,b,s1_1);
+            simdgroup_load(b,k+base+(kb+16)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0_2,a0,b,s0_2);
+            simdgroup_multiply_accumulate(s1_2,a1,b,s1_2);
+            simdgroup_load(b,k+base+(kb+24)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0_3,a0,b,s0_3);
+            simdgroup_multiply_accumulate(s1_3,a1,b,s1_3);
         }
-        float mx=-INFINITY;
-        #pragma unroll
-        for(uint i=0;i<4;i++){
-            #pragma unroll
-            for(uint e=0;e<2;e++){
-                float x=kb+i*8+col+e<=qr+row?ss[i].thread_elements()[e]*0.125f:-INFINITY;
-                ss[i].thread_elements()[e]=x;
-                mx=max(mx,x);
-            }
-        }
-        mx=max(mx,simd_shuffle_xor(mx,1));
-        mx=max(mx,simd_shuffle_xor(mx,8));
-        mx=max(m,mx);
-        float alpha=fast::exp(m-mx), sum=0.f;
-        #pragma unroll
-        for(uint i=0;i<4;i++){
-            simdgroup_half8x8 pp;
-            #pragma unroll
-            for(uint e=0;e<2;e++){
-                float x=fast::exp(ss[i].thread_elements()[e]-mx);
-                pp.thread_elements()[e]=half(x);
-                sum+=x;
-            }
-            simdgroup_store(pp,prob+sg*256+i*8,32);
-        }
-        sum+=simd_shuffle_xor(sum,1);
-        sum+=simd_shuffle_xor(sum,8);
-        l=l*alpha+sum;
-        m=mx;
-        #pragma unroll
-        for(uint i=0;i<8;i++){
-            oo[i].thread_elements()[0]*=alpha;
-            oo[i].thread_elements()[1]*=alpha;
-        }
+        float mx0=-INFINITY;
+        s0_0.thread_elements()[0]=kb+0+col+0<=qr+0+row?s0_0.thread_elements()[0]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_0.thread_elements()[0]);
+        s0_0.thread_elements()[1]=kb+0+col+1<=qr+0+row?s0_0.thread_elements()[1]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_0.thread_elements()[1]);
+        s0_1.thread_elements()[0]=kb+8+col+0<=qr+0+row?s0_1.thread_elements()[0]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_1.thread_elements()[0]);
+        s0_1.thread_elements()[1]=kb+8+col+1<=qr+0+row?s0_1.thread_elements()[1]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_1.thread_elements()[1]);
+        s0_2.thread_elements()[0]=kb+16+col+0<=qr+0+row?s0_2.thread_elements()[0]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_2.thread_elements()[0]);
+        s0_2.thread_elements()[1]=kb+16+col+1<=qr+0+row?s0_2.thread_elements()[1]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_2.thread_elements()[1]);
+        s0_3.thread_elements()[0]=kb+24+col+0<=qr+0+row?s0_3.thread_elements()[0]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_3.thread_elements()[0]);
+        s0_3.thread_elements()[1]=kb+24+col+1<=qr+0+row?s0_3.thread_elements()[1]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_3.thread_elements()[1]);
+        mx0=max(mx0,simd_shuffle_xor(mx0,1));
+        mx0=max(mx0,simd_shuffle_xor(mx0,8));
+        mx0=max(m0,mx0);
+        float alpha0=fast::exp(m0-mx0), sum0=0.f;
+        simdgroup_half8x8 p0_0;
+        {float x=fast::exp(s0_0.thread_elements()[0]-mx0);p0_0.thread_elements()[0]=half(x);sum0+=x;}
+        {float x=fast::exp(s0_0.thread_elements()[1]-mx0);p0_0.thread_elements()[1]=half(x);sum0+=x;}
+        simdgroup_store(p0_0,prob+sg*512+0,32);
+        simdgroup_half8x8 p0_1;
+        {float x=fast::exp(s0_1.thread_elements()[0]-mx0);p0_1.thread_elements()[0]=half(x);sum0+=x;}
+        {float x=fast::exp(s0_1.thread_elements()[1]-mx0);p0_1.thread_elements()[1]=half(x);sum0+=x;}
+        simdgroup_store(p0_1,prob+sg*512+8,32);
+        simdgroup_half8x8 p0_2;
+        {float x=fast::exp(s0_2.thread_elements()[0]-mx0);p0_2.thread_elements()[0]=half(x);sum0+=x;}
+        {float x=fast::exp(s0_2.thread_elements()[1]-mx0);p0_2.thread_elements()[1]=half(x);sum0+=x;}
+        simdgroup_store(p0_2,prob+sg*512+16,32);
+        simdgroup_half8x8 p0_3;
+        {float x=fast::exp(s0_3.thread_elements()[0]-mx0);p0_3.thread_elements()[0]=half(x);sum0+=x;}
+        {float x=fast::exp(s0_3.thread_elements()[1]-mx0);p0_3.thread_elements()[1]=half(x);sum0+=x;}
+        simdgroup_store(p0_3,prob+sg*512+24,32);
+        sum0+=simd_shuffle_xor(sum0,1);
+        sum0+=simd_shuffle_xor(sum0,8);
+        l0=l0*alpha0+sum0; m0=mx0;
+        o0_0.thread_elements()[0]*=alpha0;
+        o0_0.thread_elements()[1]*=alpha0;
+        o0_1.thread_elements()[0]*=alpha0;
+        o0_1.thread_elements()[1]*=alpha0;
+        o0_2.thread_elements()[0]*=alpha0;
+        o0_2.thread_elements()[1]*=alpha0;
+        o0_3.thread_elements()[0]*=alpha0;
+        o0_3.thread_elements()[1]*=alpha0;
+        o0_4.thread_elements()[0]*=alpha0;
+        o0_4.thread_elements()[1]*=alpha0;
+        o0_5.thread_elements()[0]*=alpha0;
+        o0_5.thread_elements()[1]*=alpha0;
+        o0_6.thread_elements()[0]*=alpha0;
+        o0_6.thread_elements()[1]*=alpha0;
+        o0_7.thread_elements()[0]*=alpha0;
+        o0_7.thread_elements()[1]*=alpha0;
+        float mx1=-INFINITY;
+        s1_0.thread_elements()[0]=kb+0+col+0<=qr+8+row?s1_0.thread_elements()[0]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_0.thread_elements()[0]);
+        s1_0.thread_elements()[1]=kb+0+col+1<=qr+8+row?s1_0.thread_elements()[1]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_0.thread_elements()[1]);
+        s1_1.thread_elements()[0]=kb+8+col+0<=qr+8+row?s1_1.thread_elements()[0]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_1.thread_elements()[0]);
+        s1_1.thread_elements()[1]=kb+8+col+1<=qr+8+row?s1_1.thread_elements()[1]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_1.thread_elements()[1]);
+        s1_2.thread_elements()[0]=kb+16+col+0<=qr+8+row?s1_2.thread_elements()[0]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_2.thread_elements()[0]);
+        s1_2.thread_elements()[1]=kb+16+col+1<=qr+8+row?s1_2.thread_elements()[1]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_2.thread_elements()[1]);
+        s1_3.thread_elements()[0]=kb+24+col+0<=qr+8+row?s1_3.thread_elements()[0]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_3.thread_elements()[0]);
+        s1_3.thread_elements()[1]=kb+24+col+1<=qr+8+row?s1_3.thread_elements()[1]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_3.thread_elements()[1]);
+        mx1=max(mx1,simd_shuffle_xor(mx1,1));
+        mx1=max(mx1,simd_shuffle_xor(mx1,8));
+        mx1=max(m1,mx1);
+        float alpha1=fast::exp(m1-mx1), sum1=0.f;
+        simdgroup_half8x8 p1_0;
+        {float x=fast::exp(s1_0.thread_elements()[0]-mx1);p1_0.thread_elements()[0]=half(x);sum1+=x;}
+        {float x=fast::exp(s1_0.thread_elements()[1]-mx1);p1_0.thread_elements()[1]=half(x);sum1+=x;}
+        simdgroup_store(p1_0,prob+sg*512+256,32);
+        simdgroup_half8x8 p1_1;
+        {float x=fast::exp(s1_1.thread_elements()[0]-mx1);p1_1.thread_elements()[0]=half(x);sum1+=x;}
+        {float x=fast::exp(s1_1.thread_elements()[1]-mx1);p1_1.thread_elements()[1]=half(x);sum1+=x;}
+        simdgroup_store(p1_1,prob+sg*512+264,32);
+        simdgroup_half8x8 p1_2;
+        {float x=fast::exp(s1_2.thread_elements()[0]-mx1);p1_2.thread_elements()[0]=half(x);sum1+=x;}
+        {float x=fast::exp(s1_2.thread_elements()[1]-mx1);p1_2.thread_elements()[1]=half(x);sum1+=x;}
+        simdgroup_store(p1_2,prob+sg*512+272,32);
+        simdgroup_half8x8 p1_3;
+        {float x=fast::exp(s1_3.thread_elements()[0]-mx1);p1_3.thread_elements()[0]=half(x);sum1+=x;}
+        {float x=fast::exp(s1_3.thread_elements()[1]-mx1);p1_3.thread_elements()[1]=half(x);sum1+=x;}
+        simdgroup_store(p1_3,prob+sg*512+280,32);
+        sum1+=simd_shuffle_xor(sum1,1);
+        sum1+=simd_shuffle_xor(sum1,8);
+        l1=l1*alpha1+sum1; m1=mx1;
+        o1_0.thread_elements()[0]*=alpha1;
+        o1_0.thread_elements()[1]*=alpha1;
+        o1_1.thread_elements()[0]*=alpha1;
+        o1_1.thread_elements()[1]*=alpha1;
+        o1_2.thread_elements()[0]*=alpha1;
+        o1_2.thread_elements()[1]*=alpha1;
+        o1_3.thread_elements()[0]*=alpha1;
+        o1_3.thread_elements()[1]*=alpha1;
+        o1_4.thread_elements()[0]*=alpha1;
+        o1_4.thread_elements()[1]*=alpha1;
+        o1_5.thread_elements()[0]*=alpha1;
+        o1_5.thread_elements()[1]*=alpha1;
+        o1_6.thread_elements()[0]*=alpha1;
+        o1_6.thread_elements()[1]*=alpha1;
+        o1_7.thread_elements()[0]*=alpha1;
+        o1_7.thread_elements()[1]*=alpha1;
         simdgroup_barrier(mem_flags::mem_threadgroup);
         for(uint d=0;d<32;d+=8){
-            simdgroup_half8x8 a,b;
-            simdgroup_load(a,prob+sg*256+d,32);
-            #pragma unroll
-            for(uint i=0;i<8;i++){
-                simdgroup_load(b,v+base+(kb+d)*64+i*8,64);
-                simdgroup_multiply_accumulate(oo[i],a,b,oo[i]);
-            }
+            simdgroup_half8x8 b;
+            simdgroup_half8x8 a0;
+            simdgroup_load(a0,prob+sg*512+0+d,32);
+            simdgroup_half8x8 a1;
+            simdgroup_load(a1,prob+sg*512+256+d,32);
+            simdgroup_load(b,v+base+(kb+d)*64+0,64);
+            simdgroup_multiply_accumulate(o0_0,a0,b,o0_0);
+            simdgroup_multiply_accumulate(o1_0,a1,b,o1_0);
+            simdgroup_load(b,v+base+(kb+d)*64+8,64);
+            simdgroup_multiply_accumulate(o0_1,a0,b,o0_1);
+            simdgroup_multiply_accumulate(o1_1,a1,b,o1_1);
+            simdgroup_load(b,v+base+(kb+d)*64+16,64);
+            simdgroup_multiply_accumulate(o0_2,a0,b,o0_2);
+            simdgroup_multiply_accumulate(o1_2,a1,b,o1_2);
+            simdgroup_load(b,v+base+(kb+d)*64+24,64);
+            simdgroup_multiply_accumulate(o0_3,a0,b,o0_3);
+            simdgroup_multiply_accumulate(o1_3,a1,b,o1_3);
+            simdgroup_load(b,v+base+(kb+d)*64+32,64);
+            simdgroup_multiply_accumulate(o0_4,a0,b,o0_4);
+            simdgroup_multiply_accumulate(o1_4,a1,b,o1_4);
+            simdgroup_load(b,v+base+(kb+d)*64+40,64);
+            simdgroup_multiply_accumulate(o0_5,a0,b,o0_5);
+            simdgroup_multiply_accumulate(o1_5,a1,b,o1_5);
+            simdgroup_load(b,v+base+(kb+d)*64+48,64);
+            simdgroup_multiply_accumulate(o0_6,a0,b,o0_6);
+            simdgroup_multiply_accumulate(o1_6,a1,b,o1_6);
+            simdgroup_load(b,v+base+(kb+d)*64+56,64);
+            simdgroup_multiply_accumulate(o0_7,a0,b,o0_7);
+            simdgroup_multiply_accumulate(o1_7,a1,b,o1_7);
         }
     }
-    #pragma unroll
-    for(uint i=0;i<8;i++){
-        oo[i].thread_elements()[0]/=l;
-        oo[i].thread_elements()[1]/=l;
-        simdgroup_store(oo[i],out+base+qr*64+i*8,64);
-    }
+    o0_0.thread_elements()[0]/=l0;
+    o0_0.thread_elements()[1]/=l0;
+    simdgroup_store(o0_0,out+base+(qr+0)*64+0,64);
+    o0_1.thread_elements()[0]/=l0;
+    o0_1.thread_elements()[1]/=l0;
+    simdgroup_store(o0_1,out+base+(qr+0)*64+8,64);
+    o0_2.thread_elements()[0]/=l0;
+    o0_2.thread_elements()[1]/=l0;
+    simdgroup_store(o0_2,out+base+(qr+0)*64+16,64);
+    o0_3.thread_elements()[0]/=l0;
+    o0_3.thread_elements()[1]/=l0;
+    simdgroup_store(o0_3,out+base+(qr+0)*64+24,64);
+    o0_4.thread_elements()[0]/=l0;
+    o0_4.thread_elements()[1]/=l0;
+    simdgroup_store(o0_4,out+base+(qr+0)*64+32,64);
+    o0_5.thread_elements()[0]/=l0;
+    o0_5.thread_elements()[1]/=l0;
+    simdgroup_store(o0_5,out+base+(qr+0)*64+40,64);
+    o0_6.thread_elements()[0]/=l0;
+    o0_6.thread_elements()[1]/=l0;
+    simdgroup_store(o0_6,out+base+(qr+0)*64+48,64);
+    o0_7.thread_elements()[0]/=l0;
+    o0_7.thread_elements()[1]/=l0;
+    simdgroup_store(o0_7,out+base+(qr+0)*64+56,64);
+    o1_0.thread_elements()[0]/=l1;
+    o1_0.thread_elements()[1]/=l1;
+    simdgroup_store(o1_0,out+base+(qr+8)*64+0,64);
+    o1_1.thread_elements()[0]/=l1;
+    o1_1.thread_elements()[1]/=l1;
+    simdgroup_store(o1_1,out+base+(qr+8)*64+8,64);
+    o1_2.thread_elements()[0]/=l1;
+    o1_2.thread_elements()[1]/=l1;
+    simdgroup_store(o1_2,out+base+(qr+8)*64+16,64);
+    o1_3.thread_elements()[0]/=l1;
+    o1_3.thread_elements()[1]/=l1;
+    simdgroup_store(o1_3,out+base+(qr+8)*64+24,64);
+    o1_4.thread_elements()[0]/=l1;
+    o1_4.thread_elements()[1]/=l1;
+    simdgroup_store(o1_4,out+base+(qr+8)*64+32,64);
+    o1_5.thread_elements()[0]/=l1;
+    o1_5.thread_elements()[1]/=l1;
+    simdgroup_store(o1_5,out+base+(qr+8)*64+40,64);
+    o1_6.thread_elements()[0]/=l1;
+    o1_6.thread_elements()[1]/=l1;
+    simdgroup_store(o1_6,out+base+(qr+8)*64+48,64);
+    o1_7.thread_elements()[0]/=l1;
+    o1_7.thread_elements()[1]/=l1;
+    simdgroup_store(o1_7,out+base+(qr+8)*64+56,64);
 }
 ''')
 
@@ -86,5 +257,5 @@
 def kernel(q, k, v):
     qh, kh, vh = q.half(), k.half(), v.half()
     out = torch.empty_like(q)
-    _lib.flash(qh, kh, vh, out, threads=q.numel()//2048*128, group_size=128)
+    _lib.flash(qh, kh, vh, out, threads=q.numel()//4096*128, group_size=128)
     return out

```

</details>

`python bench.py`  
→ run 6 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 73.02 ms, kernel 9.08 ms, speedup 8.05x

- `python -c 'from pathlib import Path; Path('\\''kernel.py'\\'').write_text(\"import torch\\n\\n_lib = torch.mps.compile_shader(r'\\'''\\'''\\''\\n#include <me...`
- `python check.py`  
→ OK: relative error 3.00e-04 (allowed 2e-03)
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -8,248 +8,97 @@
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup half prob[2048];
-    uint head=tg/16, qr=(tg%16)*64+sg*16, base=head*1024*64;
-    uint lane=tid%32, row=(lane%8)/2+(lane/16)*4, col=(lane%2)*2+((lane/8)%2)*4;
-    float m0=-INFINITY, l0=0.f;
-    simdgroup_float8x8 o0_0(0.f);
-    simdgroup_float8x8 o0_1(0.f);
-    simdgroup_float8x8 o0_2(0.f);
-    simdgroup_float8x8 o0_3(0.f);
-    simdgroup_float8x8 o0_4(0.f);
-    simdgroup_float8x8 o0_5(0.f);
-    simdgroup_float8x8 o0_6(0.f);
-    simdgroup_float8x8 o0_7(0.f);
-    float m1=-INFINITY, l1=0.f;
-    simdgroup_float8x8 o1_0(0.f);
-    simdgroup_float8x8 o1_1(0.f);
-    simdgroup_float8x8 o1_2(0.f);
-    simdgroup_float8x8 o1_3(0.f);
-    simdgroup_float8x8 o1_4(0.f);
-    simdgroup_float8x8 o1_5(0.f);
-    simdgroup_float8x8 o1_6(0.f);
-    simdgroup_float8x8 o1_7(0.f);
-    for(uint kb=0;kb<qr+16;kb+=32){
-        simdgroup_float8x8 s0_0(0.f);
-        simdgroup_float8x8 s0_1(0.f);
-        simdgroup_float8x8 s0_2(0.f);
-        simdgroup_float8x8 s0_3(0.f);
-        simdgroup_float8x8 s1_0(0.f);
-        simdgroup_float8x8 s1_1(0.f);
-        simdgroup_float8x8 s1_2(0.f);
-        simdgroup_float8x8 s1_3(0.f);
+    threadgroup half qtile[16*64], kv[64*64];
+    threadgroup float score[16*64];
+    threadgroup half prob[16*64];
+    threadgroup float scales[16];
+    threadgroup float maxes[16], sums[16];
+    uint head=tg/64, qr=(tg%64)*16;
+    uint base=head*1024*64;
+    uint row=tid/8, rl=tid%8;
+    uint mr=(sg/2)*8, nc=(sg%2)*32;
+    uint lane=tid%32, orow=mr+(lane%8)/2+(lane/16)*4;
+    simdgroup_float8x8 o0(0.f),o1(0.f),o2(0.f),o3(0.f);
+    for(uint idx=tid;idx<16*64;idx+=128) qtile[idx]=q[base+qr*64+idx];
+    if(rl==0){maxes[row]=-INFINITY;sums[row]=0.f;}
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    for(uint kb=0;kb<qr+16;kb+=64){
+        for(uint idx=tid;idx<64*64;idx+=128) kv[idx]=k[base+kb*64+idx];
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        simdgroup_float8x8 s0(0.f),s1(0.f),s2(0.f),s3(0.f);
         for(uint d=0;d<64;d+=8){
-            simdgroup_half8x8 b;
-            simdgroup_half8x8 a0;
-            simdgroup_load(a0,q+base+(qr+0)*64+d,64);
-            simdgroup_half8x8 a1;
-            simdgroup_load(a1,q+base+(qr+8)*64+d,64);
-            simdgroup_load(b,k+base+(kb+0)*64+d,64,ulong2(0),true);
-            simdgroup_multiply_accumulate(s0_0,a0,b,s0_0);
-            simdgroup_multiply_accumulate(s1_0,a1,b,s1_0);
-            simdgroup_load(b,k+base+(kb+8)*64+d,64,ulong2(0),true);
-            simdgroup_multiply_accumulate(s0_1,a0,b,s0_1);
-            simdgroup_multiply_accumulate(s1_1,a1,b,s1_1);
-            simdgroup_load(b,k+base+(kb+16)*64+d,64,ulong2(0),true);
-            simdgroup_multiply_accumulate(s0_2,a0,b,s0_2);
-            simdgroup_multiply_accumulate(s1_2,a1,b,s1_2);
-            simdgroup_load(b,k+base+(kb+24)*64+d,64,ulong2(0),true);
-            simdgroup_multiply_accumulate(s0_3,a0,b,s0_3);
-            simdgroup_multiply_accumulate(s1_3,a1,b,s1_3);
+            simdgroup_half8x8 a,b;
+            simdgroup_load(a,qtile+mr*64+d,64);
+            simdgroup_load(b,kv+nc*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0,a,b,s0);
+            simdgroup_load(b,kv+(nc+8)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s1,a,b,s1);
+            simdgroup_load(b,kv+(nc+16)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s2,a,b,s2);
+            simdgroup_load(b,kv+(nc+24)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s3,a,b,s3);
         }
-        float mx0=-INFINITY;
-        s0_0.thread_elements()[0]=kb+0+col+0<=qr+0+row?s0_0.thread_elements()[0]*0.125f:-INFINITY;
-        mx0=max(mx0,s0_0.thread_elements()[0]);
-        s0_0.thread_elements()[1]=kb+0+col+1<=qr+0+row?s0_0.thread_elements()[1]*0.125f:-INFINITY;
-        mx0=max(mx0,s0_0.thread_elements()[1]);
-        s0_1.thread_elements()[0]=kb+8+col+0<=qr+0+row?s0_1.thread_elements()[0]*0.125f:-INFINITY;
-        mx0=max(mx0,s0_1.thread_elements()[0]);
-        s0_1.thread_elements()[1]=kb+8+col+1<=qr+0+row?s0_1.thread_elements()[1]*0.125f:-INFINITY;
-        mx0=max(mx0,s0_1.thread_elements()[1]);
-        s0_2.thread_elements()[0]=kb+16+col+0<=qr+0+row?s0_2.thread_elements()[0]*0.125f:-INFINITY;
-        mx0=max(mx0,s0_2.thread_elements()[0]);
-        s0_2.thread_elements()[1]=kb+16+col+1<=qr+0+row?s0_2.thread_elements()[1]*0.125f:-INFINITY;
-        mx0=max(mx0,s0_2.thread_elements()[1]);
-        s0_3.thread_elements()[0]=kb+24+col+0<=qr+0+row?s0_3.thread_elements()[0]*0.125f:-INFINITY;
-        mx0=max(mx0,s0_3.thread_elements()[0]);
-        s0_3.thread_elements()[1]=kb+24+col+1<=qr+0+row?s0_3.thread_elements()[1]*0.125f:-INFINITY;
-        mx0=max(mx0,s0_3.thread_elements()[1]);
-        mx0=max(mx0,simd_shuffle_xor(mx0,1));
-        mx0=max(mx0,simd_shuffle_xor(mx0,8));
-        mx0=max(m0,mx0);
-        float alpha0=fast::exp(m0-mx0), sum0=0.f;
-        simdgroup_half8x8 p0_0;
-        {float x=fast::exp(s0_0.thread_elements()[0]-mx0);p0_0.thread_elements()[0]=half(x);sum0+=x;}
-        {float x=fast::exp(s0_0.thread_elements()[1]-mx0);p0_0.thread_elements()[1]=half(x);sum0+=x;}
-        simdgroup_store(p0_0,prob+sg*512+0,32);
-        simdgroup_half8x8 p0_1;
-        {float x=fast::exp(s0_1.thread_elements()[0]-mx0);p0_1.thread_elements()[0]=half(x);sum0+=x;}
-        {float x=fast::exp(s0_1.thread_elements()[1]-mx0);p0_1.thread_elements()[1]=half(x);sum0+=x;}
-        simdgroup_store(p0_1,prob+sg*512+8,32);
-        simdgroup_half8x8 p0_2;
-        {float x=fast::exp(s0_2.thread_elements()[0]-mx0);p0_2.thread_elements()[0]=half(x);sum0+=x;}
-        {float x=fast::exp(s0_2.thread_elements()[1]-mx0);p0_2.thread_elements()[1]=half(x);sum0+=x;}
-        simdgroup_store(p0_2,prob+sg*512+16,32);
-        simdgroup_half8x8 p0_3;
-        {float x=fast::exp(s0_3.thread_elements()[0]-mx0);p0_3.thread_elements()[0]=half(x);sum0+=x;}
-        {float x=fast::exp(s0_3.thread_elements()[1]-mx0);p0_3.thread_elements()[1]=half(x);sum0+=x;}
-        simdgroup_store(p0_3,prob+sg*512+24,32);
-        sum0+=simd_shuffle_xor(sum0,1);
-        sum0+=simd_shuffle_xor(sum0,8);
-        l0=l0*alpha0+sum0; m0=mx0;
-        o0_0.thread_elements()[0]*=alpha0;
-        o0_0.thread_elements()[1]*=alpha0;
-        o0_1.thread_elements()[0]*=alpha0;
-        o0_1.thread_elements()[1]*=alpha0;
-        o0_2.thread_elements()[0]*=alpha0;
-        o0_2.thread_elements()[1]*=alpha0;
-        o0_3.thread_elements()[0]*=alpha0;
-        o0_3.thread_elements()[1]*=alpha0;
-        o0_4.thread_elements()[0]*=alpha0;
-        o0_4.thread_elements()[1]*=alpha0;
-        o0_5.thread_elements()[0]*=alpha0;
-        o0_5.thread_elements()[1]*=alpha0;
-        o0_6.thread_elements()[0]*=alpha0;
-        o0_6.thread_elements()[1]*=alpha0;
-        o0_7.thread_elements()[0]*=alpha0;
-        o0_7.thread_elements()[1]*=alpha0;
-        float mx1=-INFINITY;
-        s1_0.thread_elements()[0]=kb+0+col+0<=qr+8+row?s1_0.thread_elements()[0]*0.125f:-INFINITY;
-        mx1=max(mx1,s1_0.thread_elements()[0]);
-        s1_0.thread_elements()[1]=kb+0+col+1<=qr+8+row?s1_0.thread_elements()[1]*0.125f:-INFINITY;
-        mx1=max(mx1,s1_0.thread_elements()[1]);
-        s1_1.thread_elements()[0]=kb+8+col+0<=qr+8+row?s1_1.thread_elements()[0]*0.125f:-INFINITY;
-        mx1=max(mx1,s1_1.thread_elements()[0]);
-        s1_1.thread_elements()[1]=kb+8+col+1<=qr+8+row?s1_1.thread_elements()[1]*0.125f:-INFINITY;
-        mx1=max(mx1,s1_1.thread_elements()[1]);
-        s1_2.thread_elements()[0]=kb+16+col+0<=qr+8+row?s1_2.thread_elements()[0]*0.125f:-INFINITY;
-        mx1=max(mx1,s1_2.thread_elements()[0]);
-        s1_2.thread_elements()[1]=kb+16+col+1<=qr+8+row?s1_2.thread_elements()[1]*0.125f:-INFINITY;
-        mx1=max(mx1,s1_2.thread_elements()[1]);
-        s1_3.thread_elements()[0]=kb+24+col+0<=qr+8+row?s1_3.thread_elements()[0]*0.125f:-INFINITY;
-        mx1=max(mx1,s1_3.thread_elements()[0]);
-        s1_3.thread_elements()[1]=kb+24+col+1<=qr+8+row?s1_3.thread_elements()[1]*0.125f:-INFINITY;
-        mx1=max(mx1,s1_3.thread_elements()[1]);
-        mx1=max(mx1,simd_shuffle_xor(mx1,1));
-        mx1=max(mx1,simd_shuffle_xor(mx1,8));
-        mx1=max(m1,mx1);
-        float alpha1=fast::exp(m1-mx1), sum1=0.f;
-        simdgroup_half8x8 p1_0;
-        {float x=fast::exp(s1_0.thread_elements()[0]-mx1);p1_0.thread_elements()[0]=half(x);sum1+=x;}
-        {float x=fast::exp(s1_0.thread_elements()[1]-mx1);p1_0.thread_elements()[1]=half(x);sum1+=x;}
-        simdgroup_store(p1_0,prob+sg*512+256,32);
-        simdgroup_half8x8 p1_1;
-        {float x=fast::exp(s1_1.thread_elements()[0]-mx1);p1_1.thread_elements()[0]=half(x);sum1+=x;}
-        {float x=fast::exp(s1_1.thread_elements()[1]-mx1);p1_1.thread_elements()[1]=half(x);sum1+=x;}
-        simdgroup_store(p1_1,prob+sg*512+264,32);
-        simdgroup_half8x8 p1_2;
-        {float x=fast::exp(s1_2.thread_elements()[0]-mx1);p1_2.thread_elements()[0]=half(x);sum1+=x;}
-        {float x=fast::exp(s1_2.thread_elements()[1]-mx1);p1_2.thread_elements()[1]=half(x);sum1+=x;}
-        simdgroup_store(p1_2,prob+sg*512+272,32);
-        simdgroup_half8x8 p1_3;
-        {float x=fast::exp(s1_3.thread_elements()[0]-mx1);p1_3.thread_elements()[0]=half(x);sum1+=x;}
-        {float x=fast::exp(s1_3.thread_elements()[1]-mx1);p1_3.thread_elements()[1]=half(x);sum1+=x;}
-        simdgroup_store(p1_3,prob+sg*512+280,32);
-        sum1+=simd_shuffle_xor(sum1,1);
-        sum1+=simd_shuffle_xor(sum1,8);
-        l1=l1*alpha1+sum1; m1=mx1;
-        o1_0.thread_elements()[0]*=alpha1;
-        o1_0.thread_elements()[1]*=alpha1;
-        o1_1.thread_elements()[0]*=alpha1;
-        o1_1.thread_elements()[1]*=alpha1;
-        o1_2.thread_elements()[0]*=alpha1;
-        o1_2.thread_elements()[1]*=alpha1;
-        o1_3.thread_elements()[0]*=alpha1;
-        o1_3.thread_elements()[1]*=alpha1;
-        o1_4.thread_elements()[0]*=alpha1;
-        o1_4.thread_elements()[1]*=alpha1;
-        o1_5.thread_elements()[0]*=alpha1;
-        o1_5.thread_elements()[1]*=alpha1;
-        o1_6.thread_elements()[0]*=alpha1;
-        o1_6.thread_elements()[1]*=alpha1;
-        o1_7.thread_elements()[0]*=alpha1;
-        o1_7.thread_elements()[1]*=alpha1;
-        simdgroup_barrier(mem_flags::mem_threadgroup);
-        for(uint d=0;d<32;d+=8){
-            simdgroup_half8x8 b;
-            simdgroup_half8x8 a0;
-            simdgroup_load(a0,prob+sg*512+0+d,32);
-            simdgroup_half8x8 a1;
-            simdgroup_load(a1,prob+sg*512+256+d,32);
-            simdgroup_load(b,v+base+(kb+d)*64+0,64);
-            simdgroup_multiply_accumulate(o0_0,a0,b,o0_0);
-            simdgroup_multiply_accumulate(o1_0,a1,b,o1_0);
-            simdgroup_load(b,v+base+(kb+d)*64+8,64);
-            simdgroup_multiply_accumulate(o0_1,a0,b,o0_1);
-            simdgroup_multiply_accumulate(o1_1,a1,b,o1_1);
-            simdgroup_load(b,v+base+(kb+d)*64+16,64);
-            simdgroup_multiply_accumulate(o0_2,a0,b,o0_2);
-            simdgroup_multiply_accumulate(o1_2,a1,b,o1_2);
-            simdgroup_load(b,v+base+(kb+d)*64+24,64);
-            simdgroup_multiply_accumulate(o0_3,a0,b,o0_3);
-            simdgroup_multiply_accumulate(o1_3,a1,b,o1_3);
-            simdgroup_load(b,v+base+(kb+d)*64+32,64);
-            simdgroup_multiply_accumulate(o0_4,a0,b,o0_4);
-            simdgroup_multiply_accumulate(o1_4,a1,b,o1_4);
-            simdgroup_load(b,v+base+(kb+d)*64+40,64);
-            simdgroup_multiply_accumulate(o0_5,a0,b,o0_5);
-            simdgroup_multiply_accumulate(o1_5,a1,b,o1_5);
-            simdgroup_load(b,v+base+(kb+d)*64+48,64);
-            simdgroup_multiply_accumulate(o0_6,a0,b,o0_6);
-            simdgroup_multiply_accumulate(o1_6,a1,b,o1_6);
-            simdgroup_load(b,v+base+(kb+d)*64+56,64);
-            simdgroup_multiply_accumulate(o0_7,a0,b,o0_7);
-            simdgroup_multiply_accumulate(o1_7,a1,b,o1_7);
+        simdgroup_store(s0,score+mr*64+nc,64);
+        simdgroup_store(s1,score+mr*64+nc+8,64);
+        simdgroup_store(s2,score+mr*64+nc+16,64);
+        simdgroup_store(s3,score+mr*64+nc+24,64);
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        float xs[8]; float mx=-INFINITY;
+        for(uint i=0;i<8;i++){
+            uint c=rl+i*8;
+            xs[i]=kb+c<=qr+row?score[row*64+c]*0.125f:-INFINITY;
+            mx=max(mx,xs[i]);
         }
+        mx=max(mx,simd_shuffle_xor(mx,1));
+        mx=max(mx,simd_shuffle_xor(mx,2));
+        mx=max(mx,simd_shuffle_xor(mx,4));
+        mx=max(mx,maxes[row]);
+        float alpha=fast::exp(maxes[row]-mx), sum=0.f;
+        for(uint i=0;i<8;i++){
+            xs[i]=fast::exp(xs[i]-mx);
+            sum+=xs[i];
+            prob[row*64+rl+i*8]=half(xs[i]);
+
+        }
+        sum+=simd_shuffle_xor(sum,1);
+        sum+=simd_shuffle_xor(sum,2);
+        sum+=simd_shuffle_xor(sum,4);
+        if(rl==0){sums[row]=sums[row]*alpha+sum;maxes[row]=mx;scales[row]=alpha;}
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for(uint idx=tid;idx<64*64;idx+=128) kv[idx]=v[base+kb*64+idx];
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        float factor=scales[orow];
+        for(uint e=0;e<2;e++){
+            o0.thread_elements()[e]*=factor;
+            o1.thread_elements()[e]*=factor;
+            o2.thread_elements()[e]*=factor;
+            o3.thread_elements()[e]*=factor;
+        }
+        for(uint d=0;d<64;d+=8){
+            simdgroup_half8x8 a,b;
+            simdgroup_load(a,prob+mr*64+d,64);
+            simdgroup_load(b,kv+d*64+nc,64);
+            simdgroup_multiply_accumulate(o0,a,b,o0);
+            simdgroup_load(b,kv+d*64+nc+8,64);
+            simdgroup_multiply_accumulate(o1,a,b,o1);
+            simdgroup_load(b,kv+d*64+nc+16,64);
+            simdgroup_multiply_accumulate(o2,a,b,o2);
+            simdgroup_load(b,kv+d*64+nc+24,64);
+            simdgroup_multiply_accumulate(o3,a,b,o3);
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
     }
-    o0_0.thread_elements()[0]/=l0;
-    o0_0.thread_elements()[1]/=l0;
-    simdgroup_store(o0_0,out+base+(qr+0)*64+0,64);
-    o0_1.thread_elements()[0]/=l0;
-    o0_1.thread_elements()[1]/=l0;
-    simdgroup_store(o0_1,out+base+(qr+0)*64+8,64);
-    o0_2.thread_elements()[0]/=l0;
-    o0_2.thread_elements()[1]/=l0;
-    simdgroup_store(o0_2,out+base+(qr+0)*64+16,64);
-    o0_3.thread_elements()[0]/=l0;
-    o0_3.thread_elements()[1]/=l0;
-    simdgroup_store(o0_3,out+base+(qr+0)*64+24,64);
-    o0_4.thread_elements()[0]/=l0;
-    o0_4.thread_elements()[1]/=l0;
-    simdgroup_store(o0_4,out+base+(qr+0)*64+32,64);
-    o0_5.thread_elements()[0]/=l0;
-    o0_5.thread_elements()[1]/=l0;
-    simdgroup_store(o0_5,out+base+(qr+0)*64+40,64);
-    o0_6.thread_elements()[0]/=l0;
-    o0_6.thread_elements()[1]/=l0;
-    simdgroup_store(o0_6,out+base+(qr+0)*64+48,64);
-    o0_7.thread_elements()[0]/=l0;
-    o0_7.thread_elements()[1]/=l0;
-    simdgroup_store(o0_7,out+base+(qr+0)*64+56,64);
-    o1_0.thread_elements()[0]/=l1;
-    o1_0.thread_elements()[1]/=l1;
-    simdgroup_store(o1_0,out+base+(qr+8)*64+0,64);
-    o1_1.thread_elements()[0]/=l1;
-    o1_1.thread_elements()[1]/=l1;
-    simdgroup_store(o1_1,out+base+(qr+8)*64+8,64);
-    o1_2.thread_elements()[0]/=l1;
-    o1_2.thread_elements()[1]/=l1;
-    simdgroup_store(o1_2,out+base+(qr+8)*64+16,64);
-    o1_3.thread_elements()[0]/=l1;
-    o1_3.thread_elements()[1]/=l1;
-    simdgroup_store(o1_3,out+base+(qr+8)*64+24,64);
-    o1_4.thread_elements()[0]/=l1;
-    o1_4.thread_elements()[1]/=l1;
-    simdgroup_store(o1_4,out+base+(qr+8)*64+32,64);
-    o1_5.thread_elements()[0]/=l1;
-    o1_5.thread_elements()[1]/=l1;
-    simdgroup_store(o1_5,out+base+(qr+8)*64+40,64);
-    o1_6.thread_elements()[0]/=l1;
-    o1_6.thread_elements()[1]/=l1;
-    simdgroup_store(o1_6,out+base+(qr+8)*64+48,64);
-    o1_7.thread_elements()[0]/=l1;
-    o1_7.thread_elements()[1]/=l1;
-    simdgroup_store(o1_7,out+base+(qr+8)*64+56,64);
+    float inv=1.f/sums[orow];
+    for(uint e=0;e<2;e++){
+        o0.thread_elements()[e]*=inv;
+        o1.thread_elements()[e]*=inv;
+        o2.thread_elements()[e]*=inv;
+        o3.thread_elements()[e]*=inv;
+    }
+    simdgroup_store(o0,out+base+(qr+mr)*64+nc,64);
+    simdgroup_store(o1,out+base+(qr+mr)*64+nc+8,64);
+    simdgroup_store(o2,out+base+(qr+mr)*64+nc+16,64);
+    simdgroup_store(o3,out+base+(qr+mr)*64+nc+24,64);
 }
 ''')
 
@@ -257,5 +106,5 @@
 def kernel(q, k, v):
     qh, kh, vh = q.half(), k.half(), v.half()
     out = torch.empty_like(q)
-    _lib.flash(qh, kh, vh, out, threads=q.numel()//4096*128, group_size=128)
+    _lib.flash(qh, kh, vh, out, threads=q.numel()//1024*128, group_size=128)
     return out

```

</details>

`python bench.py`  
→ run 8 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 70.53 ms, kernel 4.57 ms, speedup 15.42x

> The versions that kept all scores in registers were slower, around 9 ms. The best so far remains 5.46 ms. I’m using the last three timing runs to test explicit matrix operations and staged input tiles, then compare the strongest candidate.

- `python -c 'from pathlib import Path; s=Path(\"queue/run5.py\").read_text(); s=s.replace(\"[32*64]\", \"[16*64]\").replace(\"scales[32]\", \"scales[16]\").rep...`
- `python check.py`  
→ OK: relative error 3.18e-04 (allowed 2e-03)
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -8,97 +8,332 @@
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup half qtile[16*64], kv[64*64];
-    threadgroup float score[16*64];
-    threadgroup half prob[16*64];
-    threadgroup float scales[16];
-    threadgroup float maxes[16], sums[16];
-    uint head=tg/64, qr=(tg%64)*16;
-    uint base=head*1024*64;
-    uint row=tid/8, rl=tid%8;
-    uint mr=(sg/2)*8, nc=(sg%2)*32;
-    uint lane=tid%32, orow=mr+(lane%8)/2+(lane/16)*4;
-    simdgroup_float8x8 o0(0.f),o1(0.f),o2(0.f),o3(0.f);
-    for(uint idx=tid;idx<16*64;idx+=128) qtile[idx]=q[base+qr*64+idx];
-    if(rl==0){maxes[row]=-INFINITY;sums[row]=0.f;}
-    threadgroup_barrier(mem_flags::mem_threadgroup);
+    threadgroup half prob[4096];
+    uint head=tg/16, qr=(tg%16)*64+sg*16, base=head*1024*64;
+    uint lane=tid%32, row=(lane%8)/2+(lane/16)*4, col=(lane%2)*2+((lane/8)%2)*4;
+    float m0=-INFINITY, l0=0.f;
+    simdgroup_float8x8 o0_0(0.f);
+    simdgroup_float8x8 o0_1(0.f);
+    simdgroup_float8x8 o0_2(0.f);
+    simdgroup_float8x8 o0_3(0.f);
+    simdgroup_float8x8 o0_4(0.f);
+    simdgroup_float8x8 o0_5(0.f);
+    simdgroup_float8x8 o0_6(0.f);
+    simdgroup_float8x8 o0_7(0.f);
+    float m1=-INFINITY, l1=0.f;
+    simdgroup_float8x8 o1_0(0.f);
+    simdgroup_float8x8 o1_1(0.f);
+    simdgroup_float8x8 o1_2(0.f);
+    simdgroup_float8x8 o1_3(0.f);
+    simdgroup_float8x8 o1_4(0.f);
+    simdgroup_float8x8 o1_5(0.f);
+    simdgroup_float8x8 o1_6(0.f);
+    simdgroup_float8x8 o1_7(0.f);
     for(uint kb=0;kb<qr+16;kb+=64){
-        for(uint idx=tid;idx<64*64;idx+=128) kv[idx]=k[base+kb*64+idx];
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 s0(0.f),s1(0.f),s2(0.f),s3(0.f);
+        simdgroup_float8x8 s0_0(0.f);
+        simdgroup_float8x8 s0_1(0.f);
+        simdgroup_float8x8 s0_2(0.f);
+        simdgroup_float8x8 s0_3(0.f);
+        simdgroup_float8x8 s0_4(0.f);
+        simdgroup_float8x8 s0_5(0.f);
+        simdgroup_float8x8 s0_6(0.f);
+        simdgroup_float8x8 s0_7(0.f);
+        simdgroup_float8x8 s1_0(0.f);
+        simdgroup_float8x8 s1_1(0.f);
+        simdgroup_float8x8 s1_2(0.f);
+        simdgroup_float8x8 s1_3(0.f);
+        simdgroup_float8x8 s1_4(0.f);
+        simdgroup_float8x8 s1_5(0.f);
+        simdgroup_float8x8 s1_6(0.f);
+        simdgroup_float8x8 s1_7(0.f);
         for(uint d=0;d<64;d+=8){
-            simdgroup_half8x8 a,b;
-            simdgroup_load(a,qtile+mr*64+d,64);
-            simdgroup_load(b,kv+nc*64+d,64,ulong2(0),true);
-            simdgroup_multiply_accumulate(s0,a,b,s0);
-            simdgroup_load(b,kv+(nc+8)*64+d,64,ulong2(0),true);
-            simdgroup_multiply_accumulate(s1,a,b,s1);
-            simdgroup_load(b,kv+(nc+16)*64+d,64,ulong2(0),true);
-            simdgroup_multiply_accumulate(s2,a,b,s2);
-            simdgroup_load(b,kv+(nc+24)*64+d,64,ulong2(0),true);
-            simdgroup_multiply_accumulate(s3,a,b,s3);
+            simdgroup_half8x8 b;
+            simdgroup_half8x8 a0;
+            simdgroup_load(a0,q+base+(qr+0)*64+d,64);
+            simdgroup_half8x8 a1;
+            simdgroup_load(a1,q+base+(qr+8)*64+d,64);
+            simdgroup_load(b,k+base+(kb+0)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0_0,a0,b,s0_0);
+            simdgroup_multiply_accumulate(s1_0,a1,b,s1_0);
+            simdgroup_load(b,k+base+(kb+8)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0_1,a0,b,s0_1);
+            simdgroup_multiply_accumulate(s1_1,a1,b,s1_1);
+            simdgroup_load(b,k+base+(kb+16)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0_2,a0,b,s0_2);
+            simdgroup_multiply_accumulate(s1_2,a1,b,s1_2);
+            simdgroup_load(b,k+base+(kb+24)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0_3,a0,b,s0_3);
+            simdgroup_multiply_accumulate(s1_3,a1,b,s1_3);
+            simdgroup_load(b,k+base+(kb+32)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0_4,a0,b,s0_4);
+            simdgroup_multiply_accumulate(s1_4,a1,b,s1_4);
+            simdgroup_load(b,k+base+(kb+40)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0_5,a0,b,s0_5);
+            simdgroup_multiply_accumulate(s1_5,a1,b,s1_5);
+            simdgroup_load(b,k+base+(kb+48)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0_6,a0,b,s0_6);
+            simdgroup_multiply_accumulate(s1_6,a1,b,s1_6);
+            simdgroup_load(b,k+base+(kb+56)*64+d,64,ulong2(0),true);
+            simdgroup_multiply_accumulate(s0_7,a0,b,s0_7);
+            simdgroup_multiply_accumulate(s1_7,a1,b,s1_7);
         }
-        simdgroup_store(s0,score+mr*64+nc,64);
-        simdgroup_store(s1,score+mr*64+nc+8,64);
-        simdgroup_store(s2,score+mr*64+nc+16,64);
-        simdgroup_store(s3,score+mr*64+nc+24,64);
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        float xs[8]; float mx=-INFINITY;
-        for(uint i=0;i<8;i++){
-            uint c=rl+i*8;
-            xs[i]=kb+c<=qr+row?score[row*64+c]*0.125f:-INFINITY;
-            mx=max(mx,xs[i]);
+        float mx0=-INFINITY;
+        s0_0.thread_elements()[0]=kb+0+col+0<=qr+0+row?s0_0.thread_elements()[0]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_0.thread_elements()[0]);
+        s0_0.thread_elements()[1]=kb+0+col+1<=qr+0+row?s0_0.thread_elements()[1]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_0.thread_elements()[1]);
+        s0_1.thread_elements()[0]=kb+8+col+0<=qr+0+row?s0_1.thread_elements()[0]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_1.thread_elements()[0]);
+        s0_1.thread_elements()[1]=kb+8+col+1<=qr+0+row?s0_1.thread_elements()[1]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_1.thread_elements()[1]);
+        s0_2.thread_elements()[0]=kb+16+col+0<=qr+0+row?s0_2.thread_elements()[0]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_2.thread_elements()[0]);
+        s0_2.thread_elements()[1]=kb+16+col+1<=qr+0+row?s0_2.thread_elements()[1]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_2.thread_elements()[1]);
+        s0_3.thread_elements()[0]=kb+24+col+0<=qr+0+row?s0_3.thread_elements()[0]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_3.thread_elements()[0]);
+        s0_3.thread_elements()[1]=kb+24+col+1<=qr+0+row?s0_3.thread_elements()[1]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_3.thread_elements()[1]);
+        s0_4.thread_elements()[0]=kb+32+col+0<=qr+0+row?s0_4.thread_elements()[0]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_4.thread_elements()[0]);
+        s0_4.thread_elements()[1]=kb+32+col+1<=qr+0+row?s0_4.thread_elements()[1]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_4.thread_elements()[1]);
+        s0_5.thread_elements()[0]=kb+40+col+0<=qr+0+row?s0_5.thread_elements()[0]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_5.thread_elements()[0]);
+        s0_5.thread_elements()[1]=kb+40+col+1<=qr+0+row?s0_5.thread_elements()[1]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_5.thread_elements()[1]);
+        s0_6.thread_elements()[0]=kb+48+col+0<=qr+0+row?s0_6.thread_elements()[0]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_6.thread_elements()[0]);
+        s0_6.thread_elements()[1]=kb+48+col+1<=qr+0+row?s0_6.thread_elements()[1]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_6.thread_elements()[1]);
+        s0_7.thread_elements()[0]=kb+56+col+0<=qr+0+row?s0_7.thread_elements()[0]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_7.thread_elements()[0]);
+        s0_7.thread_elements()[1]=kb+56+col+1<=qr+0+row?s0_7.thread_elements()[1]*0.125f:-INFINITY;
+        mx0=max(mx0,s0_7.thread_elements()[1]);
+        mx0=max(mx0,simd_shuffle_xor(mx0,1));
+        mx0=max(mx0,simd_shuffle_xor(mx0,8));
+        mx0=max(m0,mx0);
+        float alpha0=fast::exp(m0-mx0), sum0=0.f;
+        simdgroup_half8x8 p0_0;
+        {float x=fast::exp(s0_0.thread_elements()[0]-mx0);p0_0.thread_elements()[0]=half(x);sum0+=x;}
+        {float x=fast::exp(s0_0.thread_elements()[1]-mx0);p0_0.thread_elements()[1]=half(x);sum0+=x;}
+        simdgroup_store(p0_0,prob+sg*1024+0,64);
+        simdgroup_half8x8 p0_1;
+        {float x=fast::exp(s0_1.thread_elements()[0]-mx0);p0_1.thread_elements()[0]=half(x);sum0+=x;}
+        {float x=fast::exp(s0_1.thread_elements()[1]-mx0);p0_1.thread_elements()[1]=half(x);sum0+=x;}
+        simdgroup_store(p0_1,prob+sg*1024+8,64);
+        simdgroup_half8x8 p0_2;
+        {float x=fast::exp(s0_2.thread_elements()[0]-mx0);p0_2.thread_elements()[0]=half(x);sum0+=x;}
+        {float x=fast::exp(s0_2.thread_elements()[1]-mx0);p0_2.thread_elements()[1]=half(x);sum0+=x;}
+        simdgroup_store(p0_2,prob+sg*1024+16,64);
+        simdgroup_half8x8 p0_3;
+        {float x=fast::exp(s0_3.thread_elements()[0]-mx0);p0_3.thread_elements()[0]=half(x);sum0+=x;}
+        {float x=fast::exp(s0_3.thread_elements()[1]-mx0);p0_3.thread_elements()[1]=half(x);sum0+=x;}
+        simdgroup_store(p0_3,prob+sg*1024+24,64);
+        simdgroup_half8x8 p0_4;
+        {float x=fast::exp(s0_4.thread_elements()[0]-mx0);p0_4.thread_elements()[0]=half(x);sum0+=x;}
+        {float x=fast::exp(s0_4.thread_elements()[1]-mx0);p0_4.thread_elements()[1]=half(x);sum0+=x;}
+        simdgroup_store(p0_4,prob+sg*1024+32,64);
+        simdgroup_half8x8 p0_5;
+        {float x=fast::exp(s0_5.thread_elements()[0]-mx0);p0_5.thread_elements()[0]=half(x);sum0+=x;}
+        {float x=fast::exp(s0_5.thread_elements()[1]-mx0);p0_5.thread_elements()[1]=half(x);sum0+=x;}
+        simdgroup_store(p0_5,prob+sg*1024+40,64);
+        simdgroup_half8x8 p0_6;
+        {float x=fast::exp(s0_6.thread_elements()[0]-mx0);p0_6.thread_elements()[0]=half(x);sum0+=x;}
+        {float x=fast::exp(s0_6.thread_elements()[1]-mx0);p0_6.thread_elements()[1]=half(x);sum0+=x;}
+        simdgroup_store(p0_6,prob+sg*1024+48,64);
+        simdgroup_half8x8 p0_7;
+        {float x=fast::exp(s0_7.thread_elements()[0]-mx0);p0_7.thread_elements()[0]=half(x);sum0+=x;}
+        {float x=fast::exp(s0_7.thread_elements()[1]-mx0);p0_7.thread_elements()[1]=half(x);sum0+=x;}
+        simdgroup_store(p0_7,prob+sg*1024+56,64);
+        sum0+=simd_shuffle_xor(sum0,1);
+        sum0+=simd_shuffle_xor(sum0,8);
+        l0=l0*alpha0+sum0; m0=mx0;
+        o0_0.thread_elements()[0]*=alpha0;
+        o0_0.thread_elements()[1]*=alpha0;
+        o0_1.thread_elements()[0]*=alpha0;
+        o0_1.thread_elements()[1]*=alpha0;
+        o0_2.thread_elements()[0]*=alpha0;
+        o0_2.thread_elements()[1]*=alpha0;
+        o0_3.thread_elements()[0]*=alpha0;
+        o0_3.thread_elements()[1]*=alpha0;
+        o0_4.thread_elements()[0]*=alpha0;
+        o0_4.thread_elements()[1]*=alpha0;
+        o0_5.thread_elements()[0]*=alpha0;
+        o0_5.thread_elements()[1]*=alpha0;
+        o0_6.thread_elements()[0]*=alpha0;
+        o0_6.thread_elements()[1]*=alpha0;
+        o0_7.thread_elements()[0]*=alpha0;
+        o0_7.thread_elements()[1]*=alpha0;
+        float mx1=-INFINITY;
+        s1_0.thread_elements()[0]=kb+0+col+0<=qr+8+row?s1_0.thread_elements()[0]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_0.thread_elements()[0]);
+        s1_0.thread_elements()[1]=kb+0+col+1<=qr+8+row?s1_0.thread_elements()[1]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_0.thread_elements()[1]);
+        s1_1.thread_elements()[0]=kb+8+col+0<=qr+8+row?s1_1.thread_elements()[0]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_1.thread_elements()[0]);
+        s1_1.thread_elements()[1]=kb+8+col+1<=qr+8+row?s1_1.thread_elements()[1]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_1.thread_elements()[1]);
+        s1_2.thread_elements()[0]=kb+16+col+0<=qr+8+row?s1_2.thread_elements()[0]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_2.thread_elements()[0]);
+        s1_2.thread_elements()[1]=kb+16+col+1<=qr+8+row?s1_2.thread_elements()[1]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_2.thread_elements()[1]);
+        s1_3.thread_elements()[0]=kb+24+col+0<=qr+8+row?s1_3.thread_elements()[0]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_3.thread_elements()[0]);
+        s1_3.thread_elements()[1]=kb+24+col+1<=qr+8+row?s1_3.thread_elements()[1]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_3.thread_elements()[1]);
+        s1_4.thread_elements()[0]=kb+32+col+0<=qr+8+row?s1_4.thread_elements()[0]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_4.thread_elements()[0]);
+        s1_4.thread_elements()[1]=kb+32+col+1<=qr+8+row?s1_4.thread_elements()[1]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_4.thread_elements()[1]);
+        s1_5.thread_elements()[0]=kb+40+col+0<=qr+8+row?s1_5.thread_elements()[0]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_5.thread_elements()[0]);
+        s1_5.thread_elements()[1]=kb+40+col+1<=qr+8+row?s1_5.thread_elements()[1]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_5.thread_elements()[1]);
+        s1_6.thread_elements()[0]=kb+48+col+0<=qr+8+row?s1_6.thread_elements()[0]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_6.thread_elements()[0]);
+        s1_6.thread_elements()[1]=kb+48+col+1<=qr+8+row?s1_6.thread_elements()[1]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_6.thread_elements()[1]);
+        s1_7.thread_elements()[0]=kb+56+col+0<=qr+8+row?s1_7.thread_elements()[0]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_7.thread_elements()[0]);
+        s1_7.thread_elements()[1]=kb+56+col+1<=qr+8+row?s1_7.thread_elements()[1]*0.125f:-INFINITY;
+        mx1=max(mx1,s1_7.thread_elements()[1]);
+        mx1=max(mx1,simd_shuffle_xor(mx1,1));
+        mx1=max(mx1,simd_shuffle_xor(mx1,8));
+        mx1=max(m1,mx1);
+        float alpha1=fast::exp(m1-mx1), sum1=0.f;
+        simdgroup_half8x8 p1_0;
+        {float x=fast::exp(s1_0.thread_elements()[0]-mx1);p1_0.thread_elements()[0]=half(x);sum1+=x;}
+        {float x=fast::exp(s1_0.thread_elements()[1]-mx1);p1_0.thread_elements()[1]=half(x);sum1+=x;}
+        simdgroup_store(p1_0,prob+sg*1024+512,64);
+        simdgroup_half8x8 p1_1;
+        {float x=fast::exp(s1_1.thread_elements()[0]-mx1);p1_1.thread_elements()[0]=half(x);sum1+=x;}
+        {float x=fast::exp(s1_1.thread_elements()[1]-mx1);p1_1.thread_elements()[1]=half(x);sum1+=x;}
+        simdgroup_store(p1_1,prob+sg*1024+520,64);
+        simdgroup_half8x8 p1_2;
+        {float x=fast::exp(s1_2.thread_elements()[0]-mx1);p1_2.thread_elements()[0]=half(x);sum1+=x;}
+        {float x=fast::exp(s1_2.thread_elements()[1]-mx1);p1_2.thread_elements()[1]=half(x);sum1+=x;}
+        simdgroup_store(p1_2,prob+sg*1024+528,64);
+        simdgroup_half8x8 p1_3;
+        {float x=fast::exp(s1_3.thread_elements()[0]-mx1);p1_3.thread_elements()[0]=half(x);sum1+=x;}
+        {float x=fast::exp(s1_3.thread_elements()[1]-mx1);p1_3.thread_elements()[1]=half(x);sum1+=x;}
+        simdgroup_store(p1_3,prob+sg*1024+536,64);
+        simdgroup_half8x8 p1_4;
+        {float x=fast::exp(s1_4.thread_elements()[0]-mx1);p1_4.thread_elements()[0]=half(x);sum1+=x;}
+        {float x=fast::exp(s1_4.thread_elements()[1]-mx1);p1_4.thread_elements()[1]=half(x);sum1+=x;}
+        simdgroup_store(p1_4,prob+sg*1024+544,64);
+        simdgroup_half8x8 p1_5;
+        {float x=fast::exp(s1_5.thread_elements()[0]-mx1);p1_5.thread_elements()[0]=half(x);sum1+=x;}
+        {float x=fast::exp(s1_5.thread_elements()[1]-mx1);p1_5.thread_elements()[1]=half(x);sum1+=x;}
+        simdgroup_store(p1_5,prob+sg*1024+552,64);
+        simdgroup_half8x8 p1_6;
+        {float x=fast::exp(s1_6.thread_elements()[0]-mx1);p1_6.thread_elements()[0]=half(x);sum1+=x;}
+        {float x=fast::exp(s1_6.thread_elements()[1]-mx1);p1_6.thread_elements()[1]=half(x);sum1+=x;}
+        simdgroup_store(p1_6,prob+sg*1024+560,64);
+        simdgroup_half8x8 p1_7;
+        {float x=fast::exp(s1_7.thread_elements()[0]-mx1);p1_7.thread_elements()[0]=half(x);sum1+=x;}
+        {float x=fast::exp(s1_7.thread_elements()[1]-mx1);p1_7.thread_elements()[1]=half(x);sum1+=x;}
+        simdgroup_store(p1_7,prob+sg*1024+568,64);
+        sum1+=simd_shuffle_xor(sum1,1);
+        sum1+=simd_shuffle_xor(sum1,8);
+        l1=l1*alpha1+sum1; m1=mx1;
+        o1_0.thread_elements()[0]*=alpha1;
+        o1_0.thread_elements()[1]*=alpha1;
+        o1_1.thread_elements()[0]*=alpha1;
+        o1_1.thread_elements()[1]*=alpha1;
+        o1_2.thread_elements()[0]*=alpha1;
+        o1_2.thread_elements()[1]*=alpha1;
+        o1_3.thread_elements()[0]*=alpha1;
+        o1_3.thread_elements()[1]*=alpha1;
+        o1_4.thread_elements()[0]*=alpha1;
+        o1_4.thread_elements()[1]*=alpha1;
+        o1_5.thread_elements()[0]*=alpha1;
+        o1_5.thread_elements()[1]*=alpha1;
+        o1_6.thread_elements()[0]*=alpha1;
+        o1_6.thread_elements()[1]*=alpha1;
+        o1_7.thread_elements()[0]*=alpha1;
+        o1_7.thread_elements()[1]*=alpha1;
+        simdgroup_barrier(mem_flags::mem_threadgroup);
+        for(uint d=0;d<64;d+=8){
+            simdgroup_half8x8 b;
+            simdgroup_half8x8 a0;
+            simdgroup_load(a0,prob+sg*1024+0+d,64);
+            simdgroup_half8x8 a1;
+            simdgroup_load(a1,prob+sg*1024+512+d,64);
+            simdgroup_load(b,v+base+(kb+d)*64+0,64);
+            simdgroup_multiply_accumulate(o0_0,a0,b,o0_0);
+            simdgroup_multiply_accumulate(o1_0,a1,b,o1_0);
+            simdgroup_load(b,v+base+(kb+d)*64+8,64);
+            simdgroup_multiply_accumulate(o0_1,a0,b,o0_1);
+            simdgroup_multiply_accumulate(o1_1,a1,b,o1_1);
+            simdgroup_load(b,v+base+(kb+d)*64+16,64);
+            simdgroup_multiply_accumulate(o0_2,a0,b,o0_2);
+            simdgroup_multiply_accumulate(o1_2,a1,b,o1_2);
+            simdgroup_load(b,v+base+(kb+d)*64+24,64);
+            simdgroup_multiply_accumulate(o0_3,a0,b,o0_3);
+            simdgroup_multiply_accumulate(o1_3,a1,b,o1_3);
+            simdgroup_load(b,v+base+(kb+d)*64+32,64);
+            simdgroup_multiply_accumulate(o0_4,a0,b,o0_4);
+            simdgroup_multiply_accumulate(o1_4,a1,b,o1_4);
+            simdgroup_load(b,v+base+(kb+d)*64+40,64);
+            simdgroup_multiply_accumulate(o0_5,a0,b,o0_5);
+            simdgroup_multiply_accumulate(o1_5,a1,b,o1_5);
+            simdgroup_load(b,v+base+(kb+d)*64+48,64);
+            simdgroup_multiply_accumulate(o0_6,a0,b,o0_6);
+            simdgroup_multiply_accumulate(o1_6,a1,b,o1_6);
+            simdgroup_load(b,v+base+(kb+d)*64+56,64);
+            simdgroup_multiply_accumulate(o0_7,a0,b,o0_7);
+            simdgroup_multiply_accumulate(o1_7,a1,b,o1_7);
         }
-        mx=max(mx,simd_shuffle_xor(mx,1));
-        mx=max(mx,simd_shuffle_xor(mx,2));
-        mx=max(mx,simd_shuffle_xor(mx,4));
-        mx=max(mx,maxes[row]);
-        float alpha=fast::exp(maxes[row]-mx), sum=0.f;
-        for(uint i=0;i<8;i++){
-            xs[i]=fast::exp(xs[i]-mx);
-            sum+=xs[i];
-            prob[row*64+rl+i*8]=half(xs[i]);
-
-        }
-        sum+=simd_shuffle_xor(sum,1);
-        sum+=simd_shuffle_xor(sum,2);
-        sum+=simd_shuffle_xor(sum,4);
-        if(rl==0){sums[row]=sums[row]*alpha+sum;maxes[row]=mx;scales[row]=alpha;}
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        for(uint idx=tid;idx<64*64;idx+=128) kv[idx]=v[base+kb*64+idx];
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        float factor=scales[orow];
-        for(uint e=0;e<2;e++){
-            o0.thread_elements()[e]*=factor;
-            o1.thread_elements()[e]*=factor;
-            o2.thread_elements()[e]*=factor;
-            o3.thread_elements()[e]*=factor;
-        }
-        for(uint d=0;d<64;d+=8){
-            simdgroup_half8x8 a,b;
-            simdgroup_load(a,prob+mr*64+d,64);
-            simdgroup_load(b,kv+d*64+nc,64);
-            simdgroup_multiply_accumulate(o0,a,b,o0);
-            simdgroup_load(b,kv+d*64+nc+8,64);
-            simdgroup_multiply_accumulate(o1,a,b,o1);
-            simdgroup_load(b,kv+d*64+nc+16,64);
-            simdgroup_multiply_accumulate(o2,a,b,o2);
-            simdgroup_load(b,kv+d*64+nc+24,64);
-            simdgroup_multiply_accumulate(o3,a,b,o3);
-        }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
     }
-    float inv=1.f/sums[orow];
-    for(uint e=0;e<2;e++){
-        o0.thread_elements()[e]*=inv;
-        o1.thread_elements()[e]*=inv;
-        o2.thread_elements()[e]*=inv;
-        o3.thread_elements()[e]*=inv;
-    }
-    simdgroup_store(o0,out+base+(qr+mr)*64+nc,64);
-    simdgroup_store(o1,out+base+(qr+mr)*64+nc+8,64);
-    simdgroup_store(o2,out+base+(qr+mr)*64+nc+16,64);
-    simdgroup_store(o3,out+base+(qr+mr)*64+nc+24,64);
+    o0_0.thread_elements()[0]/=l0;
+    o0_0.thread_elements()[1]/=l0;
+    simdgroup_store(o0_0,out+base+(qr+0)*64+0,64);
+    o0_1.thread_elements()[0]/=l0;
+    o0_1.thread_elements()[1]/=l0;
+    simdgroup_store(o0_1,out+base+(qr+0)*64+8,64);
+    o0_2.thread_elements()[0]/=l0;
+    o0_2.thread_elements()[1]/=l0;
+    simdgroup_store(o0_2,out+base+(qr+0)*64+16,64);
+    o0_3.thread_elements()[0]/=l0;
+    o0_3.thread_elements()[1]/=l0;
+    simdgroup_store(o0_3,out+base+(qr+0)*64+24,64);
+    o0_4.thread_elements()[0]/=l0;
+    o0_4.thread_elements()[1]/=l0;
+    simdgroup_store(o0_4,out+base+(qr+0)*64+32,64);
+    o0_5.thread_elements()[0]/=l0;
+    o0_5.thread_elements()[1]/=l0;
+    simdgroup_store(o0_5,out+base+(qr+0)*64+40,64);
+    o0_6.thread_elements()[0]/=l0;
+    o0_6.thread_elements()[1]/=l0;
+    simdgroup_store(o0_6,out+base+(qr+0)*64+48,64);
+    o0_7.thread_elements()[0]/=l0;
+    o0_7.thread_elements()[1]/=l0;
+    simdgroup_store(o0_7,out+base+(qr+0)*64+56,64);
+    o1_0.thread_elements()[0]/=l1;
+    o1_0.thread_elements()[1]/=l1;
+    simdgroup_store(o1_0,out+base+(qr+8)*64+0,64);
+    o1_1.thread_elements()[0]/=l1;
+    o1_1.thread_elements()[1]/=l1;
+    simdgroup_store(o1_1,out+base+(qr+8)*64+8,64);
+    o1_2.thread_elements()[0]/=l1;
+    o1_2.thread_elements()[1]/=l1;
+    simdgroup_store(o1_2,out+base+(qr+8)*64+16,64);
+    o1_3.thread_elements()[0]/=l1;
+    o1_3.thread_elements()[1]/=l1;
+    simdgroup_store(o1_3,out+base+(qr+8)*64+24,64);
+    o1_4.thread_elements()[0]/=l1;
+    o1_4.thread_elements()[1]/=l1;
+    simdgroup_store(o1_4,out+base+(qr+8)*64+32,64);
+    o1_5.thread_elements()[0]/=l1;
+    o1_5.thread_elements()[1]/=l1;
+    simdgroup_store(o1_5,out+base+(qr+8)*64+40,64);
+    o1_6.thread_elements()[0]/=l1;
+    o1_6.thread_elements()[1]/=l1;
+    simdgroup_store(o1_6,out+base+(qr+8)*64+48,64);
+    o1_7.thread_elements()[0]/=l1;
+    o1_7.thread_elements()[1]/=l1;
+    simdgroup_store(o1_7,out+base+(qr+8)*64+56,64);
 }
 ''')
 
@@ -106,5 +341,5 @@
 def kernel(q, k, v):
     qh, kh, vh = q.half(), k.half(), v.half()
     out = torch.empty_like(q)
-    _lib.flash(qh, kh, vh, out, threads=q.numel()//1024*128, group_size=128)
+    _lib.flash(qh, kh, vh, out, threads=q.numel()//4096*128, group_size=128)
     return out

```

</details>

`python bench.py`  
→ run 9 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 69.78 ms, kernel 10.97 ms, speedup 6.36x

- `python -c 'from pathlib import Path; Path('\\''kernel.py'\\'').write_text(\"import torch\\n\\n_lib = torch.mps.compile_shader(r'\\'''\\'''\\''\\n#include <me...`
- `python check.py`  
→ OK: relative error 3.76e-04 (allowed 2e-03)
### Bench run 11

`python bench.py`  
→ run 10 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 68.35 ms, kernel 4.51 ms, speedup 15.16x

> Explicitly unrolling the matrix operations lowered the time to 4.57 ms, a 15.42× speedup. Staging inputs in shared memory was slower. The tenth and final run is testing a larger key tile; I’ll retain the faster version and run the final correctness check.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 3.13e-04 (allowed 2e-03)
> Final kernel: **4.51 ms**, **15.16× speedup**. Final correctness check passed: relative error **3.13e-4**, below **2e-3**.
> 
> EDIT: kernel.py uses fused tiled Metal causal attention with half-precision operands, float32 accumulation, and register-resident matrices.

