# Agent `lessons-r2-1`

**Prompt:** Make kernel() as fast as you can. LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what is still true from the notes you were given. At most 300 words, in words, no code.

13 min, 97,725 tokens. Unchanged code: 65.6 ms.

| Run | Kernel time | Speedup over the reference |
|---|---|---|
| run1 | 7.645 ms | 9.02x |
| run2 | 12.424 ms | 5.93x |
| run3 | 12.129 ms | 5.68x |
| run4 | 7.285 ms | 9.38x |
| run5 | 7.697 ms | 8.95x |
| run6 | 7.159 ms | 9.34x |
| run7 | 10.198 ms | 7.05x |
| run8 | 7.245 ms | 9.51x |
| run9 | 7.132 ms | 9.98x |
| run10 | 7.083 ms | 10.21x |
| final | 7.006 ms | 9.95x |

`final` is the last kernel.py, timed 5 times on new inputs (median).

## What it did, in order

> I’ll read the task files and prior notes, then test optimizations within the ten-run timing budget.

- `pwd && rg --files -g '"'!*__pycache__*'"' -g '"'!*.pyc`
### Bench run 1

<details><summary>Code change for run 1</summary>

```diff
--- before
+++ after
@@ -1,10 +1,89 @@
-"""Make kernel() faster. It must return the same values as reference() in reference.py."""
+"""Fused tiled causal attention for MPS."""
 import torch
 
+_BQ = 64
+_BK = 64
+_lib = torch.mps.compile_shader(r'''
+#include <metal_stdlib>
+#include <metal_simdgroup_matrix>
+using namespace metal;
+constant uint BQ = 64;
+constant uint BK = 64;
+kernel void attention(device const half* Q [[buffer(0)]],
+                      device const half* K [[buffer(1)]],
+                      device const half* V [[buffer(2)]],
+                      device float* O [[buffer(3)]],
+                      uint3 group [[threadgroup_position_in_grid]],
+                      uint lane [[thread_index_in_simdgroup]],
+                      uint sg [[simdgroup_index_in_threadgroup]]) {
+    threadgroup float scores[BQ * BK];
+    threadgroup half probs[BQ * BK];
+    uint row0 = group.x * BQ + sg * 8;
+    uint base = group.y * 1024 * 64;
+    uint r = (lane % 8) / 2 + (lane / 16) * 4;
+    uint c = lane % 2 + ((lane / 8) % 2) * 2;
+    float m = -INFINITY, l = 0.0f;
+    simdgroup_float8x8 out[8];
+    for (uint j=0;j<8;j++) out[j] = simdgroup_float8x8(0.0f);
+    for (uint kb=0;kb<group.x*BQ+BQ;kb+=BK) {
+        simdgroup_float8x8 acc[8];
+        for(uint j=0;j<8;j++) acc[j] = simdgroup_float8x8(0.0f);
+        for(uint d=0;d<64;d+=8) {
+            simdgroup_half8x8 a;
+            simdgroup_load(a,Q+base+row0*64+d,64);
+            for(uint j=0;j<BK/8;j++) {
+                simdgroup_half8x8 b;
+                simdgroup_load(b,K+base+(kb+j*8)*64+d,64,ulong2(0),true);
+                simdgroup_multiply_accumulate(acc[j],a,b,acc[j]);
+            }
+        }
+        for(uint j=0;j<BK/8;j++) simdgroup_store(acc[j],scores+sg*8*BK+j*8,BK);
+        simdgroup_barrier(mem_flags::mem_threadgroup);
+        float mx = -INFINITY;
+        for(uint j=c;j<BK;j+=4) {
+            float s = kb+j<=row0+r ? scores[(sg*8+r)*BK+j]*0.125f : -INFINITY;
+            scores[(sg*8+r)*BK+j]=s;
+            mx=max(mx,s);
+        }
+        mx=max(mx,simd_shuffle_xor(mx,1));
+        mx=max(mx,simd_shuffle_xor(mx,8));
+        float nm=max(m,mx), alpha=exp(m-nm), sum=0;
+        for(uint j=c;j<BK;j+=4) {
+            float p=exp(scores[(sg*8+r)*BK+j]-nm);
+            sum+=p;
+            probs[(sg*8+r)*BK+j]=half(p);
+        }
+        sum+=simd_shuffle_xor(sum,1);
+        sum+=simd_shuffle_xor(sum,8);
+        l=l*alpha+sum; m=nm;
+        for(uint j=0;j<8;j++) {
+            out[j].thread_elements()[0]*=alpha;
+            out[j].thread_elements()[1]*=alpha;
+        }
+        simdgroup_barrier(mem_flags::mem_threadgroup);
+        for(uint d=0;d<BK;d+=8) {
+            simdgroup_half8x8 a;
+            simdgroup_load(a,probs+sg*8*BK+d,BK);
+            for(uint j=0;j<8;j++) {
+                simdgroup_half8x8 b;
+                simdgroup_load(b,V+base+(kb+d)*64+j*8,64);
+                simdgroup_multiply_accumulate(out[j],a,b,out[j]);
+            }
+        }
+        simdgroup_barrier(mem_flags::mem_threadgroup);
+    }
+    for(uint j=0;j<8;j++) {
+        out[j].thread_elements()[0]/=l;
+        out[j].thread_elements()[1]/=l;
+        simdgroup_store(out[j],O+base+row0*64+j*8,64);
+    }
+}
+''')
 
 def kernel(q, k, v):
-    T = q.size(-2)
-    s = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
-    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
-    e = (s - s.max(-1, keepdim=True).values).exp()
-    return (e / e.sum(-1, keepdim=True)) @ v
+    if q.shape != (8, 8, 1024, 64):
+        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
+    out = torch.empty_like(q)
+    _lib.attention(q.half(), k.half(), v.half(), out,
+                   threads=(16*256,64,1), group_size=(256,1,1))
+    return out

```

</details>

`cat LESSONS.md kernel.py reference.py bench.py check.py`  
→     err = (got.float() - want).abs().max().item() / want.abs().max().item()  
→ print(f"{'OK' if got.shape == want.shape and err <= TOL else 'WRONG'}: relative error {err:.2e} (allowed {TOL:.0e})")

- `python -c 'import torch; print(torch.__version__); print(torch.mps.compile_shader.__doc__)`
> The notes favor a fused Metal kernel with half precision matrix inputs and float32 accumulation. I’ll start with that approach to avoid writing the full attention matrix to memory.

- `cat > kernel.py <<'PY` (exited 1 in 467ms:)
- edited `solution.py`
- `python check.py`  
→ WRONG: relative error 8.80e-01 (allowed 2e-03)
- `python -c 'import torch; s=torch.mps.compile_shader(\"#include <metal_stdlib>\\n#include <metal_simdgroup_matrix>\\nusing namespace metal; kernel void test(d...`
- `python -c 'from pathlib import Path; p=Path(\"kernel.py\");s=p.read_text().replace(\"uint r = lane / 4;\", \"uint r = (lane % 8) / 2 + (lane / 16) * 4;\").re...`
### Bench run 2

<details><summary>Code change for run 2</summary>

```diff
--- before
+++ after
@@ -16,8 +16,6 @@
                       uint3 group [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup float scores[BQ * BK];
-    threadgroup half probs[BQ * BK];
     uint row0 = group.x * BQ + sg * 8;
     uint base = group.y * 1024 * 64;
     uint r = (lane % 8) / 2 + (lane / 16) * 4;
@@ -37,21 +35,25 @@
                 simdgroup_multiply_accumulate(acc[j],a,b,acc[j]);
             }
         }
-        for(uint j=0;j<BK/8;j++) simdgroup_store(acc[j],scores+sg*8*BK+j*8,BK);
-        simdgroup_barrier(mem_flags::mem_threadgroup);
         float mx = -INFINITY;
-        for(uint j=c;j<BK;j+=4) {
-            float s = kb+j<=row0+r ? scores[(sg*8+r)*BK+j]*0.125f : -INFINITY;
-            scores[(sg*8+r)*BK+j]=s;
-            mx=max(mx,s);
+        for(uint j=0;j<BK/8;j++) {
+            for(uint e=0;e<2;e++) {
+                uint col=j*8+c*2+e;
+                float s=kb+col<=row0+r ? acc[j].thread_elements()[e]*0.125f : -INFINITY;
+                acc[j].thread_elements()[e]=s;
+                mx=max(mx,s);
+            }
         }
         mx=max(mx,simd_shuffle_xor(mx,1));
         mx=max(mx,simd_shuffle_xor(mx,8));
         float nm=max(m,mx), alpha=exp(m-nm), sum=0;
-        for(uint j=c;j<BK;j+=4) {
-            float p=exp(scores[(sg*8+r)*BK+j]-nm);
-            sum+=p;
-            probs[(sg*8+r)*BK+j]=half(p);
+        simdgroup_half8x8 probs[BK/8];
+        for(uint j=0;j<BK/8;j++) {
+            for(uint e=0;e<2;e++) {
+                float p=exp(acc[j].thread_elements()[e]-nm);
+                sum+=p;
+                probs[j].thread_elements()[e]=half(p);
+            }
         }
         sum+=simd_shuffle_xor(sum,1);
         sum+=simd_shuffle_xor(sum,8);
@@ -60,17 +62,14 @@
             out[j].thread_elements()[0]*=alpha;
             out[j].thread_elements()[1]*=alpha;
         }
-        simdgroup_barrier(mem_flags::mem_threadgroup);
         for(uint d=0;d<BK;d+=8) {
-            simdgroup_half8x8 a;
-            simdgroup_load(a,probs+sg*8*BK+d,BK);
+            simdgroup_half8x8 a=probs[d/8];
             for(uint j=0;j<8;j++) {
                 simdgroup_half8x8 b;
                 simdgroup_load(b,V+base+(kb+d)*64+j*8,64);
                 simdgroup_multiply_accumulate(out[j],a,b,out[j]);
             }
         }
-        simdgroup_barrier(mem_flags::mem_threadgroup);
     }
     for(uint j=0;j<8;j++) {
         out[j].thread_elements()[0]/=l;

```

</details>

`python bench.py`  
→ run 1 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 68.99 ms, kernel 7.64 ms, speedup 9.02x

- edited `solution.py`
- `python check.py`  
→ OK: relative error 4.94e-04 (allowed 2e-03)
> The first fused kernel passed and measured 7.64 ms, a 9.02× speedup. I’m testing a version that keeps scores and probabilities in SIMD registers to remove shared-memory traffic and synchronization.

### Bench run 3

<details><summary>Code change for run 3</summary>

```diff
--- before
+++ after
@@ -9,29 +9,37 @@
 using namespace metal;
 constant uint BQ = 64;
 constant uint BK = 64;
-kernel void attention(device const half* Q [[buffer(0)]],
-                      device const half* K [[buffer(1)]],
-                      device const half* V [[buffer(2)]],
+kernel void attention(device const float* Q [[buffer(0)]],
+                      device const float* K [[buffer(1)]],
+                      device const float* V [[buffer(2)]],
                       device float* O [[buffer(3)]],
                       uint3 group [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
+    threadgroup half qs[BQ*64];
+    threadgroup half ks[BK*64];
+    threadgroup half vs[BK*64];
+    uint tid=sg*32+lane;
     uint row0 = group.x * BQ + sg * 8;
     uint base = group.y * 1024 * 64;
     uint r = (lane % 8) / 2 + (lane / 16) * 4;
     uint c = lane % 2 + ((lane / 8) % 2) * 2;
+    for(uint i=tid;i<BQ*64;i+=256) qs[i]=half(Q[base+group.x*BQ*64+i]);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     float m = -INFINITY, l = 0.0f;
     simdgroup_float8x8 out[8];
     for (uint j=0;j<8;j++) out[j] = simdgroup_float8x8(0.0f);
     for (uint kb=0;kb<group.x*BQ+BQ;kb+=BK) {
+        for(uint i=tid;i<BK*64;i+=256) {ks[i]=half(K[base+kb*64+i]);vs[i]=half(V[base+kb*64+i]);}
+        threadgroup_barrier(mem_flags::mem_threadgroup);
         simdgroup_float8x8 acc[8];
         for(uint j=0;j<8;j++) acc[j] = simdgroup_float8x8(0.0f);
         for(uint d=0;d<64;d+=8) {
             simdgroup_half8x8 a;
-            simdgroup_load(a,Q+base+row0*64+d,64);
+            simdgroup_load(a,qs+sg*8*64+d,64);
             for(uint j=0;j<BK/8;j++) {
                 simdgroup_half8x8 b;
-                simdgroup_load(b,K+base+(kb+j*8)*64+d,64,ulong2(0),true);
+                simdgroup_load(b,ks+j*8*64+d,64,ulong2(0),true);
                 simdgroup_multiply_accumulate(acc[j],a,b,acc[j]);
             }
         }
@@ -66,10 +74,11 @@
             simdgroup_half8x8 a=probs[d/8];
             for(uint j=0;j<8;j++) {
                 simdgroup_half8x8 b;
-                simdgroup_load(b,V+base+(kb+d)*64+j*8,64);
+                simdgroup_load(b,vs+d*64+j*8,64);
                 simdgroup_multiply_accumulate(out[j],a,b,out[j]);
             }
         }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
     }
     for(uint j=0;j<8;j++) {
         out[j].thread_elements()[0]/=l;
@@ -83,6 +92,6 @@
     if q.shape != (8, 8, 1024, 64):
         return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
     out = torch.empty_like(q)
-    _lib.attention(q.half(), k.half(), v.half(), out,
+    _lib.attention(q, k, v, out,
                    threads=(16*256,64,1), group_size=(256,1,1))
     return out

```

</details>

`python bench.py`  
→ run 2 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 73.63 ms, kernel 12.42 ms, speedup 5.93x

- `python -c 'from pathlib import Path; p=Path(\"kernel.py\");s=p.read_text();s=s.replace(\"device const half* Q\", \"device const float* Q\").replace(\"device ...`
- `python check.py`  
→ OK: relative error 4.01e-04 (allowed 2e-03)
### Bench run 4

<details><summary>Code change for run 4</summary>

```diff
--- before
+++ after
@@ -19,6 +19,7 @@
     threadgroup half qs[BQ*64];
     threadgroup half ks[BK*64];
     threadgroup half vs[BK*64];
+    threadgroup half scores[BQ*BK];
     uint tid=sg*32+lane;
     uint row0 = group.x * BQ + sg * 8;
     uint base = group.y * 1024 * 64;
@@ -32,36 +33,33 @@
     for (uint kb=0;kb<group.x*BQ+BQ;kb+=BK) {
         for(uint i=tid;i<BK*64;i+=256) {ks[i]=half(K[base+kb*64+i]);vs[i]=half(V[base+kb*64+i]);}
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        simdgroup_float8x8 acc[8];
-        for(uint j=0;j<8;j++) acc[j] = simdgroup_float8x8(0.0f);
-        for(uint d=0;d<64;d+=8) {
-            simdgroup_half8x8 a;
-            simdgroup_load(a,qs+sg*8*64+d,64);
-            for(uint j=0;j<BK/8;j++) {
-                simdgroup_half8x8 b;
+        for(uint j=0;j<BK/8;j++) {
+            simdgroup_float8x8 acc(0.0f);
+            for(uint d=0;d<64;d+=8) {
+                simdgroup_half8x8 a,b;
+                simdgroup_load(a,qs+sg*8*64+d,64);
                 simdgroup_load(b,ks+j*8*64+d,64,ulong2(0),true);
-                simdgroup_multiply_accumulate(acc[j],a,b,acc[j]);
+                simdgroup_multiply_accumulate(acc,a,b,acc);
             }
+            simdgroup_half8x8 h;
+            h.thread_elements()[0]=half(acc.thread_elements()[0]*0.125f);
+            h.thread_elements()[1]=half(acc.thread_elements()[1]*0.125f);
+            simdgroup_store(h,scores+sg*8*BK+j*8,BK);
         }
+        simdgroup_barrier(mem_flags::mem_threadgroup);
         float mx = -INFINITY;
-        for(uint j=0;j<BK/8;j++) {
-            for(uint e=0;e<2;e++) {
-                uint col=j*8+c*2+e;
-                float s=kb+col<=row0+r ? acc[j].thread_elements()[e]*0.125f : -INFINITY;
-                acc[j].thread_elements()[e]=s;
-                mx=max(mx,s);
-            }
+        for(uint j=c;j<BK;j+=4) {
+            float s=kb+j<=row0+r ? float(scores[(sg*8+r)*BK+j]) : -INFINITY;
+            scores[(sg*8+r)*BK+j]=half(s);
+            mx=max(mx,s);
         }
         mx=max(mx,simd_shuffle_xor(mx,1));
         mx=max(mx,simd_shuffle_xor(mx,8));
         float nm=max(m,mx), alpha=exp(m-nm), sum=0;
-        simdgroup_half8x8 probs[BK/8];
-        for(uint j=0;j<BK/8;j++) {
-            for(uint e=0;e<2;e++) {
-                float p=exp(acc[j].thread_elements()[e]-nm);
-                sum+=p;
-                probs[j].thread_elements()[e]=half(p);
-            }
+        for(uint j=c;j<BK;j+=4) {
+            float p=exp(float(scores[(sg*8+r)*BK+j])-nm);
+            sum+=p;
+            scores[(sg*8+r)*BK+j]=half(p);
         }
         sum+=simd_shuffle_xor(sum,1);
         sum+=simd_shuffle_xor(sum,8);
@@ -70,9 +68,11 @@
             out[j].thread_elements()[0]*=alpha;
             out[j].thread_elements()[1]*=alpha;
         }
-        for(uint d=0;d<BK;d+=8) {
-            simdgroup_half8x8 a=probs[d/8];
-            for(uint j=0;j<8;j++) {
+        simdgroup_barrier(mem_flags::mem_threadgroup);
+        for(uint j=0;j<8;j++) {
+            for(uint d=0;d<BK;d+=8) {
+                simdgroup_half8x8 a;
+                simdgroup_load(a,scores+sg*8*BK+d,BK);
                 simdgroup_half8x8 b;
                 simdgroup_load(b,vs+d*64+j*8,64);
                 simdgroup_multiply_accumulate(out[j],a,b,out[j]);

```

</details>

`python bench.py`  
→ run 3 of 10 submitted.  
→ attention: correct (relative error 6.1e-04). reference 68.91 ms, kernel 12.13 ms, speedup 5.68x

- `cat kernel.py`
> Keeping scores in registers was slower: about 12 ms. I’ll move them back to shared memory and reduce the number of live matrix accumulators. The best result so far is still 7.64 ms.

- edited `solution.py`
- `python check.py`  
→ OK: relative error 6.14e-04 (allowed 2e-03)
### Bench run 5

<details><summary>Code change for run 5</summary>

```diff
--- before
+++ after
@@ -1,14 +1,14 @@
 """Fused tiled causal attention for MPS."""
 import torch
 
-_BQ = 64
-_BK = 64
+_BQ = 32
+_BK = 32
 _lib = torch.mps.compile_shader(r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-constant uint BQ = 64;
-constant uint BK = 64;
+constant uint BQ = 32;
+constant uint BK = 32;
 kernel void attention(device const float* Q [[buffer(0)]],
                       device const float* K [[buffer(1)]],
                       device const float* V [[buffer(2)]],
@@ -25,13 +25,13 @@
     uint base = group.y * 1024 * 64;
     uint r = (lane % 8) / 2 + (lane / 16) * 4;
     uint c = lane % 2 + ((lane / 8) % 2) * 2;
-    for(uint i=tid;i<BQ*64;i+=256) qs[i]=half(Q[base+group.x*BQ*64+i]);
+    for(uint i=tid;i<BQ*64;i+=128) qs[i]=half(Q[base+group.x*BQ*64+i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     float m = -INFINITY, l = 0.0f;
     simdgroup_float8x8 out[8];
     for (uint j=0;j<8;j++) out[j] = simdgroup_float8x8(0.0f);
     for (uint kb=0;kb<group.x*BQ+BQ;kb+=BK) {
-        for(uint i=tid;i<BK*64;i+=256) {ks[i]=half(K[base+kb*64+i]);vs[i]=half(V[base+kb*64+i]);}
+        for(uint i=tid;i<BK*64;i+=128) {ks[i]=half(K[base+kb*64+i]);vs[i]=half(V[base+kb*64+i]);}
         threadgroup_barrier(mem_flags::mem_threadgroup);
         for(uint j=0;j<BK/8;j++) {
             simdgroup_float8x8 acc(0.0f);
@@ -93,5 +93,5 @@
         return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
     out = torch.empty_like(q)
     _lib.attention(q, k, v, out,
-                   threads=(16*256,64,1), group_size=(256,1,1))
+                   threads=(32*128,64,1), group_size=(128,1,1))
     return out

```

</details>

`python bench.py`  


> The shared-memory version passed with relative error 0.00061 against the 0.002 limit. Its timing is pending. I’ve prepared a smaller tile variant to test whether it improves GPU occupancy.
>  succeeded in 50091ms:
> run 4 of 10 submitted.
> attention: correct (relative error 6.8e-04). reference 68.32 ms, kernel 7.29 ms, speedup 9.38x

- `python check.py`  
→ OK: relative error 6.06e-04 (allowed 2e-03)
### Bench run 6

<details><summary>Code change for run 6</summary>

```diff
--- before
+++ after
@@ -1,14 +1,14 @@
 """Fused tiled causal attention for MPS."""
 import torch
 
-_BQ = 32
-_BK = 32
+_BQ = 64
+_BK = 64
 _lib = torch.mps.compile_shader(r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-constant uint BQ = 32;
-constant uint BK = 32;
+constant uint BQ = 64;
+constant uint BK = 64;
 kernel void attention(device const float* Q [[buffer(0)]],
                       device const float* K [[buffer(1)]],
                       device const float* V [[buffer(2)]],
@@ -25,14 +25,16 @@
     uint base = group.y * 1024 * 64;
     uint r = (lane % 8) / 2 + (lane / 16) * 4;
     uint c = lane % 2 + ((lane / 8) % 2) * 2;
-    for(uint i=tid;i<BQ*64;i+=128) qs[i]=half(Q[base+group.x*BQ*64+i]);
+    for(uint i=tid;i<BQ*64;i+=256) qs[i]=half(Q[base+group.x*BQ*64+i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     float m = -INFINITY, l = 0.0f;
     simdgroup_float8x8 out[8];
+    #pragma unroll
     for (uint j=0;j<8;j++) out[j] = simdgroup_float8x8(0.0f);
     for (uint kb=0;kb<group.x*BQ+BQ;kb+=BK) {
-        for(uint i=tid;i<BK*64;i+=128) {ks[i]=half(K[base+kb*64+i]);vs[i]=half(V[base+kb*64+i]);}
+        for(uint i=tid;i<BK*64;i+=256) {ks[i]=half(K[base+kb*64+i]);vs[i]=half(V[base+kb*64+i]);}
         threadgroup_barrier(mem_flags::mem_threadgroup);
+        #pragma unroll
         for(uint j=0;j<BK/8;j++) {
             simdgroup_float8x8 acc(0.0f);
             for(uint d=0;d<64;d+=8) {
@@ -64,11 +66,13 @@
         sum+=simd_shuffle_xor(sum,1);
         sum+=simd_shuffle_xor(sum,8);
         l=l*alpha+sum; m=nm;
+        #pragma unroll
         for(uint j=0;j<8;j++) {
             out[j].thread_elements()[0]*=alpha;
             out[j].thread_elements()[1]*=alpha;
         }
         simdgroup_barrier(mem_flags::mem_threadgroup);
+        #pragma unroll
         for(uint j=0;j<8;j++) {
             for(uint d=0;d<BK;d+=8) {
                 simdgroup_half8x8 a;
@@ -80,6 +84,7 @@
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
+    #pragma unroll
     for(uint j=0;j<8;j++) {
         out[j].thread_elements()[0]/=l;
         out[j].thread_elements()[1]/=l;
@@ -93,5 +98,5 @@
         return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
     out = torch.empty_like(q)
     _lib.attention(q, k, v, out,
-                   threads=(32*128,64,1), group_size=(128,1,1))
+                   threads=(16*256,64,1), group_size=(256,1,1))
     return out

```

</details>

`python bench.py`  


- `cat kernel.py`
- `python -c 'from pathlib import Path; lines=Path(\"queue/run4.py\").read_text().splitlines();o=[]`
### Bench run 7

<details><summary>Code change for run 7</summary>

```diff
--- before
+++ after
@@ -18,23 +18,19 @@
                       uint sg [[simdgroup_index_in_threadgroup]]) {
     threadgroup half qs[BQ*64];
     threadgroup half ks[BK*64];
-    threadgroup half vs[BK*64];
-    threadgroup half scores[BQ*BK];
+    threadgroup float scores[BQ*BK];
     uint tid=sg*32+lane;
     uint row0 = group.x * BQ + sg * 8;
     uint base = group.y * 1024 * 64;
     uint r = (lane % 8) / 2 + (lane / 16) * 4;
     uint c = lane % 2 + ((lane / 8) % 2) * 2;
-    for(uint i=tid;i<BQ*64;i+=256) qs[i]=half(Q[base+group.x*BQ*64+i]);
-    threadgroup_barrier(mem_flags::mem_threadgroup);
     float m = -INFINITY, l = 0.0f;
     simdgroup_float8x8 out[8];
-    #pragma unroll
     for (uint j=0;j<8;j++) out[j] = simdgroup_float8x8(0.0f);
     for (uint kb=0;kb<group.x*BQ+BQ;kb+=BK) {
-        for(uint i=tid;i<BK*64;i+=256) {ks[i]=half(K[base+kb*64+i]);vs[i]=half(V[base+kb*64+i]);}
+        for(uint i=tid;i<BQ*64;i+=256) qs[i]=half(Q[base+group.x*BQ*64+i]);
+        for(uint i=tid;i<BK*64;i+=256) ks[i]=half(K[base+kb*64+i]);
         threadgroup_barrier(mem_flags::mem_threadgroup);
-        #pragma unroll
         for(uint j=0;j<BK/8;j++) {
             simdgroup_float8x8 acc(0.0f);
             for(uint d=0;d<64;d+=8) {
@@ -43,16 +39,16 @@
                 simdgroup_load(b,ks+j*8*64+d,64,ulong2(0),true);
                 simdgroup_multiply_accumulate(acc,a,b,acc);
             }
-            simdgroup_half8x8 h;
-            h.thread_elements()[0]=half(acc.thread_elements()[0]*0.125f);
-            h.thread_elements()[1]=half(acc.thread_elements()[1]*0.125f);
-            simdgroup_store(h,scores+sg*8*BK+j*8,BK);
+            acc.thread_elements()[0]*=0.125f;
+            acc.thread_elements()[1]*=0.125f;
+            simdgroup_store(acc,scores+sg*8*BK+j*8,BK);
         }
-        simdgroup_barrier(mem_flags::mem_threadgroup);
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for(uint i=tid;i<BK*64;i+=256) qs[i]=half(V[base+kb*64+i]);
         float mx = -INFINITY;
         for(uint j=c;j<BK;j+=4) {
             float s=kb+j<=row0+r ? float(scores[(sg*8+r)*BK+j]) : -INFINITY;
-            scores[(sg*8+r)*BK+j]=half(s);
+            scores[(sg*8+r)*BK+j]=s;
             mx=max(mx,s);
         }
         mx=max(mx,simd_shuffle_xor(mx,1));
@@ -61,30 +57,27 @@
         for(uint j=c;j<BK;j+=4) {
             float p=exp(float(scores[(sg*8+r)*BK+j])-nm);
             sum+=p;
-            scores[(sg*8+r)*BK+j]=half(p);
+            ks[(sg*8+r)*BK+j]=half(p);
         }
         sum+=simd_shuffle_xor(sum,1);
         sum+=simd_shuffle_xor(sum,8);
         l=l*alpha+sum; m=nm;
-        #pragma unroll
         for(uint j=0;j<8;j++) {
             out[j].thread_elements()[0]*=alpha;
             out[j].thread_elements()[1]*=alpha;
         }
-        simdgroup_barrier(mem_flags::mem_threadgroup);
-        #pragma unroll
+        threadgroup_barrier(mem_flags::mem_threadgroup);
         for(uint j=0;j<8;j++) {
             for(uint d=0;d<BK;d+=8) {
                 simdgroup_half8x8 a;
-                simdgroup_load(a,scores+sg*8*BK+d,BK);
+                simdgroup_load(a,ks+sg*8*BK+d,BK);
                 simdgroup_half8x8 b;
-                simdgroup_load(b,vs+d*64+j*8,64);
+                simdgroup_load(b,qs+d*64+j*8,64);
                 simdgroup_multiply_accumulate(out[j],a,b,out[j]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
-    #pragma unroll
     for(uint j=0;j<8;j++) {
         out[j].thread_elements()[0]/=l;
         out[j].thread_elements()[1]/=l;

```

</details>

`python bench.py`  


- `python check.py`  
→ OK: relative error 4.16e-04 (allowed 2e-03)
### Bench run 8

<details><summary>Code change for run 8</summary>

```diff
--- before
+++ after
@@ -18,37 +18,42 @@
                       uint sg [[simdgroup_index_in_threadgroup]]) {
     threadgroup half qs[BQ*64];
     threadgroup half ks[BK*64];
-    threadgroup float scores[BQ*BK];
+    threadgroup half vs[BK*64];
+    threadgroup half scores[BQ*BK];
     uint tid=sg*32+lane;
     uint row0 = group.x * BQ + sg * 8;
     uint base = group.y * 1024 * 64;
     uint r = (lane % 8) / 2 + (lane / 16) * 4;
     uint c = lane % 2 + ((lane / 8) % 2) * 2;
+    for(uint i=tid;i<BQ*64;i+=256) qs[i]=half(Q[base+group.x*BQ*64+i]);
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     float m = -INFINITY, l = 0.0f;
     simdgroup_float8x8 out[8];
+    #pragma unroll
     for (uint j=0;j<8;j++) out[j] = simdgroup_float8x8(0.0f);
     for (uint kb=0;kb<group.x*BQ+BQ;kb+=BK) {
-        for(uint i=tid;i<BQ*64;i+=256) qs[i]=half(Q[base+group.x*BQ*64+i]);
-        for(uint i=tid;i<BK*64;i+=256) ks[i]=half(K[base+kb*64+i]);
+        for(uint i=tid;i<BK*64;i+=256) {ks[i]=half(K[base+kb*64+i]);vs[i]=half(V[base+kb*64+i]);}
         threadgroup_barrier(mem_flags::mem_threadgroup);
+        #pragma unroll
         for(uint j=0;j<BK/8;j++) {
             simdgroup_float8x8 acc(0.0f);
+            #pragma unroll
             for(uint d=0;d<64;d+=8) {
                 simdgroup_half8x8 a,b;
                 simdgroup_load(a,qs+sg*8*64+d,64);
                 simdgroup_load(b,ks+j*8*64+d,64,ulong2(0),true);
                 simdgroup_multiply_accumulate(acc,a,b,acc);
             }
-            acc.thread_elements()[0]*=0.125f;
-            acc.thread_elements()[1]*=0.125f;
-            simdgroup_store(acc,scores+sg*8*BK+j*8,BK);
+            simdgroup_half8x8 h;
+            h.thread_elements()[0]=half(acc.thread_elements()[0]*0.125f);
+            h.thread_elements()[1]=half(acc.thread_elements()[1]*0.125f);
+            simdgroup_store(h,scores+sg*8*BK+j*8,BK);
         }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
-        for(uint i=tid;i<BK*64;i+=256) qs[i]=half(V[base+kb*64+i]);
+        simdgroup_barrier(mem_flags::mem_threadgroup);
         float mx = -INFINITY;
         for(uint j=c;j<BK;j+=4) {
             float s=kb+j<=row0+r ? float(scores[(sg*8+r)*BK+j]) : -INFINITY;
-            scores[(sg*8+r)*BK+j]=s;
+            scores[(sg*8+r)*BK+j]=half(s);
             mx=max(mx,s);
         }
         mx=max(mx,simd_shuffle_xor(mx,1));
@@ -57,27 +62,31 @@
         for(uint j=c;j<BK;j+=4) {
             float p=exp(float(scores[(sg*8+r)*BK+j])-nm);
             sum+=p;
-            ks[(sg*8+r)*BK+j]=half(p);
+            scores[(sg*8+r)*BK+j]=half(p);
         }
         sum+=simd_shuffle_xor(sum,1);
         sum+=simd_shuffle_xor(sum,8);
         l=l*alpha+sum; m=nm;
+        #pragma unroll
         for(uint j=0;j<8;j++) {
             out[j].thread_elements()[0]*=alpha;
             out[j].thread_elements()[1]*=alpha;
         }
-        threadgroup_barrier(mem_flags::mem_threadgroup);
+        simdgroup_barrier(mem_flags::mem_threadgroup);
+        #pragma unroll
         for(uint j=0;j<8;j++) {
+            #pragma unroll
             for(uint d=0;d<BK;d+=8) {
                 simdgroup_half8x8 a;
-                simdgroup_load(a,ks+sg*8*BK+d,BK);
+                simdgroup_load(a,scores+sg*8*BK+d,BK);
                 simdgroup_half8x8 b;
-                simdgroup_load(b,qs+d*64+j*8,64);
+                simdgroup_load(b,vs+d*64+j*8,64);
                 simdgroup_multiply_accumulate(out[j],a,b,out[j]);
             }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
+    #pragma unroll
     for(uint j=0;j<8;j++) {
         out[j].thread_elements()[0]/=l;
         out[j].thread_elements()[1]/=l;

```

</details>

`python bench.py`  


> Unrolling the output-matrix loops improved the best time slightly to 7.16 ms. The float32-score version passed correctness and is being timed. Three benchmark submissions remain.

- `python check.py`  
→ OK: relative error 3.98e-04 (allowed 2e-03)
### Bench run 9

<details><summary>Code change for run 9</summary>

```diff
--- before
+++ after
@@ -1,14 +1,14 @@
 """Fused tiled causal attention for MPS."""
 import torch
 
-_BQ = 64
-_BK = 64
+_BQ = 128
+_BK = 32
 _lib = torch.mps.compile_shader(r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
-constant uint BQ = 64;
-constant uint BK = 64;
+constant uint BQ = 128;
+constant uint BK = 32;
 kernel void attention(device const float* Q [[buffer(0)]],
                       device const float* K [[buffer(1)]],
                       device const float* V [[buffer(2)]],
@@ -25,19 +25,18 @@
     uint base = group.y * 1024 * 64;
     uint r = (lane % 8) / 2 + (lane / 16) * 4;
     uint c = lane % 2 + ((lane / 8) % 2) * 2;
-    for(uint i=tid;i<BQ*64;i+=256) qs[i]=half(Q[base+group.x*BQ*64+i]);
+    for(uint i=tid;i<BQ*64;i+=512) qs[i]=half(Q[base+group.x*BQ*64+i]);
     threadgroup_barrier(mem_flags::mem_threadgroup);
     float m = -INFINITY, l = 0.0f;
     simdgroup_float8x8 out[8];
     #pragma unroll
     for (uint j=0;j<8;j++) out[j] = simdgroup_float8x8(0.0f);
     for (uint kb=0;kb<group.x*BQ+BQ;kb+=BK) {
-        for(uint i=tid;i<BK*64;i+=256) {ks[i]=half(K[base+kb*64+i]);vs[i]=half(V[base+kb*64+i]);}
+        for(uint i=tid;i<BK*64;i+=512) {ks[i]=half(K[base+kb*64+i]);vs[i]=half(V[base+kb*64+i]);}
         threadgroup_barrier(mem_flags::mem_threadgroup);
         #pragma unroll
         for(uint j=0;j<BK/8;j++) {
             simdgroup_float8x8 acc(0.0f);
-            #pragma unroll
             for(uint d=0;d<64;d+=8) {
                 simdgroup_half8x8 a,b;
                 simdgroup_load(a,qs+sg*8*64+d,64);
@@ -75,7 +74,6 @@
         simdgroup_barrier(mem_flags::mem_threadgroup);
         #pragma unroll
         for(uint j=0;j<8;j++) {
-            #pragma unroll
             for(uint d=0;d<BK;d+=8) {
                 simdgroup_half8x8 a;
                 simdgroup_load(a,scores+sg*8*BK+d,BK);
@@ -100,5 +98,5 @@
         return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
     out = torch.empty_like(q)
     _lib.attention(q, k, v, out,
-                   threads=(16*256,64,1), group_size=(256,1,1))
+                   threads=(8*512,64,1), group_size=(512,1,1))
     return out

```

</details>

`python bench.py`  


> Reusing query storage for values took 10.20 ms, so I’m keeping the separate shared buffers. The remaining tests target loop overhead and larger query tiles that share key and value loads across more queries.
>  succeeded in 65147ms:
> run 8 of 10 submitted.
> attention: correct (relative error 6.8e-04). reference 68.90 ms, kernel 7.24 ms, speedup 9.51x

- `python check.py`  
→ OK: relative error 5.19e-04 (allowed 2e-03)
### Bench run 10

<details><summary>Code change for run 10</summary>

```diff
--- before
+++ after
@@ -2,13 +2,13 @@
 import torch
 
 _BQ = 128
-_BK = 32
+_BK = 64
 _lib = torch.mps.compile_shader(r'''
 #include <metal_stdlib>
 #include <metal_simdgroup_matrix>
 using namespace metal;
 constant uint BQ = 128;
-constant uint BK = 32;
+constant uint BK = 64;
 kernel void attention(device const float* Q [[buffer(0)]],
                       device const float* K [[buffer(1)]],
                       device const float* V [[buffer(2)]],
@@ -16,7 +16,6 @@
                       uint3 group [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
-    threadgroup half qs[BQ*64];
     threadgroup half ks[BK*64];
     threadgroup half vs[BK*64];
     threadgroup half scores[BQ*BK];
@@ -25,8 +24,13 @@
     uint base = group.y * 1024 * 64;
     uint r = (lane % 8) / 2 + (lane / 16) * 4;
     uint c = lane % 2 + ((lane / 8) % 2) * 2;
-    for(uint i=tid;i<BQ*64;i+=512) qs[i]=half(Q[base+group.x*BQ*64+i]);
-    threadgroup_barrier(mem_flags::mem_threadgroup);
+    simdgroup_half8x8 query[8];
+    #pragma unroll
+    for(uint d=0;d<8;d++) {
+        half2 h=half2(*reinterpret_cast<device const float2*>(Q+base+(row0+r)*64+d*8+c*2));
+        query[d].thread_elements()[0]=h.x;
+        query[d].thread_elements()[1]=h.y;
+    }
     float m = -INFINITY, l = 0.0f;
     simdgroup_float8x8 out[8];
     #pragma unroll
@@ -37,9 +41,9 @@
         #pragma unroll
         for(uint j=0;j<BK/8;j++) {
             simdgroup_float8x8 acc(0.0f);
+            #pragma unroll
             for(uint d=0;d<64;d+=8) {
-                simdgroup_half8x8 a,b;
-                simdgroup_load(a,qs+sg*8*64+d,64);
+                simdgroup_half8x8 a=query[d/8],b;
                 simdgroup_load(b,ks+j*8*64+d,64,ulong2(0),true);
                 simdgroup_multiply_accumulate(acc,a,b,acc);
             }

```

</details>

`python bench.py`  


- `python -c 'from pathlib import Path;p=Path(\"kernel.py\");s=p.read_text().replace(\"device const half* Q\", \"device const float* Q\").replace(\"    for(uint...`
- `python check.py`  
→ OK: relative error 5.70e-04 (allowed 2e-03)
> The 128-query, 32-key tile measured 7.13 ms, about 10× faster than its simultaneous reference run. The final test keeps queries in registers, freeing shared memory for a larger key tile. It passed correctness.

### Bench run 11

`python bench.py`  
→ run 10 of 10 submitted.  
→ attention: correct (relative error 6.8e-04). reference 72.30 ms, kernel 7.08 ms, speedup 10.21x

- `cat kernel.py`
> The final variant measured 7.08 ms, a 10.21× speedup, with benchmark relative error 0.00068. I’m keeping it, updating the notes, and running the final correctness check.

- edited `solution.py`
### Bench run 12

`python -c 'from pathlib import Path; Path('\\''LESSONS.md'\\'').write_text(\"The workload is contiguous float32 causal attention: batch eight, eight heads, sequence length 1024, head size 64. Only benchmark through bench.py; shared GPU load varies. This round used all ten runs; reference times ranged from 66.89 to 73.63 milliseconds.\\n\\nThe final kernel measured 7.08 milliseconds against 72.30 milliseconds, a 10.21 times speedup. Benchmark relative error was 0.00068 against the 0.002 tolerance.\\n\\nUse a fused Metal shader compiled once at import. The final configuration processes 128 queries and 64 keys per tile with 512 threads. Each SIMD group handles eight query rows. Load queries directly from float inputs into half matrix registers and retain them across key tiles. Cooperatively stage keys and values as half in shared memory. Accumulate matrix products and outputs in float32. Store scaled scores as half, overwrite them with half probabilities, and maintain stable online softmax maxima and denominators in float32. Skip wholly future tiles.\\n\\nMeasured alternatives, in milliseconds: direct half buffers with shared float scores, 7.64; register scores and probabilities, 12.42; staging inputs with register scores, 12.13; shared half scores with 64 by 64 tiles, 7.29; 32 by 32 tiles, 7.70; unrolled output matrix loops, 7.16; float scores with query storage reused for values, 10.20; additionally unrolled reduction loops, 7.24; 128 by 32 tiles, 7.13.\\n\\nEarlier rounds reported built-in causal attention around 17 milliseconds in float32 and 10.5 milliseconds with half inputs. A different fused float-score implementation reached 4.99 milliseconds. Preserve float32 matrix accumulation: half dot-product accumulation and repeated half output accumulation previously exceeded tolerance. Half score storage after float accumulation passed comfortably here. Avoid judging tiny timing differences confidently.\\n\"); print('\\''Lesson words:'\\'', len(Path('\\''LESSONS.md'\\'').read_text().split()))`  
→ Lesson words: 264

- `python check.py`  
→ OK: relative error 5.07e-04 (allowed 2e-03)
> Achieved **7.08 ms**, a **10.21× benchmark speedup**. Final check passed with relative error **0.000507**. Updated LESSONS.md.
> 
> EDIT: kernel.py now computes fused tiled causal attention in Metal using half precision operands and float32 accumulation.

