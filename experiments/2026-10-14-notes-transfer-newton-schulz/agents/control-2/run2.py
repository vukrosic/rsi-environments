import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
kernel void normalize(device const float* g, device half* x,
    uint t [[thread_index_in_threadgroup]], uint b [[threadgroup_position_in_grid]]) {
    threadgroup float sums[256];
    float s = 0.0f;
    for (uint i=t; i<180224; i+=256) { float v=g[b*180224+i]; s+=v*v; }
    sums[t]=s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d=128; d>0; d>>=1) {
        if (t<d) sums[t]+=sums[t+d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv=1.0f/(sqrt(sums[0])+1e-7f);
    for (uint i=t; i<180224; i+=256) {
        uint r=i/704, c=i%704;
        x[b*180224+i]=half(g[b*180224+c*256+r]*inv);
    }
}
kernel void polynomial(device const half* a, device half* aa,
    uint i [[thread_position_in_grid]]) {
    uint j=i%65536;
    aa[i]=half(-4.775f*float(a[i])+2.0315f*float(aa[i])+(j/256==j%256?3.4445f:0.0f));
}
""")


def kernel(g):
    X = torch.empty((10, 256, 704),device=g.device,dtype=torch.float16)
    _lib.normalize(g, X, threads=2560, group_size=256)
    for _ in range(5):
        A = X @ X.mT
        B = A @ A
        _lib.polynomial(A, B, threads=655360)
        X = B @ X
    return X.mT
