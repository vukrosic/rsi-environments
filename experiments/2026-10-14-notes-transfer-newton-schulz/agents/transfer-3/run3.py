"""Muon orthogonalization using reduced-precision matrix products."""
import torch

def _make_shader():
    lines = [r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void mm(device const half *L, device const half *R, device half *O,
               constant uint &mode, uint3 tg [[threadgroup_position_in_grid]],
               uint tid [[thread_index_in_threadgroup]],
               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half ls[64*32];
    threadgroup half rs[32*64];
    const uint M=256, N=mode==2 ? 704 : 256, K=mode==0 ? 704 : 256;
    const uint row=tg.x*64, col=tg.y*64, batch=tg.z;
    const uint r=sg/4*32, c=sg%4*16;
''']
    for i in range(4):
        for j in range(2):
            lines.append(f"simdgroup_float8x8 z{i}{j}(0);")
    lines.append(r'''
    for(uint k=0;k<K;k+=32) {
        for(uint i=tid;i<2048;i+=256) {
            uint rr=i/32, cc=i%32;
            ls[i]=L[batch*M*K+(row+rr)*K+k+cc];
            rr=i/64; cc=i%64;
            rs[i]=mode==0 ? R[batch*M*K+(col+cc)*K+k+rr] : R[batch*K*N+(k+rr)*N+col+cc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
''')
    for k in range(0,32,8):
        lines.append("{")
        for i in range(4):
            lines.append(f"simdgroup_half8x8 a{i}; simdgroup_load(a{i},ls+(r+{i*8})*32+{k},32);")
        for j in range(2):
            lines.append(f"simdgroup_half8x8 b{j}; simdgroup_load(b{j},rs+{k}*64+c+{j*8},64);")
        for i in range(4):
            for j in range(2):
                lines.append(f"simdgroup_multiply_accumulate(z{i}{j},a{i},b{j},z{i}{j});")
        lines.append("}")
    lines.append("threadgroup_barrier(mem_flags::mem_threadgroup); } threadgroup float out[64*64];")
    for i in range(4):
        for j in range(2):
            lines.append(f"simdgroup_store(z{i}{j},out+(r+{i*8})*64+c+{j*8},64);")
    lines.append(r'''
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<4096;i+=256) {
        uint rr=row+i/64, cc=col+i%64;
        float val=out[i];
        if(mode==1) val=2.0315f*val-4.7750f*float(L[batch*65536+rr*256+cc])+(rr==cc ? 3.4445f : 0.0f);
        O[batch*M*N+rr*N+cc]=half(val);
    }
}
''')
    return "\n".join(lines)
_lib = torch.mps.compile_shader(_make_shader())


def kernel(g):
    transposed = g.size(-2) > g.size(-1)
    X = g.mT if transposed else g
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).to(torch.float16).contiguous()
    A = torch.empty((10,256,256),device=g.device,dtype=torch.float16)
    B = torch.empty_like(A)
    Y = torch.empty_like(X)
    for _ in range(5):
        _lib.mm(X, X, A, 0, threads=(1024,4,10), group_size=(256,1,1))
        _lib.mm(A, A, B, 1, threads=(1024,4,10), group_size=(256,1,1))
        _lib.mm(B, X, Y, 2, threads=(1024,11,10), group_size=(256,1,1))
        X,Y = Y,X
    return X.mT if transposed else X
