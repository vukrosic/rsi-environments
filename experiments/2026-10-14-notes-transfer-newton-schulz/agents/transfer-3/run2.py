"""Muon orthogonalization using reduced-precision matrix products."""
import torch

_source = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void mm(device const half *L, device const half *R, device half *O,
               constant uint &mode, uint3 tg [[threadgroup_position_in_grid]],
               uint tid [[thread_index_in_threadgroup]],
               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half ls[32*32];
    threadgroup half rs[32*32];
    const uint M=256, N=mode==2 ? 704 : 256, K=mode==0 ? 704 : 256;
    const uint row=tg.x*32, col=tg.y*32, batch=tg.z;
    const uint r=sg/2*16, c=sg%2*16;
    simdgroup_float8x8 c00(0), c01(0), c10(0), c11(0);
    for(uint k=0;k<K;k+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint rr=i/32, cc=i%32;
            ls[i]=L[batch*M*K+(row+rr)*K+k+cc];
            rs[i]=mode==0 ? R[batch*M*K+(col+cc)*K+k+rr] : R[batch*K*N+(k+rr)*N+col+cc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint j=0;j<32;j+=8) {
            simdgroup_half8x8 a0,a1,b0,b1;
            simdgroup_load(a0,ls+r*32+j,32);
            simdgroup_load(a1,ls+(r+8)*32+j,32);
            simdgroup_load(b0,rs+j*32+c,32);
            simdgroup_load(b1,rs+j*32+c+8,32);
            simdgroup_multiply_accumulate(c00,a0,b0,c00);
            simdgroup_multiply_accumulate(c01,a0,b1,c01);
            simdgroup_multiply_accumulate(c10,a1,b0,c10);
            simdgroup_multiply_accumulate(c11,a1,b1,c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float out[32*32];
    simdgroup_store(c00,out+r*32+c,32);
    simdgroup_store(c01,out+r*32+c+8,32);
    simdgroup_store(c10,out+(r+8)*32+c,32);
    simdgroup_store(c11,out+(r+8)*32+c+8,32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<1024;i+=128) {
        uint rr=row+i/32, cc=col+i%32;
        float val=out[i];
        if(mode==1) val=2.0315f*val-4.7750f*float(L[batch*65536+rr*256+cc])+(rr==cc ? 3.4445f : 0.0f);
        O[batch*M*N+rr*N+cc]=half(val);
    }
}
'''
_lib = torch.mps.compile_shader(_source)


def kernel(g):
    transposed = g.size(-2) > g.size(-1)
    X = g.mT if transposed else g
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).to(torch.float16).contiguous()
    A = torch.empty((10,256,256),device=g.device,dtype=torch.float16)
    B = torch.empty_like(A)
    Y = torch.empty_like(X)
    for _ in range(5):
        _lib.mm(X, X, A, 0, threads=(1024,8,10), group_size=(128,1,1))
        _lib.mm(A, A, B, 1, threads=(1024,8,10), group_size=(128,1,1))
        _lib.mm(B, X, Y, 2, threads=(1024,22,10), group_size=(128,1,1))
        X,Y = Y,X
    return X.mT if transposed else X
