"""Muon Newton-Schulz orthogonalization on MPS."""
import torch

_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void mm(device const half* a, device const half* b, device half* out,
               device const half* add, constant uint& K, constant uint& N,
               constant uint& mode,
               uint3 group [[threadgroup_position_in_grid]],
               uint tid [[thread_index_in_threadgroup]],
               uint sid [[simdgroup_index_in_threadgroup]]) {
    uint M=256, batch=group.z;
    if(mode==3 && batch>=10) { a=b; b=add; batch-=10; }
    threadgroup half ash[2048], bsh[2048];
    threadgroup float tmp[4096];
    simdgroup_float8x8 c00(0);
    simdgroup_float8x8 c01(0);
    simdgroup_float8x8 c02(0);
    simdgroup_float8x8 c03(0);
    simdgroup_float8x8 c10(0);
    simdgroup_float8x8 c11(0);
    simdgroup_float8x8 c12(0);
    simdgroup_float8x8 c13(0);
    simdgroup_float8x8 c20(0);
    simdgroup_float8x8 c21(0);
    simdgroup_float8x8 c22(0);
    simdgroup_float8x8 c23(0);
    simdgroup_float8x8 c30(0);
    simdgroup_float8x8 c31(0);
    simdgroup_float8x8 c32(0);
    simdgroup_float8x8 c33(0);
    for(uint base=0;base<K;base+=32) {
        for(uint t=tid;t<2048;t+=128) {
            ash[t]=a[batch*M*K+(group.y*64+t/32)*K+base+t%32];
            bsh[t]=mode==1 ? b[batch*N*K+(group.x*64+t%64)*K+base+t/64] : b[batch*K*N+(base+t/64)*N+group.x*64+t%64];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<32;k+=8) {
            simdgroup_half8x8 a0,b0;
            simdgroup_load(a0,ash+((sid/2)*32+0)*32+k,32);
            simdgroup_load(b0,bsh+k*64+(sid%2)*32+0,64);
            simdgroup_half8x8 a1,b1;
            simdgroup_load(a1,ash+((sid/2)*32+8)*32+k,32);
            simdgroup_load(b1,bsh+k*64+(sid%2)*32+8,64);
            simdgroup_half8x8 a2,b2;
            simdgroup_load(a2,ash+((sid/2)*32+16)*32+k,32);
            simdgroup_load(b2,bsh+k*64+(sid%2)*32+16,64);
            simdgroup_half8x8 a3,b3;
            simdgroup_load(a3,ash+((sid/2)*32+24)*32+k,32);
            simdgroup_load(b3,bsh+k*64+(sid%2)*32+24,64);
            simdgroup_multiply_accumulate(c00,a0,b0,c00);
            simdgroup_multiply_accumulate(c01,a0,b1,c01);
            simdgroup_multiply_accumulate(c02,a0,b2,c02);
            simdgroup_multiply_accumulate(c03,a0,b3,c03);
            simdgroup_multiply_accumulate(c10,a1,b0,c10);
            simdgroup_multiply_accumulate(c11,a1,b1,c11);
            simdgroup_multiply_accumulate(c12,a1,b2,c12);
            simdgroup_multiply_accumulate(c13,a1,b3,c13);
            simdgroup_multiply_accumulate(c20,a2,b0,c20);
            simdgroup_multiply_accumulate(c21,a2,b1,c21);
            simdgroup_multiply_accumulate(c22,a2,b2,c22);
            simdgroup_multiply_accumulate(c23,a2,b3,c23);
            simdgroup_multiply_accumulate(c30,a3,b0,c30);
            simdgroup_multiply_accumulate(c31,a3,b1,c31);
            simdgroup_multiply_accumulate(c32,a3,b2,c32);
            simdgroup_multiply_accumulate(c33,a3,b3,c33);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint off=(sid/2)*32*64+(sid%2)*32;
    simdgroup_store(c00,tmp+off+0,64);
    simdgroup_store(c01,tmp+off+8,64);
    simdgroup_store(c02,tmp+off+16,64);
    simdgroup_store(c03,tmp+off+24,64);
    simdgroup_store(c10,tmp+off+512,64);
    simdgroup_store(c11,tmp+off+520,64);
    simdgroup_store(c12,tmp+off+528,64);
    simdgroup_store(c13,tmp+off+536,64);
    simdgroup_store(c20,tmp+off+1024,64);
    simdgroup_store(c21,tmp+off+1032,64);
    simdgroup_store(c22,tmp+off+1040,64);
    simdgroup_store(c23,tmp+off+1048,64);
    simdgroup_store(c30,tmp+off+1536,64);
    simdgroup_store(c31,tmp+off+1544,64);
    simdgroup_store(c32,tmp+off+1552,64);
    simdgroup_store(c33,tmp+off+1560,64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint t=tid;t<4096;t+=128) {
        uint row=group.y*64+t/64,col=group.x*64+t%64;
        uint idx=group.z*M*N+row*N+col;
        float v=tmp[t];
        if(mode==2) v=2.0315f*v-4.775f*float(add[idx])+(row==col?3.4445f:0.f);
        out[idx]=half(v);
    }
}
kernel void normalize(device const float* g, device const float* norm,
                      device half* x, uint i [[thread_position_in_grid]]) {
    uint b=i/(256*704), k=i%(256*704);
    uint row=k/704, col=k%704;
    x[i]=half(g[b*256*704+col*256+row]/(norm[b]+1e-7f));
}
kernel void polynomial(device const half* a, device half* aa,
                       uint i [[thread_position_in_grid]]) {
    uint k=i%(256*256);
    aa[i]=half(2.0315f*float(aa[i])-4.775f*float(a[i])+(k/256==k%256 ? 3.4445f : 0.f));
}
''')


def kernel(g):
    if tuple(g.shape) == (10, 704, 256):
        norms=g.norm(dim=(-2,-1))
        X=torch.empty((10,256,704),dtype=torch.float16,device=g.device)
        _lib.normalize(g,norms,X,threads=X.numel())
        A=X@X.mT
        Q=None
        for iteration in range(5):
            B=torch.empty_like(A)
            _lib.mm(A,A,B,A,256,256,2,threads=(4*128,4,10),group_size=(128,1,1))
            if iteration==0:
                Q=B
                A=B@(A@B)
            elif iteration==4:
                Q=B@Q
            else:
                pair=torch.empty((20,256,256),dtype=torch.float16,device=g.device)
                _lib.mm(A,B,pair,Q,256,256,3,threads=(4*128,4,20),group_size=(128,1,1))
                Q=pair[10:]
                A=B@pair[:10]
        return (Q@X).mT
    transposed = g.size(-2) > g.size(-1)
    X = (g / (g.norm(dim=(-2, -1), keepdim=True) + 1e-7)).to(torch.float16)
    if transposed:
        X = X.mT
    for _ in range(5):
        A = X @ X.mT
        B = torch.baddbmm(A, A, A, beta=-4.7750, alpha=2.0315)
        X = torch.baddbmm(X, B, X, beta=3.4445)
    return X.mT if transposed else X
