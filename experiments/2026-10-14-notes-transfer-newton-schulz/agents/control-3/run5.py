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
    uint M=256;
    uint r=group.y*32+(sid/2)*16, c=group.x*32+(sid%2)*16;
    uint batch=group.z;
    if(mode==3 && batch>=10) {
        a=b;
        b=add;
        batch-=10;
    }
    threadgroup float tmp[1024];
    threadgroup half ash[1024], bsh[1024];
    simdgroup_float8x8 c00(0),c01(0),c10(0),c11(0);
    for(uint base=0;base<K;base+=32) {
        for(uint t=tid;t<1024;t+=128) {
            uint rr=t/32,cc=t%32;
            ash[t]=a[batch*M*K+(group.y*32+rr)*K+base+cc];
            bsh[t]=mode==1 ? b[batch*N*K+(group.x*32+cc)*K+base+rr] : b[batch*K*N+(base+rr)*N+group.x*32+cc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<32;k+=8) {
        simdgroup_half8x8 a0,a1,b0,b1;
        simdgroup_load(a0,ash+(sid/2)*16*32+k,32);
        simdgroup_load(a1,ash+((sid/2)*16+8)*32+k,32);
        simdgroup_load(b0,bsh+k*32+(sid%2)*16,32);
        simdgroup_load(b1,bsh+k*32+(sid%2)*16+8,32);
        simdgroup_multiply_accumulate(c00,a0,b0,c00);
        simdgroup_multiply_accumulate(c01,a0,b1,c01);
        simdgroup_multiply_accumulate(c10,a1,b0,c10);
        simdgroup_multiply_accumulate(c11,a1,b1,c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint off=(sid/2)*16*32+(sid%2)*16;
    simdgroup_store(c00,tmp+off,32);
    simdgroup_store(c01,tmp+off+8,32);
    simdgroup_store(c10,tmp+off+8*32,32);
    simdgroup_store(c11,tmp+off+8*32+8,32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint t=tid;t<1024;t+=128) {
        uint row=group.y*32+t/32,col=group.x*32+t%32;
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
            _lib.mm(A,A,B,A,256,256,2,threads=(8*128,8,10),group_size=(128,1,1))
            if iteration==0:
                Q=B
                A=B@(A@B)
            elif iteration==4:
                Q=B@Q
            else:
                pair=torch.empty((20,256,256),dtype=torch.float16,device=g.device)
                _lib.mm(A,B,pair,Q,256,256,3,threads=(8*128,8,20),group_size=(128,1,1))
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
