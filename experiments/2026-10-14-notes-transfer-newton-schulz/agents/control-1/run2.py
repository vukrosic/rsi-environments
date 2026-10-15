import torch

_src = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void mm(device const half *a, device const half *b, device half *c,
               constant uint &op, uint3 gp [[threadgroup_position_in_grid]],
               uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half aa[1024];
    threadgroup half bb[1024];
    uint M=256, N=op==2?704:256, K=op==0?704:256;
    uint row=gp.y*32, col=gp.x*32, batch=gp.z;
    uint ar=(sg/2)*16, bc=(sg%2)*16;
    simdgroup_float8x8 s00(0),s01(0),s10(0),s11(0);
    for(uint k=0;k<K;k+=32) {
        for(uint i=t;i<1024;i+=128) {
            uint r=i/32, q=i%32;
            aa[i]=a[batch*M*K+(row+r)*K+k+q];
            if(op==0) bb[i]=b[batch*N*K+(col+q)*K+k+r];
            else bb[i]=b[batch*K*N+(k+r)*N+col+q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<32;j+=8) {
            simdgroup_half8x8 a0,a1,b0,b1;
            simdgroup_load(a0,aa+ar*32+j,32);
            simdgroup_load(a1,aa+(ar+8)*32+j,32);
            simdgroup_load(b0,bb+j*32+bc,32);
            simdgroup_load(b1,bb+j*32+bc+8,32);
            simdgroup_multiply_accumulate(s00,a0,b0,s00);
            simdgroup_multiply_accumulate(s01,a0,b1,s01);
            simdgroup_multiply_accumulate(s10,a1,b0,s10);
            simdgroup_multiply_accumulate(s11,a1,b1,s11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float cc[1024];
    simdgroup_store(s00,cc+ar*32+bc,32);
    simdgroup_store(s01,cc+ar*32+bc+8,32);
    simdgroup_store(s10,cc+(ar+8)*32+bc,32);
    simdgroup_store(s11,cc+(ar+8)*32+bc+8,32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=t;i<1024;i+=128) {
        uint r=row+i/32, q=col+i%32;
        float v=cc[i];
        if(op==1) v=2.0315f*v-4.7750f*float(a[batch*65536+r*256+q]);
        if(op==2) v+=3.4445f*float(b[batch*256*704+r*704+q]);
        c[batch*M*N+r*N+q]=half(v);
    }
}
'''
_lib=torch.mps.compile_shader(_src)

def kernel(g):
    X=(g.mT/(g.norm(dim=(-2,-1),keepdim=True)+1e-7)).half().contiguous()
    A=torch.empty((10,256,256),device=g.device,dtype=torch.float16)
    B=torch.empty_like(A)
    Y=torch.empty_like(X)
    for _ in range(5):
        _lib.mm(X,X,A,0,threads=(8*128,8,10),group_size=(128,1,1))
        _lib.mm(A,A,B,1,threads=(8*128,8,10),group_size=(128,1,1))
        _lib.mm(B,X,Y,2,threads=(22*128,8,10),group_size=(128,1,1))
        X,Y=Y,X
    return X.mT
