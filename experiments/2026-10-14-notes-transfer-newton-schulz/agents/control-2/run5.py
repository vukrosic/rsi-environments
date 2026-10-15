import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
template<uint K, uint N, bool TRANS, bool POLY>
void mm(device const half* a, device const half* b, device half* out,
        uint3 group, uint sg, uint lane, threadgroup float* buf) {
    uint batch=group.z, row=group.x*32+(sg/2)*16, col=group.y*32+(sg%2)*16;
    a+=batch*256*K;
    b+=batch*(TRANS?N*K:K*N);
    out+=batch*256*N;
    simdgroup_float8x8 c00(0),c01(0),c10(0),c11(0);
    for(uint k=0;k<K;k+=8) {
        simdgroup_half8x8 a0,a1,b0,b1;
        simdgroup_load(a0,a+row*K+k,K);
        simdgroup_load(a1,a+(row+8)*K+k,K);
        if(TRANS) {
            simdgroup_load(b0,b+col*K+k,K,ulong2(0),true);
            simdgroup_load(b1,b+(col+8)*K+k,K,ulong2(0),true);
        } else {
            simdgroup_load(b0,b+k*N+col,N);
            simdgroup_load(b1,b+k*N+col+8,N);
        }
        simdgroup_multiply_accumulate(c00,a0,b0,c00);
        simdgroup_multiply_accumulate(c01,a0,b1,c01);
        simdgroup_multiply_accumulate(c10,a1,b0,c10);
        simdgroup_multiply_accumulate(c11,a1,b1,c11);
    }
    threadgroup float* local=buf+sg*256;
    simdgroup_store(c00,local,16);
    simdgroup_store(c01,local+8,16);
    simdgroup_store(c10,local+128,16);
    simdgroup_store(c11,local+136,16);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=lane;j<256;j+=32) {
        uint r=row+j/16,c=col+j%16;
        float v=local[j];
        if(POLY) v=2.0315f*v-4.775f*float(a[r*K+c])+(r==c?3.4445f:0.0f);
        out[r*N+c]=half(v);
        if(TRANS && group.x!=group.y) out[c*N+r]=half(v);
    }
}
kernel void gram(device const half* x,device half* a,uint3 g [[threadgroup_position_in_grid]],
 uint sg [[simdgroup_index_in_threadgroup]],uint l [[thread_index_in_simdgroup]]) {
    if(g.y<g.x) return;
    threadgroup float buf[1024]; mm<704,256,true,false>(x,x,a,g,sg,l,buf);
}
kernel void square(device const half* a,device half* b,uint3 g [[threadgroup_position_in_grid]],
 uint sg [[simdgroup_index_in_threadgroup]],uint l [[thread_index_in_simdgroup]]) {
    threadgroup float buf[1024]; mm<256,256,false,true>(a,a,b,g,sg,l,buf);
}
kernel void update(device const half* b,device const half* x,device half* y,uint3 g [[threadgroup_position_in_grid]],
 uint sg [[simdgroup_index_in_threadgroup]],uint l [[thread_index_in_simdgroup]]) {
    threadgroup float buf[1024]; mm<256,704,false,false>(b,x,y,g,sg,l,buf);
}
kernel void norm(device const float* g, device float* norms,
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
    if(t==0) norms[b]=1.0f/(sqrt(sums[0])+1e-7f);
}
kernel void normalize(device const float* g,device const float* norms,device half* x,
    uint3 block [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
    threadgroup half tile[32*33];
    for(uint j=t;j<1024;j+=256) {
        uint r=block.x*32+j/32,c=block.y*32+j%32;
        tile[(j/32)*33+j%32]=half(g[block.z*180224+r*256+c]*norms[block.z]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=t;j<1024;j+=256) {
        uint r=block.y*32+j/32,c=block.x*32+j%32;
        x[block.z*180224+r*704+c]=tile[(j%32)*33+j/32];
    }
}
""")


def kernel(g):
    X = torch.empty((10, 256, 704),device=g.device,dtype=torch.float16)
    norms = torch.empty((10,),device=g.device,dtype=torch.float32)
    _lib.norm(g, norms, threads=2560, group_size=256)
    _lib.normalize(g,norms,X,threads=(5632,8,10),group_size=(256,1,1))
    Y = torch.empty_like(X)
    A = torch.empty((10,256,256),device=g.device,dtype=torch.float16)
    B = torch.empty_like(A)
    for _ in range(5):
        _lib.gram(X,A,threads=(1024,8,10),group_size=(128,1,1))
        _lib.square(A,B,threads=(1024,8,10),group_size=(128,1,1))
        _lib.update(B,X,Y,threads=(1024,22,10),group_size=(128,1,1))
        X,Y=Y,X
    return X.mT
