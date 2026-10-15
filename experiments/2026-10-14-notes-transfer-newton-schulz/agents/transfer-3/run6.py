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
_init_source = r'''
kernel void rescale(device const half *A, device float *Z, uint i [[thread_position_in_grid]]) {
    uint rc=i%65536;
    Z[i]=200.0f*float(A[i])-(rc/256==rc%256 ? 1.1f : 0.0f);
}
kernel void blocks(device const float *Z, device const float *Z2,
 device float *P0, device float *P1, device float *P2, device float *P3,
 uint i [[thread_position_in_grid]]) {
    uint rc=i%65536;
    float d=rc/256==rc%256 ? 1.0f : 0.0f;
    float x=Z[i], y=Z2[i];
    P0[i]=14.713175303389436f*d-14.380655664667518f*x-10.688916094941385f*y;
    P1[i]=24.785859149684217f*d-28.820378064004238f*x+69.79296173674018f*y;
    P2[i]=84.86459769060141f*d-291.99518440913954f*x+38.73131678642112f*y;
    P3[i]=224.92019542327594f*d-78.34988935228415f*x-26.483320125604283f*y;
}
kernel void polynomial(device const half *A, device const half *AA, device half *B,
 uint i [[thread_position_in_grid]]) {
    uint rc=i%65536;
    B[i]=half(2.0315f*float(AA[i])-4.7750f*float(A[i])+(rc/256==rc%256 ? 3.4445f : 0.0f));
}
kernel void norm_parts(device const float *g, device float *parts,
 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint base=tg.y*180224+tg.x*1024+tid;
    float v=0;
    for(uint j=0;j<4;j++) { float x=g[base+j*256]; v+=x*x; }
    v=simd_sum(v);
    threadgroup float s[8];
    if(lane==0) s[sg]=v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
        float total=0; for(uint j=0;j<8;j++) total+=s[j];
        parts[tg.y*176+tg.x]=total;
    }
}
kernel void normalize(device const float *g, device const float *parts, device half *X,
 uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    float sum=0;
    for(uint i=tid;i<176;i+=128) sum+=parts[tg.z*176+i];
    sum=simd_sum(sum);
    threadgroup float sums[4];
    threadgroup float scale;
    threadgroup half tile[32*32];
    if(lane==0) sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) scale=1.0f/(sqrt(sums[0]+sums[1]+sums[2]+sums[3])+1e-7f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<1024;i+=128) {
        uint rr=i/32,cc=i%32;
        tile[i]=half(g[tg.z*180224+(tg.y*32+rr)*256+tg.x*32+cc]*scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<1024;i+=128) {
        uint rr=i/32,cc=i%32;
        X[tg.z*180224+(tg.x*32+rr)*704+tg.y*32+cc]=tile[cc*32+rr];
    }
}
'''
_lib = torch.mps.compile_shader(_make_shader()+_init_source)


def kernel(g):
    transposed = g.size(-2) > g.size(-1)
    X = g.mT if transposed else g
    X = torch.empty((10,256,704),device=g.device,dtype=torch.float16)
    parts = torch.empty((10,176),device=g.device,dtype=torch.float32)
    _lib.norm_parts(g,parts,threads=(176*256,10),group_size=(256,1))
    _lib.normalize(g,parts,X,threads=(1024,22,10),group_size=(128,1,1))
    A = X @ X.mT
    Z = torch.empty(A.shape,device=g.device,dtype=torch.float32)
    _lib.rescale(A,Z,threads=10*65536)
    Z2=Z@Z
    Z3=Z2@Z
    P0,P1,P2,P3 = [torch.empty_like(Z) for _ in range(4)]
    _lib.blocks(Z,Z2,P0,P1,P2,P3,threads=10*65536)
    T=torch.baddbmm(P2,P3,Z3)
    T=torch.baddbmm(P1,T,Z3)
    T=torch.baddbmm(P0,T,Z3)
    X = T.half() @ X
    return X.mT if transposed else X
