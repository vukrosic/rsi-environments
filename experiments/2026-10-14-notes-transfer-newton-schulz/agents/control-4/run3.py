"""Muon Newton-Schulz orthogonalization on MPS."""
import torch

_source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void normalize(device const float* g, device half* x,
                      uint t [[thread_index_in_threadgroup]],
                      uint s [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint b [[threadgroup_position_in_grid]]) {
    threadgroup float sums[8];
    float v=0;
    for (uint i=t;i<180224;i+=256) { float a=g[b*180224+i]; v+=a*a; }
    v=simd_sum(v);
    if(lane==0) sums[s]=v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(s==0) {
        v=simd_sum(lane<8?sums[lane]:0.0f);
        if(lane==0) sums[0]=1.0f/(sqrt(v)+1.e-7f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale=sums[0];
    for(uint i=t;i<180224;i+=256) {
        uint r=i/704,c=i%704;
        x[b*180224+i]=half(g[b*180224+c*256+r]*scale);
    }
}
"""

def _matmul(name, n, k, transpose, alpha, beta):
    bload = "B[batch*256*704+col*704+kk]" if transpose else f"B[batch*{k*n}+kk*{n}+col]"
    extra = f"+{beta}f*float(C[batch*256*{n}+row*{n}+col])" if beta else ""
    return f"""
kernel void {name}(device const half* A,device const half* B,device const half* C,device half* D,
                   uint t [[thread_index_in_threadgroup]],
                   uint s [[simdgroup_index_in_threadgroup]],
                   uint3 pos [[threadgroup_position_in_grid]]) {{
    threadgroup half aa[1024],bb[1024];
    threadgroup float oo[1024];
    uint batch=pos.z, baseR=pos.y*32,baseC=pos.x*32;
    uint sr=(s/2)*16,sc=(s%2)*16;
    simdgroup_float8x8 o00(0.0f),o01(0.0f),o10(0.0f),o11(0.0f);
    for(uint baseK=0;baseK<{k};baseK+=32) {{
        for(uint i=t;i<1024;i+=128) {{
            uint row=baseR+i/32,kk=baseK+i%32;
            aa[i]=A[batch*256*{k}+row*{k}+kk];
            kk=baseK+i/32;uint col=baseC+i%32;
            bb[i]={bload};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<32;j+=8) {{
            simdgroup_half8x8 a0,a1,b0,b1;
            simdgroup_load(a0,aa+sr*32+j,32);
            simdgroup_load(a1,aa+(sr+8)*32+j,32);
            simdgroup_load(b0,bb+j*32+sc,32);
            simdgroup_load(b1,bb+j*32+sc+8,32);
            simdgroup_multiply_accumulate(o00,a0,b0,o00);
            simdgroup_multiply_accumulate(o01,a0,b1,o01);
            simdgroup_multiply_accumulate(o10,a1,b0,o10);
            simdgroup_multiply_accumulate(o11,a1,b1,o11);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    simdgroup_store(o00,oo+sr*32+sc,32);
    simdgroup_store(o01,oo+sr*32+sc+8,32);
    simdgroup_store(o10,oo+(sr+8)*32+sc,32);
    simdgroup_store(o11,oo+(sr+8)*32+sc+8,32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=t;i<1024;i+=128) {{
        uint row=baseR+i/32,col=baseC+i%32;
        D[batch*256*{n}+row*{n}+col]=half({alpha}f*oo[i]{extra});
    }}
}}
"""

_source += _matmul("gram",256,704,True,1.0,0)
_source += _matmul("polynomial",256,256,False,2.0315,-4.7750)
_source += _matmul("update",704,256,False,1.0,3.4445)
_lib = torch.mps.compile_shader(_source)

def kernel(g):
    x=torch.empty((10,256,704),device=g.device,dtype=torch.float16)
    _lib.normalize(g,x,threads=2560,group_size=256)
    a=x @ x.mT
    eye=torch.eye(256,device=g.device,dtype=torch.float16).expand(10,-1,-1)
    b=torch.baddbmm(a,a,a,beta=-4.7750,alpha=2.0315)+3.4445*eye
    p=b
    for _ in range(4):
        a=(b@a)@b
        b=torch.baddbmm(a,a,a,beta=-4.7750,alpha=2.0315)+3.4445*eye
        p=b@p
    return (p@x).mT
