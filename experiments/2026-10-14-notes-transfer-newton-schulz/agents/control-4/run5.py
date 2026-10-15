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

def _square(name, polynomial=False):
    declarations=[]
    multiply=[]
    stores=[]
    for r in range(2):
        for c in range(4):
            declarations.append(f"simdgroup_float8x8 o{r}{c}(0.0f);")
            multiply.append(f"simdgroup_multiply_accumulate(o{r}{c},a{r},b{c},o{r}{c});")
            formula = f"2.0315f*o{r}{c}.thread_elements()[j]-4.775f*float(A[off+row*256+col])+(row==col?3.4445f:0.0f)" if polynomial else f"o{r}{c}.thread_elements()[j]"
            stores.append(f"for(uint j=0;j<2;j++) {{uint row=baseR+sr+{r*8}+((lane>>1)&3)+(lane/16)*4,col=baseC+sc+{c*8}+(lane%2)*2+((lane/8)%2)*4+j;D[off+row*256+col]=half({formula});}}")
    formula="2.0315f*oo[i]-4.775f*float(A[off+row*256+col])+(row==col?3.4445f:0.0f)" if polynomial else "oo[i]"
    return f"""
kernel void {name}(device const half* A,device const half* B,device half* D,
uint t [[thread_index_in_threadgroup]],uint s [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint3 pos [[threadgroup_position_in_grid]]) {{
threadgroup half aa[2048],bb[2048];
uint off=pos.z*65536,baseR=pos.y*64,baseC=pos.x*64;
uint sr=(s/2)*16,sc=(s%2)*32;
{' '.join(declarations)}
for(uint bk=0;bk<256;bk+=32) {{
for(uint i=t;i<2048;i+=256) {{
aa[i]=A[off+(baseR+i/32)*256+bk+i%32];
bb[i]=B[off+(bk+i/64)*256+baseC+i%64];
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint j=0;j<32;j+=8) {{
simdgroup_half8x8 a0,a1,b0,b1,b2,b3;
simdgroup_load(a0,aa+sr*32+j,32);simdgroup_load(a1,aa+(sr+8)*32+j,32);
simdgroup_load(b0,bb+j*64+sc,64);simdgroup_load(b1,bb+j*64+sc+8,64);
simdgroup_load(b2,bb+j*64+sc+16,64);simdgroup_load(b3,bb+j*64+sc+24,64);
{' '.join(multiply)}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
}}
{' '.join(stores)}
}}
"""

_source += _square("square")
_source += _square("polynomial",True)
_lib = torch.mps.compile_shader(_source)

def kernel(g):
    x=torch.empty((10,256,704),device=g.device,dtype=torch.float16)
    _lib.normalize(g,x,threads=2560,group_size=256)
    a=x @ x.mT
    b=torch.empty_like(a)
    _lib.polynomial(a,a,b,threads=(4*256,4,10),group_size=(256,1,1))
    p=b
    for _ in range(4):
        tmp=torch.empty_like(a)
        an=torch.empty_like(a)
        bn=torch.empty_like(a)
        pn=torch.empty_like(a)
        _lib.square(b,a,tmp,threads=(4*256,4,10),group_size=(256,1,1))
        _lib.square(tmp,b,an,threads=(4*256,4,10),group_size=(256,1,1))
        _lib.polynomial(an,an,bn,threads=(4*256,4,10),group_size=(256,1,1))
        _lib.square(bn,p,pn,threads=(4*256,4,10),group_size=(256,1,1))
        a,b,p=an,bn,pn
    return (p@x).mT
