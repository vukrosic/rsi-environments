"""Muon Newton-Schulz orthogonalization on MPS."""
import torch

_source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void normalize_tiled(device const float* g, device const float* norm, device half* x,
uint t [[thread_index_in_threadgroup]],uint3 pos [[threadgroup_position_in_grid]]) {
    threadgroup half tile[32*33];
    float scale=1.0f/(norm[pos.z]+1.e-7f);
    for(uint i=t;i<1024;i+=256) {
        uint r=pos.y*32+i/32,c=pos.x*32+i%32;
        tile[(i/32)*33+i%32]=half(g[pos.z*180224+r*256+c]*scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=t;i<1024;i+=256) {
        uint r=pos.x*32+i/32,c=pos.y*32+i%32;
        x[pos.z*180224+r*704+c]=tile[(i%32)*33+i/32];
    }
}
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

def _square(name, polynomial=False, gram=False):
    k = 704 if gram else 256
    aload = f"A[pos.z*{256*k}+(baseR+i/32)*{k}+bk+i%32]"
    bload = f"bb[(i%32)*32+i/32]=B[pos.z*{256*k}+(baseC+i/32)*{k}+bk+i%32];" if gram else "bb[i]=B[off+(bk+i/32)*256+baseC+i%32];"
    declarations=[]
    multiply=[]
    stores=[]
    for r in range(2):
        for c in range(2):
            declarations.append(f"simdgroup_float8x8 o{r}{c}(0.0f);")
            multiply.append(f"simdgroup_multiply_accumulate(o{r}{c},a{r},b{c},o{r}{c});")
            formula = f"2.0315f*o{r}{c}.thread_elements()[j]-4.775f*float(A[off+row*256+col])+(row==col?3.4445f:0.0f)" if polynomial else f"o{r}{c}.thread_elements()[j]"
            stores.append(f"for(uint j=0;j<2;j++) {{uint row=baseR+sr+{r*8}+((lane>>1)&3)+(lane/16)*4,col=baseC+sc+{c*8}+(lane%2)*2+((lane/8)%2)*4+j;half val=half({formula});D[off+row*256+col]=val;if(pos.y!=pos.x) D[off+col*256+row]=val;}}")
    formula="2.0315f*oo[i]-4.775f*float(A[off+row*256+col])+(row==col?3.4445f:0.0f)" if polynomial else "oo[i]"
    return f"""
kernel void {name}(device const half* A,device const half* B,device half* D,
uint t [[thread_index_in_threadgroup]],uint s [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint3 pos [[threadgroup_position_in_grid]]) {{
if(pos.y>pos.x) return;
threadgroup half aa[1024],bb[1024];
uint off=pos.z*65536,baseR=pos.y*32,baseC=pos.x*32;
uint sr=(s/2)*16,sc=(s%2)*16;
{' '.join(declarations)}
for(uint bk=0;bk<{k};bk+=32) {{
for(uint i=t;i<1024;i+=128) {{
aa[i]={aload};
{bload}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint j=0;j<32;j+=8) {{
simdgroup_half8x8 a0,a1,b0,b1;
simdgroup_load(a0,aa+sr*32+j,32);simdgroup_load(a1,aa+(sr+8)*32+j,32);
simdgroup_load(b0,bb+j*32+sc,32);simdgroup_load(b1,bb+j*32+sc+8,32);

{' '.join(multiply)}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
}}
{' '.join(stores)}
}}
"""

_source += _square("square")
_source += _square("polynomial",True)
_source += _square("gram",gram=True)
_lib = torch.mps.compile_shader(_source)

def kernel(g):
    x=torch.empty((10,256,704),device=g.device,dtype=torch.float16)
    _lib.normalize(g,x,threads=2560,group_size=256)
    a=torch.empty((10,256,256),device=g.device,dtype=torch.float16)
    _lib.gram(x,x,a,threads=(8*128,8,10),group_size=(128,1,1))
    b=torch.empty_like(a)
    _lib.polynomial(a,a,b,threads=(8*128,8,10),group_size=(128,1,1))
    p=b
    for _ in range(4):
        tmp=torch.empty_like(a)
        an=torch.empty_like(a)
        bn=torch.empty_like(a)
        pn=torch.empty_like(a)
        _lib.square(b,a,tmp,threads=(8*128,8,10),group_size=(128,1,1))
        _lib.square(tmp,b,an,threads=(8*128,8,10),group_size=(128,1,1))
        _lib.polynomial(an,an,bn,threads=(8*128,8,10),group_size=(128,1,1))
        _lib.square(bn,p,pn,threads=(8*128,8,10),group_size=(128,1,1))
        a,b,p=an,bn,pn
    return (p@x).mT
