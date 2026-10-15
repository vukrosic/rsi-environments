"""Muon Newton-Schulz orthogonalization on MPS."""
import torch

_source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void norm_partials(device const float* g,device float* parts,
uint t [[thread_index_in_threadgroup]],uint s [[simdgroup_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],uint p [[threadgroup_position_in_grid]]) {
    threadgroup float sums[8];
    float v=0;
    for(uint i=t;i<5632;i+=256) {float a=g[p*5632+i];v+=a*a;}
    v=simd_sum(v);
    if(lane==0) sums[s]=v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(s==0) {
        v=simd_sum(lane<8?sums[lane]:0.0f);
        if(lane==0) parts[p]=v;
    }
}
kernel void normalize_tiled(device const float* g, device const float* norm, device half* x,
uint t [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint3 pos [[threadgroup_position_in_grid]]) {
    threadgroup half tile[32*33];
    float scale=1.0f/(sqrt(simd_sum(norm[pos.z*32+lane]))+1.e-7f);
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
"""

def _square(name, polynomial=False):
    declarations=[]
    multiply=[]
    stores=[]
    for r in range(2):
        for c in range(2):
            declarations.append(f"simdgroup_float8x8 o{r}{c}(0.0f);")
            multiply.append(f"simdgroup_multiply_accumulate(o{r}{c},a{r},b{c},o{r}{c});")
            formula = f"2.0315f*o{r}{c}.thread_elements()[j]-4.775f*float(A[off+row*256+col])+(row==col?3.4445f:0.0f)" if polynomial else f"o{r}{c}.thread_elements()[j]"
            stores.append(f"for(uint j=0;j<2;j++) {{uint row=baseR+sr+{r*8}+((lane>>1)&3)+(lane/16)*4,col=baseC+sc+{c*8}+(lane%2)*2+((lane/8)%2)*4+j;half val=half({formula});D[off+row*256+col]=val;if(pos.y!=pos.x) D[off+col*256+row]=val;}}")
    return f"""
kernel void {name}(device const half* A,device const half* B,device half* D,
uint t [[thread_index_in_threadgroup]],uint s [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint3 pos [[threadgroup_position_in_grid]]) {{
if(pos.y>pos.x) return;
threadgroup half aa[1024],bb[1024];
uint off=pos.z*65536,baseR=pos.y*32,baseC=pos.x*32;
uint sr=(s/2)*16,sc=(s%2)*16;
{' '.join(declarations)}
for(uint bk=0;bk<256;bk+=32) {{
for(uint i=t;i<1024;i+=128) {{
aa[i]=A[off+(baseR+i/32)*256+bk+i%32];
bb[i]=B[off+(bk+i/32)*256+baseC+i%32];
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
_lib = torch.mps.compile_shader(_source)

def kernel(g):
    x=torch.empty((10,256,704),device=g.device,dtype=torch.float16)
    norm=torch.empty((10,32),device=g.device,dtype=torch.float32)
    _lib.norm_partials(g,norm,threads=320*256,group_size=256)
    _lib.normalize_tiled(g,norm,x,threads=(8*256,22,10),group_size=(256,1,1))
    # Evolve the Gram matrix and its accumulated polynomial transform.
    # These polynomials commute, so their products can use symmetry.
    a=x @ x.mT
    b=torch.empty_like(a)
    _lib.polynomial(a,a,b,threads=(8*128,8,10),group_size=(128,1,1))
    p=b
    tmp=torch.empty_like(a)
    an=torch.empty_like(a)
    bn=torch.empty_like(a)
    for _ in range(4):
        pn=torch.empty_like(a)
        _lib.square(b,a,tmp,threads=(8*128,8,10),group_size=(128,1,1))
        _lib.square(tmp,b,an,threads=(8*128,8,10),group_size=(128,1,1))
        _lib.polynomial(an,an,bn,threads=(8*128,8,10),group_size=(128,1,1))
        _lib.square(bn,p,pn,threads=(8*128,8,10),group_size=(128,1,1))
        a,an=an,a
        b,bn=bn,b
        p=pn
    return (p@x).mT
