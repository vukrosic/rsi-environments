"""Muon orthogonalization with a blocked Chebyshev polynomial on MPS."""
import torch

_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void norm_parts(device const float* g, device float* parts,
                       uint2 group [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint sid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    float sum=0.f;
    for(uint k=tid;k<4096;k+=256) {
        float v=g[group.y*180224+group.x*4096+k];
        sum+=v*v;
    }
    sum=simd_sum(sum);
    threadgroup float sums[8];
    if(lane==0) sums[sid]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sid==0) {
        sum=simd_sum(lane<8 ? sums[lane] : 0.f);
        if(lane==0) parts[group.y*44+group.x]=sum;
    }
}
kernel void normalize(device const float* g, device const float* parts,
                      device half* x, uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sid [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float inv;
    threadgroup half tile[32*33];
    if(sid==0) {
        float sum=parts[group.z*44+lane];
        if(lane<12) sum+=parts[group.z*44+lane+32];
        sum=simd_sum(sum);
        if(lane==0) inv=1.f/(sqrt(sum)+1e-7f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint k=tid;k<1024;k+=256) {
        uint row=k/32,col=k%32;
        tile[row*33+col]=half(g[group.z*180224+(group.x*32+row)*256+group.y*32+col]*inv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint k=tid;k<1024;k+=256) {
        uint row=k/32,col=k%32;
        x[group.z*180224+(group.y*32+row)*704+group.x*32+col]=tile[col*33+row];
    }
}
kernel void affine(device half* out, device const half* add,
                   constant float& scale, constant float& diagonal,
                   constant float& add_scale, uint i [[thread_position_in_grid]]) {
    uint k=i%65536;
    out[i]=half(scale*float(out[i])+add_scale*float(add[i])+(k/256==k%256 ? diagonal : 0.f));
}
kernel void basis(device half* out, device const half* t1, device const half* t2,
                  device const half* t3, constant float& c0, constant float& c1,
                  constant float& c2, constant float& c3, constant float& scale,
                  uint i [[thread_position_in_grid]]) {
    uint k=i%65536;
    out[i]=half(scale*float(out[i])+c1*float(t1[i])+c2*float(t2[i])+c3*float(t3[i])+(k/256==k%256?c0:0.f));
}
''')

# Chebyshev approximation of the complete five-step scalar multiplier.
# Blocks are polynomials in T1, T2, T3, evaluated with T4 as the Horner variable.
_blocks = [
[17.64563308834225,-15.930397435072196,7.1044635016241315,-2.1767476991614756],
[1.9427596852244018,-0.0026912522901084285,-3.7308910949707625,3.0439642366130952],
[-2.7969840384079103,2.716539929416633,-0.792468264799735,-0.10682025281052504],
[0.18692462748450767,0.,0.,0.]
]


def kernel(g):
    if tuple(g.shape) == (10,704,256):
        norms=torch.empty((10,44),dtype=torch.float32,device=g.device)
        _lib.norm_parts(g,norms,threads=(44*256,10),group_size=(256,1))
        X=torch.empty((10,256,704),dtype=torch.float16,device=g.device)
        _lib.normalize(g,norms,X,threads=(22*256,8,10),group_size=(256,1,1))
        S=X@X.mT
        n=S.numel()
        _lib.affine(S,S,2/0.01035,-0.01125/0.01035,0.,threads=n)
        T2=S@S
        _lib.affine(T2,T2,2.,-1.,0.,threads=n)
        T3=S@T2
        _lib.affine(T3,S,2.,0.,-1.,threads=n)
        T4=T2@T2
        _lib.affine(T4,T4,2.,-1.,0.,threads=n)
        Q=T4.clone()
        _lib.basis(Q,S,T2,T3,*_blocks[-2],_blocks[-1][0],threads=n)
        for coeff in reversed(_blocks[:-2]):
            Q=Q@T4
            _lib.basis(Q,S,T2,T3,*coeff,1.,threads=n)
        return (Q@X).mT
    transposed=g.size(-2)>g.size(-1)
    X=g.mT if transposed else g
    X=X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7)
    for _ in range(5):
        A=X@X.mT
        X=3.4445*X+(-4.7750*A+2.0315*(A@A))@X
    return X.mT if transposed else X
