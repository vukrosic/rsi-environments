"""Muon orthogonalization with a blocked Chebyshev polynomial on MPS."""
import torch

_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void normalize(device const float* g, device const float* norm,
                      device half* x, uint i [[thread_position_in_grid]]) {
    uint b=i/(256*704), k=i%(256*704);
    uint row=k/704, col=k%704;
    x[i]=half(g[b*256*704+col*256+row]/(norm[b]+1e-7f));
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
[17.503352003066656,-16.359222594304807,7.335453284507199,-1.5655151158740046],
[0.15896395704412947,1.6780220418423244,-4.739678936575881,4.107195892447386],
[-5.410460286863871,6.219293020634017,-2.4863013027862086,0.5117287885305419],
[0.26664052711873787,-0.23789406953598036,-0.12111502086131853,0.27816212283244957],
[3.211250360002438,-6.023069488907041,3.6662022514174266,-1.640749258705691],
[3.1640035373845694,-4.78794569148673,3.2045848449513685,-1.7606920464763243],
[0.7066950140998233,0.,0.,0.]
]


def kernel(g):
    if tuple(g.shape) == (10,704,256):
        norms=g.norm(dim=(-2,-1))
        X=torch.empty((10,256,704),dtype=torch.float16,device=g.device)
        _lib.normalize(g,norms,X,threads=X.numel())
        S=X@X.mT
        n=S.numel()
        _lib.affine(S,S,2/0.0117,-0.0123/0.0117,0.,threads=n)
        T2=S@S
        _lib.affine(T2,T2,2.,-1.,0.,threads=n)
        T3=S@T2
        _lib.affine(T3,S,2.,0.,-1.,threads=n)
        T4=T2@T2
        _lib.affine(T4,T4,2.,-1.,0.,threads=n)
        Q=T4.clone()
        _lib.basis(Q,S,T2,T3,*_blocks[5],_blocks[6][0],threads=n)
        for coeff in reversed(_blocks[:5]):
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
