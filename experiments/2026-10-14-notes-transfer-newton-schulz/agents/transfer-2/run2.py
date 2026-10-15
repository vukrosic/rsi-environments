import torch

_SHADER = r"""
#include <metal_stdlib>
using namespace metal;

kernel void norm_partial(device const float *g, device float *partial,
    uint t [[thread_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
    uint b = group.y, start = group.x * 1024 + t;
    float v = 0;
    for (uint j = 0; j < 4; j++) { float x = g[b*180224+start+j*256]; v += x*x; }
    v = simd_sum(v);
    threadgroup float sums[8];
    if ((t & 31) == 0) sums[t/32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 32) { v = simd_sum(t < 8 ? sums[t] : 0.0f); if (t == 0) partial[b*176+group.x] = v; }
}

kernel void normalize(device const float *g, device const float *partial, device half *x,
    uint t [[thread_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
    uint b = group.y;
    float v = t < 176 ? partial[b*176+t] : 0;
    v = simd_sum(v);
    threadgroup float sums[8];
    if ((t & 31) == 0) sums[t/32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 32) { v = simd_sum(t < 8 ? sums[t] : 0.0f); if (t == 0) sums[0] = 1.0f/(sqrt(v)+1e-7f); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = sums[0];
    for (uint j = 0; j < 4; j++) {
        uint i = group.x*1024+t+j*256;
        x[b*180224+i] = half(g[b*180224+(i%704)*256+i/704]*scale);
    }
}
"""


def _matmul(name, k_size, n_size, gram=False, polynomial=False, update=False):
    load_a = "x[batch*180224 + (row+r)*704 + kk+c]" if gram else "x[batch*65536 + (row+r)*256 + kk+c]"
    load_b = "x[batch*180224 + (col+c)*704 + kk+r]" if gram else ("y[batch*180224 + (kk+r)*704+col+c]" if update else "x[batch*65536 + (kk+r)*256+col+c]")
    ops = []
    for k in range(4):
        ops.append(f"simdgroup_load(a0, as + sr*32 + {k*8}, 32); simdgroup_load(a1, as + (sr+8)*32 + {k*8}, 32);")
        ops.append(f"simdgroup_load(b0, bs + {k*8}*32 + sc, 32); simdgroup_load(b1, bs + {k*8}*32 + sc+8, 32);")
        for i in range(2):
            for j in range(2):
                ops.append(f"simdgroup_multiply_accumulate(c{i}{j}, a{i}, b{j}, c{i}{j});")
    epilogue = "out[batch*"+str(256*n_size)+"+(row+r)*"+str(n_size)+"+col+c] = half(cs[r*32+c]);"
    if polynomial:
        epilogue = "out[batch*65536+(row+r)*256+col+c] = half(2.0315f*cs[r*32+c]-4.7750f*float(x[batch*65536+(row+r)*256+col+c]) + ((row+r == col+c) ? 3.4445f : 0.0f));"
    return f"""
kernel void {name}(device const half *x, device const half *y, device half *out,
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {{
    uint row=group.y*32, col=group.x*32, batch=group.z;
    threadgroup half as[1024], bs[1024];
    threadgroup float cs[1024];
    uint sr=(sg/2)*16, sc=(sg%2)*16;
    simdgroup_float8x8 c00(0), c01(0), c10(0), c11(0);
    simdgroup_half8x8 a0, a1, b0, b1;
    for (uint kk=0; kk<{k_size}; kk+=32) {{
        for (uint i=t; i<1024; i+=128) {{
            uint r=i/32, c=i%32;
            as[i]={load_a}; bs[i]={load_b};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {chr(10).join(ops)}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    simdgroup_store(c00, cs+sr*32+sc, 32); simdgroup_store(c01, cs+sr*32+sc+8, 32);
    simdgroup_store(c10, cs+(sr+8)*32+sc, 32); simdgroup_store(c11, cs+(sr+8)*32+sc+8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=t; i<1024; i+=128) {{ uint r=i/32,c=i%32; {epilogue} }}
}}
"""

_SHADER += _matmul("gram", 704, 256, gram=True)
_SHADER += _matmul("poly", 256, 256, polynomial=True)
_SHADER += _matmul("update", 256, 704, update=True)
_LIB = torch.mps.compile_shader(_SHADER)


def kernel(g):
    if g.shape != (10, 704, 256) or g.dtype != torch.float32 or not g.is_contiguous():
        X = g.mT if g.size(-2) > g.size(-1) else g
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(5):
            A = X @ X.mT
            X = 3.4445*X + (-4.7750*A + 2.0315*(A@A))@X
        return X.mT if g.size(-2) > g.size(-1) else X
    partial = torch.empty((10,176),device=g.device)
    x = torch.empty((10,256,704),device=g.device,dtype=torch.float16)
    z = torch.empty_like(x)
    a = torch.empty((10,256,256),device=g.device,dtype=torch.float16)
    b = torch.empty_like(a)
    _LIB.norm_partial(g,partial,threads=(176*256,10,1),group_size=(256,1,1))
    _LIB.normalize(g,partial,x,threads=(176*256,10,1),group_size=(256,1,1))
    for _ in range(5):
        _LIB.gram(x,x,a,threads=(8*128,8,10),group_size=(128,1,1))
        _LIB.poly(a,a,b,threads=(8*128,8,10),group_size=(128,1,1))
        _LIB.update(b,x,z,threads=(22*128,8,10),group_size=(128,1,1))
        x,z=z,x
    return x.mT
