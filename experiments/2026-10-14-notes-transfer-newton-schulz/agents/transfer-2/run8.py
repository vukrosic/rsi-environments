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
    float scale = 16.0f*sums[0];
    for (uint j = 0; j < 4; j++) {
        uint i = group.x*1024+t+j*256;
        x[b*180224+i] = half(g[b*180224+(i%704)*256+i/704]*scale);
    }
}
"""


def _matmul(name, k_size, n_size, gram=False, polynomial=False, update=False, prefix_square=False, prefix_poly=False):
    m = n = 64
    threads = 256
    load_a = "x[batch*180224 + (row+r)*704 + kk+c]" if gram else "x[batch*65536 + (row+r)*256 + kk+c]"
    load_b = "x[batch*180224 + (col+r)*704 + kk+c]" if gram else ("y[batch*180224 + (kk+r)*704+col+c]" if update else "x[batch*65536 + (col+r)*256+kk+c]")
    if prefix_poly:
        load_b = "y[batch*65536 + (col+r)*256+kk+c]"
    ops = []
    for k in range(8):
        for i in range(4):
            ops.append(f"simdgroup_load(a{i}, as + (sr+{i*8})*64 + {k*8}, 64);")
        for j in range(4):
            if update:
                ops.append(f"simdgroup_load(b{j}, bs + {k*8}*64 + sc+{j*8}, 64);")
            else:
                ops.append(f"simdgroup_load(b{j}, bs + (sc+{j*8})*64 + {k*8}, 64, ulong2(0), true);")
        for i in range(4):
            for j in range(4):
                ops.append(f"simdgroup_multiply_accumulate(c{i}{j}, a{i}, b{j}, c{i}{j});")
    stores = []
    for i in range(4):
        for j in range(4):
            value = f"c{i}{j}.thread_elements()[e]"
            if polynomial:
                value = f"2.0315f*({value})-4.7750f*float(x[batch*65536+rr*256+cc])+(rr == cc ? 3.4445f : 0.0f)"
            if prefix_poly:
                value = f"({value})-2.10837867f*float(aux[batch*65536+rr*256+cc])+(rr == cc ? 2.55261219f : 0.0f)"
            extra = ""
            if prefix_square:
                extra = f"half d=half(0.01346927f*({value})-0.1620066f*float(x[batch*65536+rr*256+cc])+(rr == cc ? 0.82790153f : 0.0f)); aux[batch*65536+rr*256+cc]=d; if (row != col) aux[batch*65536+cc*256+rr]=d;"
            mirror = "if (row != col) out[batch*65536+cc*256+rr]=v;" if not update else ""
            stores.append(f"for (uint e=0;e<2;e++) {{ uint rr=row+sr+lr+{i*8}, cc=col+sc+lc+{j*8}+e; half v=half({value}); out[batch*{256*n_size}+rr*{n_size}+cc]=v; {mirror} {extra} }}")
    accs = ", ".join(f"c{i}{j}(0)" for i in range(4) for j in range(4))
    return f"""
kernel void {name}(device const half *x, device const half *y, device half *out, device half *aux,
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {{
    uint row=group.y*64, col=group.x*64, batch=group.z;
    {chr(105)+chr(102)+" (row > col) return;" if not update else ""}
    threadgroup half as[4096], bs[4096];
    uint sr=(sg/2)*32, sc=(sg%2)*32;
    simdgroup_float8x8 {accs};
    simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3;
    for (uint kk=0; kk<{k_size}; kk+=64) {{
        for (uint i=t*4; i<4096; i+=512) {{
            uint r=i/64, c=i%64;
            *reinterpret_cast<threadgroup half4*>(as+i)=*reinterpret_cast<device const half4*>(&{load_a});
        }}
        for (uint i=t*4; i<4096; i+=512) {{
            uint r=i/{64}, c=i%{64};
            *reinterpret_cast<threadgroup half4*>(bs+i)=*reinterpret_cast<device const half4*>(&{load_b});
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {chr(10).join(ops)}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    uint lane=t%32, lr=(lane/2)%4+(lane/16)*4, lc=(lane%2)*2+((lane/8)%2)*4;
    {chr(10).join(stores)}
}}
"""

_SHADER += _matmul("gram", 704, 256, gram=True)
_SHADER += _matmul("poly", 256, 256, polynomial=True)
_SHADER += _matmul("update", 256, 704, update=True)
_SHADER += _matmul("prefix_square", 256, 256, prefix_square=True)
_SHADER += _matmul("prefix_poly", 256, 256, prefix_poly=True)
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
    h = torch.empty_like(a)
    d = torch.empty_like(a)
    _LIB.gram(x,x,a,a,threads=(4*128,4,10),group_size=(128,1,1))
    _LIB.prefix_square(a,a,h,d,threads=(4*128,4,10),group_size=(128,1,1))
    _LIB.prefix_poly(h,d,b,a,threads=(4*128,4,10),group_size=(128,1,1))
    _LIB.update(b,x,z,z,threads=(11*128,4,10),group_size=(128,1,1))
    x,z=z,x
    for _ in range(2):
        _LIB.gram(x,x,a,a,threads=(4*128,4,10),group_size=(128,1,1))
        _LIB.poly(a,a,b,b,threads=(4*128,4,10),group_size=(128,1,1))
        _LIB.update(b,x,z,z,threads=(11*128,4,10),group_size=(128,1,1))
        x,z=z,x
    return x.mT
