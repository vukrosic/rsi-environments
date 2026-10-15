import torch


_source = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void normalize(device const float* g, device const float* norm,
                      device half* x, uint i [[thread_position_in_grid]]) {
    uint b=i/(256*704), r=(i/704)%256, c=i%704;
    x[i]=half(g[b*256*704+c*256+r]/(norm[b]+1e-7f));
}
'''


def _matmul(name, kind):
    n = 704 if kind == 'update' else 256
    kmax = 704 if kind == 'gram' else 256
    a_load = 'a[batch*256*704+(row+rr)*704+kk+t]' if kind == 'gram' else 'a[batch*256*256+(row+rr)*256+kk+t]'
    if kind == 'gram':
        b_load = 'b[batch*256*704+(col+cc)*704+kk+t]'
        b_index = 'uint cc=i/32, t=i%32;'
    else:
        b_load = f'b[batch*256*{n}+(kk+t)*{n}+col+cc]'
        b_index = 'uint t=i/64, cc=i%64;'
    output = 'float v=out[i];'
    if kind == 'poly':
        output += ' v=2.0315f*v-4.7750f*a[batch*65536+(row+rr)*256+col+cc]+((row+rr==col+cc)?3.4445f:0.0f);'
    code = f'''
kernel void {name}(device const half* a, device const half* b, device half* c,
                    uint3 block [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {{
    uint row=block.y*64, col=block.x*64, batch=block.z;
    threadgroup half aa[64*32], bb[32*64];
    threadgroup float out[64*64];
    uint mr=(sg/2)*16, mc=(sg%2)*32;
    simdgroup_float8x8 c00(0), c01(0), c02(0), c03(0), c10(0), c11(0), c12(0), c13(0);
    for(uint kk=0; kk<{kmax}; kk+=32) {{
        for(uint i=tid;i<64*32;i+=256) {{
            uint rr=i/32,t=i%32;
            aa[i]={a_load};
        }}
        for(uint i=tid;i<32*64;i+=256) {{
            {b_index}
            bb[t*64+cc]={b_load};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
'''
    for t in range(0, 32, 8):
        code += f'''
        {{
            simdgroup_half8x8 a0,a1,b0,b1,b2,b3;
            simdgroup_load(a0,aa+mr*32+{t},32);
            simdgroup_load(a1,aa+(mr+8)*32+{t},32);
            simdgroup_load(b0,bb+{t}*64+mc,64);
            simdgroup_load(b1,bb+{t}*64+mc+8,64);
            simdgroup_load(b2,bb+{t}*64+mc+16,64);
            simdgroup_load(b3,bb+{t}*64+mc+24,64);
            simdgroup_multiply_accumulate(c00,a0,b0,c00);
            simdgroup_multiply_accumulate(c01,a0,b1,c01);
            simdgroup_multiply_accumulate(c10,a1,b0,c10);
            simdgroup_multiply_accumulate(c11,a1,b1,c11);
            simdgroup_multiply_accumulate(c02,a0,b2,c02);
            simdgroup_multiply_accumulate(c03,a0,b3,c03);
            simdgroup_multiply_accumulate(c12,a1,b2,c12);
            simdgroup_multiply_accumulate(c13,a1,b3,c13);
        }}
'''
    code += f'''
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    simdgroup_store(c00,out+mr*64+mc,64);
    simdgroup_store(c01,out+mr*64+mc+8,64);
    simdgroup_store(c10,out+(mr+8)*64+mc,64);
    simdgroup_store(c11,out+(mr+8)*64+mc+8,64);
    simdgroup_store(c02,out+mr*64+mc+16,64);
    simdgroup_store(c03,out+mr*64+mc+24,64);
    simdgroup_store(c12,out+(mr+8)*64+mc+16,64);
    simdgroup_store(c13,out+(mr+8)*64+mc+24,64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<64*64;i+=256) {{
        uint rr=i/64,cc=i%64;
        {output}
        c[batch*256*{n}+(row+rr)*{n}+col+cc]=half(v);
    }}
}}
'''
    return code


_lib = torch.mps.compile_shader(_source + _matmul('gram', 'gram') +
                                _matmul('poly', 'poly') + _matmul('update', 'update'))


def kernel(g):
    if g.shape != (10, 704, 256):
        X = g.mT if g.size(-2) > g.size(-1) else g
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(5):
            A = X @ X.mT
            X = 3.4445 * X + (-4.7750 * A + 2.0315 * (A @ A)) @ X
        return X.mT if g.size(-2) > g.size(-1) else X
    norm = g.norm(dim=(-2, -1))
    X = torch.empty((10, 256, 704), device=g.device, dtype=torch.float16)
    Y = torch.empty_like(X)
    A = torch.empty((10, 256, 256), device=g.device, dtype=torch.float16)
    B = torch.empty_like(A)
    _lib.normalize(g, norm, X, threads=(X.numel(),), group_size=(256,))
    for _ in range(5):
        _lib.gram(X, X, A, threads=(4*256, 4, 10), group_size=(256, 1, 1))
        _lib.poly(A, A, B, threads=(4*256, 4, 10), group_size=(256, 1, 1))
        _lib.update(B, X, Y, threads=(11*256, 4, 10), group_size=(256, 1, 1))
        X, Y = Y, X
    return X.mT
