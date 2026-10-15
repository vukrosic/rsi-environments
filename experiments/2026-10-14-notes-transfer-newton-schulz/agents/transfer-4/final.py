"""Five Newton-Schulz steps using a small Gram matrix and tiled Metal products.

Matrix operands use half precision; reductions and matrix accumulators use float.
"""
import torch


_source = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void norms(device const float* g, device float* norm,
                  uint2 block [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    float acc=0.0f;
    for(uint i=tid;i<5632;i+=256) {
        float v=g[block.y*180224+block.x*5632+i];
        acc+=v*v;
    }
    acc=simd_sum(acc);
    threadgroup float tmp[8];
    if(lane==0) tmp[sg]=acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        acc=simd_sum(lane<8 ? tmp[lane]:0.0f);
        if(lane==0) norm[block.y*32+block.x]=acc;
    }
}

kernel void normalize(device const float* g, device const float* norm,
                      device half* x, uint3 block [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scale;
    threadgroup half tile[32*32];
    if(tid<32) {
        float sum=simd_sum(norm[block.z*32+tid]);
        if(tid==0) scale=1.0f/(sqrt(sum)+1e-7f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<1024;i+=256) {
        uint r=i/32,c=i%32;
        tile[i]=half(g[block.z*180224+(block.x*32+r)*256+block.y*32+c]*scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<1024;i+=256) {
        uint r=i/32,c=i%32;
        x[block.z*180224+(block.y*32+r)*704+block.x*32+c]=tile[c*32+r];
    }
}
'''


def _matmul(name, kind):
    n = 704 if kind == 'update' else 256
    kmax = 704 if kind == 'gram' else 256
    a_load = 'a[batch*256*704+(row+rr)*704+kk+t]' if kind == 'gram' else 'a[batch*256*256+(row+rr)*256+kk+t]'
    if kind == 'gram':
        b_load = 'b[batch*256*704+(col+cc)*704+kk+t]'
        b_index = 'uint cc=i/16, t=(i%16)*4;'
        b_store = 'bb+cc*64+t'
    else:
        b_load = f'b[batch*256*{n}+(kk+t)*{n}+col+cc]'
        b_index = 'uint t=i/16, cc=(i%16)*4;'
        b_store = 'bb+t*64+cc'
    skip = 'if(block.x>block.y) return;' if kind != 'update' else ''
    code = f'''
kernel void {name}(device const half* a, device const half* b, device half* c,
                    uint3 block [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {{
    uint row=block.y*64, col=block.x*64, batch=block.z;
    {skip}
    threadgroup half aa[64*64], bb[64*64];
    uint mr=(sg/2)*32, mc=(sg%2)*32;
    simdgroup_float8x8 c00(0), c01(0), c02(0), c03(0), c10(0), c11(0), c12(0), c13(0), c20(0), c21(0), c22(0), c23(0), c30(0), c31(0), c32(0), c33(0);
    for(uint kk=0; kk<{kmax}; kk+=64) {{
        for(uint i=tid;i<64*16;i+=128) {{
            uint rr=i/16,t=(i%16)*4;
            *((threadgroup half4*)(aa+rr*64+t))=*((device const half4*)(&{a_load}));
        }}
        for(uint i=tid;i<64*16;i+=128) {{
            {b_index}
            *((threadgroup half4*)({b_store}))=*((device const half4*)(&{b_load}));
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
'''
    for t in range(0, 64, 8):
        code += "        {\n"
        for r in range(4):
            code += f"            simdgroup_half8x8 a{r}; simdgroup_load(a{r},aa+(mr+{r*8})*64+{t},64);\n"
        for cpart in range(4):
            code += f"            simdgroup_half8x8 b{cpart}; simdgroup_load(b{cpart},bb+{t}*64+mc+{cpart*8},64);\n"
        for r in range(4):
            for cpart in range(4):
                code += f"            simdgroup_multiply_accumulate(c{r}{cpart},a{r},b{cpart},c{r}{cpart});\n"
        code += "        }\n"
    if kind == 'gram':
        # Stage both Gram operands contiguously, then transpose the second load.
        for t in range(0,64,8):
            code = code.replace(f'bb+{t}*64+mc+0,64', f'bb+mc*64+{t},64,ulong2(0),true')
            for colpart in range(1,4):
                code = code.replace(f'bb+{t}*64+mc+{colpart*8},64', f'bb+(mc+{colpart*8})*64+{t},64,ulong2(0),true')
    code += "        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n"
    for r in range(4):
        for colpart in range(4):
            acc=f"c{r}{colpart}"
            code += f"    {{ simdgroup_half8x8 h; h.thread_elements()[0]=half({acc}.thread_elements()[0]); h.thread_elements()[1]=half({acc}.thread_elements()[1]);\n"
            if kind == "poly":
                code += f"        uint rr=row+mr+{r*8}+(tid%32)/16*4+((tid%32)%8)/2; uint cc=col+mc+{colpart*8}+((tid%32)/8)%2*4+((tid%32)%2)*2;\n"
                for elem in range(2):
                    code += f"        h.thread_elements()[{elem}]=half(2.0315f*{acc}.thread_elements()[{elem}]-4.7750f*a[batch*65536+rr*256+cc+{elem}]+(rr==cc+{elem}?3.4445f:0.0f));\n"
            code += f"        simdgroup_store(h,c+batch*256*{n}+(row+mr+{r*8})*{n}+col+mc+{colpart*8},{n});\n"
            if kind != 'update':
                code += f"        if(block.x!=block.y) simdgroup_store(h,c+batch*65536+(col+mc+{colpart*8})*256+row+mr+{r*8},256,ulong2(0),true);\n"
            code += "    }\n"
    code += "}\n"
    return code


_lib = torch.mps.compile_shader(_source + _matmul('gram', 'gram') +
                                _matmul('poly', 'poly') + _matmul('product', 'product') + _matmul('update', 'update'))


def kernel(g):
    if g.shape != (10, 704, 256) or g.dtype != torch.float32 or not g.is_contiguous():
        X = g.mT if g.size(-2) > g.size(-1) else g
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(5):
            A = X @ X.mT
            X = 3.4445 * X + (-4.7750 * A + 2.0315 * (A @ A)) @ X
        return X.mT if g.size(-2) > g.size(-1) else X
    norm = torch.empty((10, 32), device=g.device, dtype=torch.float32)
    X = torch.empty((10, 256, 704), device=g.device, dtype=torch.float16)
    Y = torch.empty_like(X)
    A = torch.empty((10, 256, 256), device=g.device, dtype=torch.float16)
    B = torch.empty_like(A)
    _lib.norms(g, norm, threads=(32*256, 10), group_size=(256, 1))
    _lib.normalize(g, norm, X, threads=(22*256, 8, 10), group_size=(256, 1, 1))
    P = torch.empty_like(A)
    Q = torch.empty_like(A)
    T = torch.empty_like(A)
    small = dict(threads=(4*128, 4, 10), group_size=(128, 1, 1))
    _lib.gram(X, X, A, **small)
    # B = aI + bA + cA². Evolve A <- B A B and P <- B P;
    # apply P to the normalized input only once, after all five iterations.
    # These matrices are commuting polynomials of the initial Gram matrix.
    for iteration in range(5):
        if iteration == 0:
            _lib.poly(A, A, P, **small)
            current = P
        else:
            _lib.poly(A, A, B, **small)
            _lib.product(P, B, Q, **small)
            P, Q = Q, P
            current = B
        if iteration != 4:
            _lib.product(A, current, T, **small)
            _lib.product(T, current, A, **small)
    _lib.update(P, X, Y, threads=(11*128, 4, 10), group_size=(128, 1, 1))
    return Y.mT
